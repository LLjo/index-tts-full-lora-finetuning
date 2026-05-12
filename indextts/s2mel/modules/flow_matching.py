from abc import ABC

import torch
import torch.nn.functional as F

from indextts.s2mel.modules.diffusion_transformer import DiT
from indextts.s2mel.modules.commons import sequence_mask

from tqdm import tqdm

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, solver_type="euler"):
        """Forward diffusion

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
            f0: None
            n_timesteps (int): number of diffusion steps (ignored when solver_type="single_step").
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            solver_type (str):
                - "euler" — 1st-order, 1 estimator call/step.
                - "heun"  — 2nd-order predictor-corrector, 2 estimator calls/step.
                - "single_step" — ONE estimator call total. Only meaningful for a
                  distilled (reflow / consistency) student where the learned flow
                  from z to x_1 is approximately straight, so x_1 ≈ z + v(z, t=0).
                  Will produce garbage with the unmodified teacher.

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, 80, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        if solver_type == "single_step":
            return self.solve_single_step(z, x_lens, prompt, mu, style, f0, inference_cfg_rate)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        if solver_type == "heun":
            return self.solve_heun(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def _estimate(self, x, prompt_x, x_lens, t, style, mu, inference_cfg_rate):
        """One vector-field evaluation, with optional batched classifier-free guidance."""
        if inference_cfg_rate > 0:
            stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
            stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
            stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
            stacked_x = torch.cat([x, x], dim=0)
            stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)
            stacked_dphi_dt = self.estimator(
                stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
            )
            dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
            return (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
        return self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

    def solve_single_step(self, x, x_lens, prompt, mu, style, f0, inference_cfg_rate=0.0):
        """Single-step ODE solve for distilled students (Phase 3).

        Theory: in flow matching the training target is u = x_1 - z (with sigma_min ≈ 0).
        After reflow/consistency distillation the learned vector field is approximately
        constant along the trajectory, so evaluating it once at t=0 and integrating
        the full unit interval gives x_1 ≈ z + v(z, t=0). One estimator call, no loop.

        With the unmodified teacher the field is NOT straight and this produces noise —
        this method is only meaningful behind a distilled checkpoint.

        CFG defaults to 0 because the conventional teacher CFG mechanism is incompatible
        with single-step distilled inference (the student typically absorbs CFG behavior
        during training). Override at your own risk.
        """
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        # t must be a 0-dim scalar tensor — `_estimate` calls t.unsqueeze(0) once,
        # producing shape [1] which is what the timestep embedder expects. Passing a
        # 1-d tensor here causes an extra dim to ripple through the DiT transformer
        # (you see it surface as `too many values to unpack` in attention).
        t = torch.zeros((), device=x.device, dtype=x.dtype)
        v = self._estimate(x, prompt_x, x_lens, t, style, mu, inference_cfg_rate)
        x_final = x + v
        x_final[:, :, :prompt_len] = 0
        return x_final

    def solve_heun(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        """Heun's 2nd-order ODE solver (improved Euler / trapezoidal predictor-corrector).

        Per step:
            k1 = f(t,         x)
            k2 = f(t + dt,    x + dt * k1)
            x_new = x + dt * (k1 + k2) / 2

        Cost is 2 estimator calls per step (vs 1 for Euler), but quality at N Heun steps is
        typically equivalent to ~2N Euler steps, so total compute is comparable while
        truncation error drops from O(dt^2) to O(dt^3) — a real win at small step counts.
        """
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        for step in tqdm(range(1, len(t_span))):
            t = t_span[step - 1]
            dt = t_span[step] - t

            k1 = self._estimate(x, prompt_x, x_lens, t, style, mu, inference_cfg_rate)

            x_pred = x + dt * k1
            x_pred[:, :, :prompt_len] = 0
            k2 = self._estimate(x_pred, prompt_x, x_lens, t + dt, style, mu, inference_cfg_rate)

            x = x + dt * 0.5 * (k1 + k2)
            x[:, :, :prompt_len] = 0

        return x

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)
        """
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]
    def forward(self, x1, x_lens, prompt_lens, mu, style):
        """Computes diffusion loss

        Args:
            mu (torch.Tensor): semantic info of reference audio and altered audio
                shape: (batch_size, mel_timesteps(795+1069), 512)
            x1: mel
            x_lens (torch.Tensor): mel frames output
                shape: (batch_size, mel_timesteps)
            prompt (torch.Tensor): reference mel
                shape: (batch_size, 80, 795)
            style (torch.Tensor): reference global style
                shape: (batch_size, 192)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z



class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")

    def enable_torch_compile(self):
        """Enable torch.compile optimization for the estimator model.
        
        This method applies torch.compile to the estimator (DiT model) for significant
        performance improvements during inference. It also configures distributed
        training optimizations if applicable.
        """
        if torch.distributed.is_initialized():
            torch._inductor.config.reorder_for_compute_comm_overlap = True
        self.estimator = torch.compile(
            self.estimator, 
            fullgraph=True,
            dynamic=True,
        )
