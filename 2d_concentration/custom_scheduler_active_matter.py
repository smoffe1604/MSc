import glob
from dataclasses import dataclass
from typing import Union, Tuple
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from torch.distributions import MultivariateNormal
import torch
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from torchvision.transforms.functional import pil_to_tensor
from diffusers.utils.torch_utils import randn_tensor




class CustomScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n = (128*128)-1
        # Create matrices directly on CUDA with float16 precision
        Sigma_c = (
            torch.eye(n, device='cuda') 
            - 1 / (n + 1) * torch.ones((n, n), device='cuda')
        )
        self.dist = MultivariateNormal(
            torch.zeros(n, device='cuda'),
            covariance_matrix=Sigma_c
        )

    def step(
            self,
            model_output: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            generator=None,
            return_dict: bool = True,
            condition=True
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep



        prev_t = self.previous_timestep(t)
        predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # 3. Clip or threshold "predicted x_0"
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        # 6. Add noise with block diagonal covariance
        if t > 0:       
            device = model_output.device
            s2 = self._get_variance(t, predicted_variance=predicted_variance)


            if condition:
                B, C, H, W = pred_prev_sample.shape
                mu = pred_prev_sample
                mu_flat = mu.reshape(B, C, -1)  # shape [B, C, n]
                n = mu_flat.shape[2]
                mu_c = mu_flat[:, :, :-1] + 1.0 - mu_flat.mean(dim=2, keepdim=True) 
                xs = mu_c + (torch.sqrt(s2) * self.dist.sample((B, C)))
                xs_last = (n) * 1.0 - xs.sum(dim=2, keepdim=True)  # shape [B, C, 1]
                xs = torch.cat([xs, xs_last], dim=2)  # [B, C, n]
                pred_prev_sample = xs.reshape(B, C, H, W)
            else:
                pred_prev_sample = pred_prev_sample + torch.randn_like(pred_prev_sample) * torch.sqrt(s2)

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


