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



class CustomScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(
            self,
            model_output: torch.Tensor,
            input_image: torch.Tensor,
            timestep: int,
            sample: torch.Tensor,
            generator=None,
            return_dict: bool = True,
            condition=True
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        t = timestep

        assert model_output.shape[-1] % input_image.shape[-1] == 0
        scale = model_output.shape[-1] // input_image.shape[-1]

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

        # 6. Add noise
        if t > 0:
            device = model_output.device
            variance_noise = torch.randn(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )

            s2 = self._get_variance(t, predicted_variance=predicted_variance)


            if condition:
                A = input_image
                mu = pred_prev_sample

                mus = []
                for i in range(scale):
                    for j in range(scale):
                        mus.append(mu[:, :, i::scale, j::scale])
                mus = torch.stack(mus, dim=0)
                

                mu_c = mus[:-1] + A[None] - mus.mean(dim=0)[None]


                

                Sigma_c = s2 * (torch.eye(mu_c.shape[0], device=device, dtype=model_output.dtype)
                                - 1 / (mu_c.shape[0] + 1) * torch.ones(mu_c.shape[0], device=device, dtype=model_output.dtype))

                dist = MultivariateNormal(torch.zeros(mu_c.shape[0], device=device, dtype=model_output.dtype),
                                          covariance_matrix=Sigma_c)

                xs = mu_c.permute((1, 2, 3, 4, 0)) + dist.sample(mu_c.shape[1:])

                xs_last = (mu_c.shape[0] + 1) * A - xs.sum(dim=-1)
                xs = torch.cat((xs, xs_last[..., None]), dim=-1)
                k = 0
                for i in range(scale):
                    for j in range(scale):
                        pred_prev_sample[:, :, i::scale, j::scale] = xs[..., k]
                        k += 1
            else:
                pred_prev_sample = pred_prev_sample + torch.sqrt(s2) * variance_noise

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

