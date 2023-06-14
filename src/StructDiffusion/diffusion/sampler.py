import torch
from tqdm import tqdm
from StructDiffusion.diffusion.noise_schedule import extract

class Sampler:

    def __init__(self, model_class, checkpoint_path, device, debug=False):

        self.debug = debug
        self.device = device

        self.model = model_class.load_from_checkpoint(checkpoint_path)
        self.backbone = self.model.model
        self.backbone.to(device)
        self.backbone.eval()

    def sample(self, batch, num_poses):

        noise_schedule = self.model.noise_schedule

        B = batch["pcs"].shape[0]

        x_noisy = torch.randn((B, num_poses, 9), device=self.device)

        xs = []
        for t_index in tqdm(reversed(range(0, noise_schedule.timesteps)),
                            desc='sampling loop time step', total=noise_schedule.timesteps):

            t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

            # noise schedule
            betas_t = extract(noise_schedule.betas, t, x_noisy.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(noise_schedule.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape)
            sqrt_recip_alphas_t = extract(noise_schedule.sqrt_recip_alphas, t, x_noisy.shape)

            # predict noise
            pcs = batch["pcs"]
            sentence = batch["sentence"]
            type_index = batch["type_index"]
            position_index = batch["position_index"]
            pad_mask = batch["pad_mask"]
            # calling the backbone instead of the pytorch-lightning model
            with torch.no_grad():
                predicted_noise = self.backbone.forward(t, pcs, sentence, x_noisy, type_index, position_index, pad_mask)

            # compute noisy x at t
            model_mean = sqrt_recip_alphas_t * (x_noisy - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)
            if t_index == 0:
                x_noisy = model_mean
            else:
                posterior_variance_t = extract(noise_schedule.posterior_variance, t, x_noisy.shape)
                noise = torch.randn_like(x_noisy)
                x_noisy = model_mean + torch.sqrt(posterior_variance_t) * noise

            xs.append(x_noisy)

        xs = list(reversed(xs))
        return xs