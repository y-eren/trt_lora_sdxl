import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import os

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
OUTPUT_DIR = "outputs/trt-lora-sdxl"
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-4
IMG_SIZE = 1024

device = "cuda"
dtype = torch.bfloat16

pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    variant=None,
).to(device)

vae = pipe.vae
unet = pipe.unet
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")


vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.text_encoder_2.requires_grad_(False)
unet.requires_grad_(False)


lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"],
    lora_dropout=0.1,
    bias="none",
    task_type="UNET",
)
unet = get_peft_model(unet, lora_config)

optimizer = torch.optim.Adam(unet.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):

        pixel_values = batch["pixel_values"].to(device, dtype=dtype)

        # 1) prompt encode
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = pipe.encode_prompt(
                batch["caption"],
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        # 2) add_time_ids
        bs = pixel_values.shape[0]
        add_time_ids_list = []
        for i in range(bs):
            oh, ow = batch["original_size"][i].tolist()
            th, tw = batch["target_size"][i].tolist()
            add_time_ids_list.append([oh, ow, 0, 0, th, tw])
        add_time_ids = torch.tensor(add_time_ids_list, device=device, dtype=dtype)

        with autocast(dtype=dtype):
            # 3) latents
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # noise ekle
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bs,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            # UNet
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
unet.save_pretrained(OUTPUT_DIR)
print("LoRA kaydedildi:", OUTPUT_DIR)
