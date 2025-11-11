import os
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from diffusers import StableDiffusionXLPipeline, DDPMScheduler


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CAPTIONS_FILE = "captions.jsonl"
OUTPUT_DIR = "outputs/trt-lora-sdxl"

IMG_SIZE = 384
BATCH_SIZE = 1
EPOCHS = 3
LR = 1e-5
SAVE_EVERY = 200
GRAD_ACCUM_STEPS = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


# ================= DATASET =================
class TRTFramesDataset(Dataset):
    def __init__(self, captions_path: str, img_size: int = 384):
        self.items = []
        with open(captions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_path = Path(obj["file_name"])
                if img_path.exists():
                    self.items.append({"image_path": img_path, "caption": obj["text"]})
        print(f"[DATASET] {len(self.items)} görüntü bulundu.")

        self.img_size = img_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        orig_w, orig_h = image.size
        pixel_values = self.transform(image)
        return {
            "pixel_values": pixel_values,
            "caption": item["caption"],
            "original_size": torch.tensor([orig_h, orig_w]),
            "target_size": torch.tensor([self.img_size, self.img_size]),
        }


# =============== MANUAL LORA ===============
class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 4, alpha: int = 4, device="cuda"):
        super().__init__()
        self.base = base_layer
        self.scaling = alpha / r

        in_f = base_layer.in_features
        out_f = base_layer.out_features

        self.lora_A = nn.Linear(in_f, r, bias=False, device=device)
        self.lora_B = nn.Linear(r, out_f, bias=False, device=device)


        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(x))
        lora_out = lora_out.to(base_out.dtype)
        return base_out + self.scaling * lora_out


def inject_lora_into_unet(unet: nn.Module, target_modules, r=4, alpha=4, device="cuda"):
    count = 0
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            for tgt in target_modules:
                if name.endswith(tgt):
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = unet
                    if parent_name:
                        for attr in parent_name.split("."):
                            parent = getattr(parent, attr)
                    setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha, device=device))
                    count += 1
                    break
    print(f"[LORA] {count} katmana LoRA takıldı.")


# ============== SDXL PROMPT ENCODE =========
def encode_prompt_sdxl(pipe, prompts, device, dtype):
    if isinstance(prompts, str):
        prompts = [prompts]

    tok1 = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tok2 = pipe.tokenizer_2(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt",
    )


    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)

    tok1 = {k: v.to(device) for k, v in tok1.items()}
    tok2 = {k: v.to(device) for k, v in tok2.items()}

    with torch.no_grad():
        enc1 = pipe.text_encoder(tok1["input_ids"], output_hidden_states=True)
        enc2 = pipe.text_encoder_2(tok2["input_ids"], output_hidden_states=True)

    prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1).to(dtype)
    pooled = enc2.text_embeds.to(dtype)


    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")

    return prompt_embeds, pooled


# ================= MODEL LOAD =================
print("=" * 60)
print("SDXL pipeline yükleniyor...")
print("=" * 60)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    use_safetensors=True,
    torch_dtype=dtype,
).to(device)

# stabil memory
pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()

vae = pipe.vae
unet = pipe.unet
noise_scheduler = DDPMScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

unet.enable_gradient_checkpointing()


pipe.text_encoder.to("cpu")
pipe.text_encoder_2.to("cpu")

vae.requires_grad_(False)
unet.requires_grad_(False)
vae.eval()


target_modules = ["to_q"]
inject_lora_into_unet(unet, target_modules, r=4, alpha=4, device=device)

# ------------ AUTO RESUME ------------
latest_ckpt = None
last_finished_epoch = 0
if os.path.exists(OUTPUT_DIR):
    epoch_dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("epoch-")]
    if epoch_dirs:
        epoch_dirs_sorted = sorted(epoch_dirs, key=lambda x: int(x.split("-")[1]))
        latest_epoch_dir = epoch_dirs_sorted[-1]
        last_finished_epoch = int(latest_epoch_dir.split("-")[1])
        candidate = os.path.join(OUTPUT_DIR, latest_epoch_dir, "unet_lora.pt")
        if os.path.exists(candidate):
            latest_ckpt = candidate

if latest_ckpt:
    state = torch.load(latest_ckpt, map_location=device)
    unet.load_state_dict(state, strict=False)
    print(f"[RESUME] Son checkpoint yüklendi: {latest_ckpt}")
    print(f"[RESUME] Son tamamlanan epoch: {last_finished_epoch}")
else:
    print("[RESUME] checkpoint yok, 0'dan başlayacak.")
    last_finished_epoch = 0
# ------------ AUTO RESUME END ------------

trainable_params = [p for p in unet.parameters() if p.requires_grad]
print(f"[LORA] Eğitilebilir parametreler: {sum(p.numel() for p in trainable_params):,}")

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LR,
    weight_decay=0.01,
    eps=1e-8,
)

# ================= DATA =================
print("=" * 60)
print("Dataset hazırlanıyor...")
print("=" * 60)
dataset = TRTFramesDataset(CAPTIONS_FILE, IMG_SIZE)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
)
print(f"[INFO] toplam step: {len(dataloader) * EPOCHS}")

# ================= TRAIN =================
print("=" * 60)
print("Eğitim başlıyor...")
print("=" * 60)
unet.train()
global_step = 0
accum_counter = 0

for epoch in range(last_finished_epoch, EPOCHS):
    print(f"\n===== EPOCH {epoch+1}/{EPOCHS} =====")
    running_loss = 0.0
    valid_steps = 0

    for step, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device, dtype=dtype)

        prompt_embeds, pooled_prompt_embeds = encode_prompt_sdxl(
            pipe, batch["caption"], device=device, dtype=dtype
        )

        bs = pixel_values.shape[0]
        add_time_ids = []
        for i in range(bs):
            oh, ow = batch["original_size"][i].tolist()
            th, tw = batch["target_size"][i].tolist()
            add_time_ids.append([oh, ow, 0, 0, th, tw])
        add_time_ids = torch.tensor(add_time_ids, device=device, dtype=dtype)

        with torch.amp.autocast("cuda", dtype=dtype):
            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            added_cond_kwargs = {
                "text_embeds": pooled_prompt_embeds,
                "time_ids": add_time_ids,
            }

            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if torch.isnan(noise_pred).any():
                print(f"[UYARI] step {step}: unet çıktısı NaN, adımı atlıyorum.")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            loss = torch.nn.functional.mse_loss(noise_pred.float(), noise.float())
            loss = loss / GRAD_ACCUM_STEPS

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[UYARI] step {step}: loss NaN/Inf, adımı atlıyorum.")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue

        loss.backward()
        accum_counter += 1

        running_loss += loss.item() * GRAD_ACCUM_STEPS
        valid_steps += 1

        if accum_counter % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % 10 == 0:
                torch.cuda.empty_cache()

            if global_step % 10 == 0:
                avg = running_loss / max(1, valid_steps)
                print(
                    f"global_step {global_step} | epoch {epoch+1} | "
                    f"step {step}/{len(dataloader)} | loss {avg:.4f}"
                )

            if global_step % SAVE_EVERY == 0:
                ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{global_step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(unet.state_dict(), os.path.join(ckpt_dir, "unet_lora.pt"))
                print(f"✓ checkpoint kaydedildi: {ckpt_dir}")

    epoch_avg = running_loss / max(1, valid_steps)
    print(f"Epoch {epoch+1} bitti. Ortalama (NaN hariç) loss: {epoch_avg:.4f}")
    epoch_dir = os.path.join(OUTPUT_DIR, f"epoch-{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(epoch_dir, "unet_lora.pt"))
    print(f" epoch kaydedildi: {epoch_dir}")


final_dir = os.path.join(OUTPUT_DIR, "final")
os.makedirs(final_dir, exist_ok=True)
torch.save(unet.state_dict(), os.path.join(final_dir, "unet_lora.pt"))
print("\n====================================")
print("EĞİTİM BİTTİ ✅")
print(f"Final LoRA: {os.path.join(final_dir, 'unet_lora.pt')}")
print("====================================\n")
