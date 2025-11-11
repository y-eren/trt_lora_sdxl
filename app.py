import os
import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


LOCAL_LORA_PATH = "outputs/trt-lora-sdxl/epoch-3/unet_lora.pt"


HF_LORA_URL = "https://huggingface.co/ByteN1ght/trt-sdxl-lora/resolve/main/unet_lora.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

print("SDXL yükleniyor...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
).to(device)


pipe.enable_attention_slicing(1)
pipe.enable_vae_slicing()


if os.path.exists(LOCAL_LORA_PATH):
    print(f" Lokal LoRA bulundu: {LOCAL_LORA_PATH}")
    state_dict = torch.load(LOCAL_LORA_PATH, map_location=device)
else:
    print(" Lokal LoRA bulunamadı, HF'ten indiriyorum...")
    state_dict = torch.hub.load_state_dict_from_url(
        HF_LORA_URL,
        map_location=device
    )


pipe.unet.load_state_dict(state_dict, strict=False)
print("LoRA UNet'e takıldı.")


def generate(prompt, steps, guidance, height, width, seed):

    if seed is not None and seed != 0:
        generator = torch.Generator(device=device).manual_seed(int(seed))
    else:
        generator = None

    image = pipe(
        prompt,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        height=int(height),
        width=int(width),
        generator=generator,
    ).images[0]
    return image

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(
            value="TRT arşiv görüntüsü, siyah beyaz, 1970ler Türkiye, stüdyo çekimi",
            label="Prompt"
        ),
        gr.Slider(10, 50, value=30, step=1, label="Steps"),
        gr.Slider(1, 12, value=7.0, step=0.5, label="Guidance"),
        gr.Radio([512, 768, 1024], value=768, label="Height"),
        gr.Radio([512, 768, 1024], value=768, label="Width"),
        gr.Number(value=0, label="Seed (0 = random)"),
    ],
    outputs=gr.Image(type="pil"),
    title="TRT SDXL LoRA",
    description="Runpod üzerinde TRT arşiv stilinde görüntü üretimi",
)


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False
)
