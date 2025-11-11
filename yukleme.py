from huggingface_hub import upload_folder


repo_id = "ByteN1ght/trt-sdxl-lora"

upload_folder(
    folder_path="/trt_lora_sdxl/hf_upload",
    repo_id=repo_id,
    repo_type="model"
)

print("Yükleme bitti.")
