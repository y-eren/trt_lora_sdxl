# TRT SDXL LoRA

Bu repo, TRT'nin eski arşiv, haber, yarışma vb. belgesellerine benzeyen görseller üretebilmek için **STABLE DİFFUSION XL (SDXL)** tabanlı bir **LoRA** eğitim sürecini içermektedir.

# Amaç
- 1970'ler / 1980'ler Türkiye görüntüleri
- siyah, beyaz ya da soluk renkli eski TRT arşiv estetiği
- sokak roportajları, trafik polisi, eski arabalar, studyo çekimleri, meclis sahneleri gibi TRT'de gördüğümüz motifleri yakalamak

  ---

  # Örnek Çıktılar

  <img width="720" height="732" alt="image" src="https://github.com/user-attachments/assets/920fc6a4-f9c3-4c02-b815-2a3dd64559ab" />

  <img width="736" height="735" alt="image" src="https://github.com/user-attachments/assets/af486aed-9c30-4810-81ee-4af534c9a42c" />

  <img width="732" height="725" alt="image" src="https://github.com/user-attachments/assets/ffd9fe86-fd0f-451c-ba19-963b0edc702d" />

  <img width="738" height="733" alt="image" src="https://github.com/user-attachments/assets/32fa5a38-9c3a-47b0-90a1-6606f77d11c7" />

  <img width="1696" height="603" alt="image" src="https://github.com/user-attachments/assets/aa6a5708-8c25-4096-b66b-3d817e7cca7b" />

  <img width="1568" height="669" alt="image" src="https://github.com/user-attachments/assets/c9a5bc96-1aa0-4d1f-810e-bf90194e81f8" />

  <img width="1567" height="671" alt="image" src="https://github.com/user-attachments/assets/8d17dc7a-bd63-4fad-9e1b-169638a5a2b7" />

  # Eğitim

  Eğitim RTX 4000 Ada (20 GB VRAM) makinede yapılmıştır.

  SDXL'in VAE ve text encoderları dondurularak sadece UNET içerisindeki belirli linear attention katmanlarına kendi LoRA katmanlarım eklenmiştir.

  train_trtlora dosyası ile SDXL pipeline indiriliyor LoRA katmanların uygun yerlere enjekte ediliyor (to_q, to_v), captions.jsl dosyasından okuma yapılıyor ve epoch sonunda ağırlıklar kaydedilmektir.

  # Hugging Face

  Eğitilmiş ağırlıklar https://huggingface.co/ByteN1ght/trt-sdxl-lora/tree/main reposuna yüklenmiştir. Bu dosya UNET state_dict'i ile SDXL'e takılmaktadır. Gradio arayüzü ile websitesi üzerinden arayüz sağlanmıştır.

  # Promptlar

  TRT archive footage, 1970s Istanbul street, old cars, traffic police, black and white
  TRT arşiv görüntüsü, 1970ler Türkiye, meclis konuşması, kürsüde adam, siyah beyaz
  old Turkish TV news anchor, 1970s studio, 4:3, low TV quality
  TRT yayın akışı ekranı, 1980ler, mavi arka plan, low-res capture

  
# Notlar

Bu model tamamen kişisel/deneysel bir çalışmadır.




