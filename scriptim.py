import os, json, re

images_dir = "./frames"   # resimlerin olduğu klasör
out_path = "./captions.jsonl"

# 1) Burayı kendi aralıklarına göre DÜZENLE
# (start, end, caption) = start ve end dahil
ranges = [
  
     (1, 17, "1980'ler Türkiye, telefonun Türkiye'ye gelmesi, eski siyah-beyaz arşiv görüntüsü, eski insanlar"),
    (18, 47, "1970'ler Türkiye, okul hayatı, çocuklar sınıfta, siyah-beyaz arşiv görüntüsü, okulda beslenme zamanı, eski çocuklar, röportaj"),
    (48, 85, "1970'ler Türkiye, tiyatro oyunu, doktor rolünde adam, siyah-beyaz arşiv görüntüleri, sahne performansı, eski televizyon dizileri"),
    (86, 125, "1980'ler Türkiyesi, şehir hayatı, İstanbul'da kar sonrası, şehir görüntüleri, karlı İstanbul, televizyon haber spikeri"),
    (139,148, "TRT arşiv görüntüsü, siyah-beyaz, eski polis, eski polis arabası, eski İstanbul"),
    (149,156, "1970'ler Türkiye, siyah-beyaz arşiv görüntüsü, eski insanlar"),
    (157, 163, "TRT arşiv görüntüsü, siyah-beyaz, eski insanlar, eski araba"),
    (164, 166, "TRT arşiv görüntüsü, siyah-beyaz, arabada röportaj, eski insanlar, eski araba, eski Türkiye"),
    (167, 178, "TRT arşiv görüntüsü, siyah-beyaz, sokak röportajı, eski Türkiye, trafik polisi ve insanlar"),
    (179, 181, "TRT arşiv görüntüsü, siyah-beyaz, sokak röportajı, eski Türk kadınları"),
    (182, 184, "TRT arşiv görüntüsü, siyah-beyaz, eski trafik, eski köprü görüntüleri, eski Türkiye"),
    (185,186, "TRT arşiv görüntüleri, siyah-beyaz, eski Türkiye, eski İstanbul manzaraları"),
    (187, 189, "TRT arşiv görüntüsü, siyah-beyaz, el arabası, at arabası, eski Türkiye"),
    (190, 209, "TRT arşiv görüntüsü, siyah-beyaz, eski yol, eski köprü, eski arabalar, eski Türkiye"),
    (210, 213, "TRT arşiv görüntüsü, siyah-beyaz, eski Türkiye, eski arabalar, trafik polisi"),
    (214, 232, "TRT arşiv görüntüsü, siyah-beyaz, eski Türkiye, yoğun trafik, kalabalık yol, trafik polisi"),
    (233, 236, "TRT arşiv görüntüsü, siyah-beyaz, seyirciler, sahne / gösteri, eski giyim"),
    (237, 269, "TRT arşiv görüntüsü, siyah-beyaz stüdyo çekimi, Orhan Boran sunuculu TV programı, eski moda"),
    (270, 271, "TRT arşiv görüntüsü, siyah-beyaz, seyirciler, sahne / gösteri, eski giyim"),
    (272, 285, "TRT arşiv görüntüsü, siyah-beyaz stüdyo çekimi, Orhan Boran sunuculu TV programı, eski moda"),
    (286, 287, "TRT arşiv görüntüsü, siyah-beyaz, seyirciler, sahne / gösteri, eski giyim"),
    (288, 402, "TRT arşiv görüntüsü, siyah-beyaz stüdyo çekimi, Orhan Boran sunuculu TV programı, 4:3 yayın formatı"),
    (403, 455, "TRT arşiv görüntüsü, eski tren garı, tarihi mimari, saat kulesi, eski İstanbul, 1970'ler Türkiye"),
    (456,457, "TRT arşiv görüntüsü, eski tren görevlisi, elinde düdük, eski giyimli insanlar"),
    (458, 459, "TRT arşiv görüntüsü, renkli görüntü, eski tren"),
    (460, 494, "TRT arşiv görüntüsü, 1990'lar Türkiye, stüdyo çekimi, televizyon programı, yarışma formatı, eski giyim tarzları"),
    (495,561, "TRT arşiv görüntüsü, eski Türk sineması, kadın oyuncu, Türkan Şoray tarzı filmler, 1970'ler Türkiye"),
    (562, 563, "TRT arşiv görüntüsü, siyah-beyaz, film çekimi, kamera çekimi, film yönetmeni"),
    (564, 567, "TRT arşiv görüntüsü, film çekimi, kamera arkası, Cüneyt Arkın, eski film seti"),
    (568, 579, "TRT arşiv görüntüsü, film çekimi, yönetmen, kamera arkası, siyah-beyaz filmler"),
    (580, 600, "TRT arşiv görüntüsü, siyah-beyaz filmler, kamera arkası, film çekimi, eski oyuncular"),
    (601, 637, "TRT arşiv görüntüsü, TRT yayın akışı ekranı, mavi arka plan, 1980'ler"),
    (638, 668, "TRT arşiv görüntüsü, eski haber jeneriği, özet haber görüntüleri"),
    (669, 673, "TRT haber spikeri, stüdyo çekimi, eski moda, haber sunumu"),
    (678, 686, "TRT haber spikeri, stüdyo çekimi, eski moda, haber sunumu"),
    (699, 704, "TRT haber spikeri, stüdyo çekimi, eski moda, haber sunumu"),
    (674, 677, "TRT arşiv görüntüsü, eski yapı, eski bina, 1980'ler Türkiye"),
    (687, 692, "TRT arşiv görüntüsü, eski otobüs, eski arabalar, trafik kazası haberi, siyah-beyaz"),
    (693, 698, "TRT arşiv görüntüsü, eski hastane odası, hemşire, hasta bakıcı, renkli görüntü, eski hemşire kıyafeti"),
    (747, 757, "TRT arşiv görüntüleri, eski insanlar, tokalaşma / selamlaşma, 1970'ler Türkiye"),
    (758, 771, "TRT haber spikeri, stüdyo çekimi, eski moda, haber"),
    (773, 802, "TRT arşiv görüntüleri, siyah-beyaz, meclis görüntüleri, konuşma, 1970'ler Türkiye"),
    (832, 836, "TRT arşiv görüntüleri, eski insanlar, konuşma, eski giyim, 1970'ler Türkiye"),
    (837, 842, "TRT arşiv görüntüleri, siyah-beyaz, konuşma, eski insanlar, 1970'ler Türkiye"),
    (851, 858, "TRT arşiv görüntüleri, eski giyim, röportaj, koltukta oturan adam, 1970'ler Türkiye"),
    (865, 874, "TRT arşiv görüntüsü, sokak, kalabalık insanlar, 1970'ler Türkiye"),
    (898, 905, "TRT arşiv görüntüsü, deniz, tekneler, yelkenliler, 1980'ler Türkiye"),
    
    
    
    
]

# 2) Numara hiçbir aralığa girmiyorsa kullanılacak genel caption
default_caption = "TRT arşivi tarzı, 1970ler 1980ler türkiye, belgesel çekimi / TV kaydı, hafif solmuş renkler"

def filename_to_num(fname):
    # frame_00853.png -> 853
    m = re.search(r"(\d+)", fname)
    return int(m.group(1)) if m else None

with open(out_path, "w", encoding="utf-8") as f:
    for name in sorted(os.listdir(images_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        num = filename_to_num(name)
        if num is None:
            continue

        # bu numaraya uygun caption'ı bul
        caption = default_caption
        for start, end, cap in ranges:
            if start <= num <= end:
                caption = cap
                break

        rec = {
            "file_name": f"./frames/{name}",
            "text": caption,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("captions.jsonl yazıldı ✅")
