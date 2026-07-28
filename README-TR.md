# Uyku ve Tarih V10 — Longform Core

Bu sürüm, 45–60 dakikalık videoyu tek ve kırılgan bir işlem yerine bölüm bölüm hazırlar.

## V10.0 kapsamı

- 6 bölümlü uzun hikâye planı
- Wikipedia tabanlı ücretsiz araştırma paketi
- Wikimedia Commons üzerinden lisans bilgili gerçek görseller
- Harita/zaman çizelgesi için kontrollü Türkçe yerel grafikler
- Gemini Charon ile tek anlatıcı
- Aynı görselde yapay zoom yok; sabit kare ve temiz kesmeler
- Bölüm bazlı checkpoint/resume
- Tamamlanan metin, görsel, ses ve bölüm videosunu yeniden üretmeme
- Görsel kaynak ve lisans listesini YouTube açıklamasına hazırlama

YouTube yükleme bu pakette kapalıdır. Otomatik yükleme ve yayın zamanlama V11 aşamasıdır.

## İlk çalıştırma: mutlaka pilot

GitHub > Actions > **Uyku ve Tarih V10 - Longform Core** > Run workflow

Şu değerlerle başlayın:

```text
topic: İstanbul'un fethi nasıl gerçekleşti?
project_slug: istanbulun-fethi
mode: pilot
stage: all
chapter: 0
runner: ubuntu-latest
```

Pilot yaklaşık 8–12 dakikalık tek bölüm üretir. Pilot başarılı olmadan `full` kullanmayın.

## 1 saatlik üretim

Tam üretim 6 bölümden oluşur. Ücretsiz kota ve hata riskini azaltmak için bölüm bölüm çalıştırın:

```text
mode: full
stage: all
chapter: 1
```

Sonra aynı `project_slug` ile chapter 2, 3, 4, 5 ve 6 çalıştırılır. Cache önceki checkpointi geri yükler. Altıncı bölüm tamamlandığında bütün hazır bölümler final videoda birleştirilir.

Uzun render için `runner: self-hosted` önerilir. İlk pilot `ubuntu-latest` üzerinde çalışabilir.

## Üretim klasörü

```text
projects/<project_slug>/
├── state.json
├── research.json
├── story-plan.json
├── used-commons-pageids.json
├── chapters/
│   ├── chapter-01/
│   │   ├── script.json
│   │   ├── assets-manifest.json
│   │   ├── credits.json
│   │   ├── narration.wav
│   │   ├── timeline.json
│   │   └── chapter.mp4
│   └── ...
└── deliverables/
    ├── <project_slug>-v10-longform.mp4
    ├── youtube-description.txt
    ├── credits.json
    └── DEVAM_ETME_RAPORU.txt
```

## Kontrollü durma

TTS kotası, Wikimedia görsel açığı veya geçici servis sorunu oluşursa ajan kırmızı bir final üretmez. `state.json` ve `DEVAM_ETME_RAPORU.txt` kaydedilir. Aynı `project_slug` ile yeniden çalıştırıldığında tamamlanan aşamalardan devam eder.

## Gerekli secret

Repository > Settings > Secrets and variables > Actions:

```text
GEMINI_API_KEY
```

Cloudflare anahtarı V10.0 için gerekli değildir.
