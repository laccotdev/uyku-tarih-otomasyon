# Uyku ve Tarih V2 — Profesyonel Kalite Laboratuvarı

Bu paket doğrudan uzun video üretmez. Önce kaliteyi ayrı ayrı doğrular:

- Aynı metinle üç doğal Gemini TTS sesi
- Gerçek ve lisanslı arşiv görsellerinden storyboard
- Schedar sesiyle kısa sinematik önizleme
- Yeni kapak yönü
- Görsel kaynak ve lisans dökümü

## Üretilen dosyalar

- `ses-charon.wav`
- `ses-schedar.wav`
- `ses-sulafat.wav`
- `storyboard.jpg`
- `onizleme-schedar.mp4`
- `kapak-v2.jpg`
- `ses-test-metni.txt`
- `gorsel-kaynaklari-v2.txt`
- `ONCE-BUNU-OKU.txt`

## GitHub kurulumu

1. `requirements-v2.txt` dosyasını deponun köküne yükle.
2. `src/quality_lab.py` dosyasını `src` klasörüne yükle.
3. GitHub'da `.github/workflows/quality-lab.yml` dosyasını oluştur.
4. Workflow içeriğini paketteki aynı adlı dosyadan kopyala.
5. Actions bölümünden `Uyku ve Tarih - Profesyonel Kalite Testi` iş akışını çalıştır.

`GEMINI_API_KEY` repository secret olarak önceden eklenmiş olmalıdır.
