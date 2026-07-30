# Uyku ve Tarih V11.1 — Free Cloud Edition

Bu sürüm bilgisayar açık kalmadan standart GitHub `ubuntu-latest` runner üzerinde çalışır.
R2, self-hosted runner, yerel Ollama ve yerel ComfyUI kullanılmaz.

## Sistem

- Gemini: konu çözümleme, kaynak seçimi, hikâye anayasası ve senaryo
- Cloudflare FLUX.1 Schnell: sahne görselleri
- GitHub runner: Tesseract OCR, OpenCV ve CLIP kalite kontrolü
- Gemini Charon: tek anlatıcı
- FFmpeg: altı bölüm ve final video
- GitHub Actions cache: checkpoint
- Zamanlanmış workflow: günde dört kez otomatik devam

## Ücretsiz güvenlik sınırları

- Aynı anda yalnızca tek aktif proje
- Proje cache üst sınırı 3,8 GB
- Yeni cache kaydedildikten sonra eski proje cache'leri otomatik silinir
- Reddedilen görsel adayları ve geçici renderlar silinir
- Cloudflare veya Charon kotası dolunca kontrollü durur
- Ücretli plana otomatik geçmez

## Gerekli GitHub Secrets

`Settings → Secrets and variables → Actions`

- `GEMINI_API_KEY`
- `CLOUDFLARE_ACCOUNT_ID`
- `CLOUDFLARE_API_TOKEN`

## Başlatma

`Actions → Uyku ve Tarih V11.1 - Yeni Video`

Yalnız konu ve hedef süre girilir. Konuya özel kod veya dosya gerekmez.

## Uzun video

60 dakikalık video altı bağlantılı bölüm hâlinde hazırlanır. Her bölümde 18 ayrı
sahne vardır. Ajan anlatıcının söylediği olaydan görsel sözleşmesi çıkarır;
görselde kişi, eylem ve mekân birlikte istenir.

Görseller OCR, netlik, pozlama ve yerel CLIP anlamsal eşleşme kontrolünden geçer.
Final videoya storyboard, timeline veya bilgi kartı girmez.

## Çıktı

5 ve 10 dakikalık pilot tamamlanırsa MP4 artifact olarak indirilebilir.
60 dakikalık final cache içinde korunur. Sonraki aşamada YouTube OAuth bağlanarak
final video doğrudan private olarak yüklenecektir.

## Not

GitHub cache kalıcı arşiv değildir. Workflow aktif projeye günde dört kez erişir
ve eski proje cache'ini yeni sürümle değiştirir. Final için kalıcı hedef YouTube'dur.
