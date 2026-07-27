# Uyku ve Tarih V3 — Konudan Videoya

Kullanıcı yalnızca video konusunu yazar. Sistem otomatik olarak:

- Konuyu yorumlar.
- Video açısını belirler.
- Türkçe senaryoyu yazar.
- Senaryoyu 12 sahneye ayırır.
- Her sahnenin AI görsel promptunu kendi oluşturur.
- Görselleri AI ile üretir.
- Her görseli sahne anlamına göre AI ile kontrol eder.
- Kötü görseli yeniden üretir.
- Türkçe seslendirme yapar.
- Yazısız ve numarasız final videoyu oluşturur.
- Ayrı YouTube kapağı, başlık, açıklama ve altyazı hazırlar.

## GitHub Secrets

İki secret gerekir:

- `GEMINI_API_KEY`
- `POLLINATIONS_API_KEY`

Anahtarları hiçbir zaman kaynak koduna yazma.

## Çıktılar

- `pilot-video-v3.mp4`
- `kapak.jpg`
- `seslendirme.wav`
- `senaryo.txt`
- `altyazi.srt`
- `storyboard-kontrol.jpg`
- `video-paketi.json`
- `youtube-aciklamasi.txt`
- `uretim-raporu.json`

Storyboard yalnızca kontrol içindir. Final videoda sahne numarası veya sahne adı
yer almaz.

## İlk hedef

V3 ilk çalıştırmada yaklaşık 90 saniyelik pilot üretir. Görsel ve ses kalitesi
onaylandıktan sonra aynı sistem bölüm bazlı çalıştırılarak 10, 30 ve 60 dakikalık
videolara genişletilir.
