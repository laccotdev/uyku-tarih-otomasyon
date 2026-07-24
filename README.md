# Uyku ve Tarih Otomasyonu — V1 Pilot

Bu paket, bilgisayarında video render programı çalıştırmadan GitHub Actions
sunucusunda kısa bir pilot video üretir.

İlk testte API anahtarı gerekmez. Sistem:

1. Hazır pilot senaryosunu kullanır.
2. Wikimedia Commons'tan serbest lisanslı tarihî görseller arar.
3. Piper ile Türkçe seslendirme üretir.
4. FFmpeg ile yavaş yakınlaşmalı MP4 hazırlar.
5. Kapak, altyazı, başlık, açıklama ve görsel kaynaklarını tek pakette verir.

## İlk kurulum

### 1. GitHub deposu oluştur

GitHub'da **New repository** seçeneğine gir.

- Repository name: `uyku-tarih-otomasyon`
- Visibility: `Private`
- README, .gitignore veya lisans ekleme seçeneklerini işaretleme.
- `Create repository` seçeneğine bas.

### 2. Bu paketin içeriğini yükle

ZIP dosyasını bilgisayarında normal klasöre çıkar.

GitHub'daki boş depo ekranında:

- `uploading an existing file` bağlantısına bas.
- ZIP'in içindeki bütün dosya ve klasörleri sürükle.
- `.github`, `src`, `README.md`, `requirements.txt` ve diğer dosyaların
  yüklendiğini kontrol et.
- Alttaki `Commit changes` düğmesine bas.

ZIP dosyasının kendisini yüklemek yeterli değildir; ZIP'in içeriği yüklenmelidir.

## İlk anahtarsız test

1. Depoda üst menüden **Actions** sekmesine gir.
2. Soldan **Uyku ve Tarih - Pilot Video** iş akışını seç.
3. Sağ tarafta **Run workflow** düğmesine bas.
4. Alanları ilk çalıştırmada değiştirme:
   - Hedef süre: `2`
   - Görsel sayısı: `8`
   - Test modu: açık
5. Yeşil **Run workflow** düğmesine bas.

Çalışma genellikle birkaç dakika sürer. Yeşil tik oluştuğunda çalışmayı aç.
Sayfanın en altında **Artifacts** bölümündeki `uyku-tarih-cikti-...` dosyasını indir.

İndirilen çıktıda şunlar bulunur:

- `pilot-video.mp4`
- `kapak.jpg`
- `seslendirme.wav`
- `senaryo.txt`
- `altyazi.srt`
- `baslik.txt`
- `youtube-aciklamasi.txt`
- `gorsel-kaynaklari.txt`
- `uretim-ozeti.json`

## Gemini ile gerçek konu üretimi

Pilot başarılı olduktan sonra Google AI Studio üzerinden ücretsiz Gemini API
anahtarı oluştur.

GitHub deposunda:

1. `Settings`
2. `Secrets and variables`
3. `Actions`
4. `New repository secret`
5. Name: `GEMINI_API_KEY`
6. Secret: Google AI Studio'dan aldığın anahtar
7. `Add secret`

Sonraki çalıştırmada `test_mode` seçeneğini kapat. Konu ve süre alanlarını
değiştir. Sistem senaryoyu Gemini ile o konuya özel hazırlayacaktır.

## V1 sınırları

- Bu sürüm önce teknik hattı doğrulamak için 2–10 dakikalık pilot üretir.
- Tarihsel araştırma ve kaynak doğrulama katmanı V2'de genişletilecektir.
- YouTube'a otomatik yükleme V2'de OAuth bağlantısıyla eklenecektir.
- Ücretsiz servislerin kota ve kullanım şartları zamanla değişebilir.
- Wikimedia görsellerinin lisans bilgileri çıktıda tutulur; yine de yayın
  öncesinde görsel kaynakları dosyasını kontrol et.
