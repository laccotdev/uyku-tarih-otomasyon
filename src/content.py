from __future__ import annotations

import json
import os
import re
from typing import Any


def _clean_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model yanıtında JSON bulunamadı.")
    return json.loads(text[start : end + 1])


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("title", "thumbnail_text", "description", "narration", "search_queries")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"Eksik alanlar: {', '.join(missing)}")

    payload["title"] = str(payload["title"]).strip()
    payload["thumbnail_text"] = str(payload["thumbnail_text"]).strip()
    payload["description"] = str(payload["description"]).strip()
    payload["narration"] = str(payload["narration"]).strip()

    queries = payload.get("search_queries", [])
    if not isinstance(queries, list):
        queries = []
    payload["search_queries"] = [str(q).strip() for q in queries if str(q).strip()]

    chapters = payload.get("chapters", [])
    if not isinstance(chapters, list):
        chapters = []
    payload["chapters"] = [
        str(item.get("title", "") if isinstance(item, dict) else item).strip()
        for item in chapters
    ]
    payload["chapters"] = [x for x in payload["chapters"] if x]

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        tags = []
    payload["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]

    if len(payload["search_queries"]) < 4:
        payload["search_queries"].extend(
            [
                "ancient history archaeological site",
                "ancient artifacts museum public domain",
                "historical map public domain",
                "ancient ruins landscape",
            ]
        )

    if not payload["chapters"]:
        payload["chapters"] = [
            "Geceye giriş",
            "Dönemin dünyası",
            "Gündelik hayat",
            "Sessiz kapanış",
        ]

    return payload


def sample_payload(topic: str) -> dict[str, Any]:
    return _validate_payload(
        {
            "title": f"{topic} | Uyku İçin Sakin Tarih Anlatısı",
            "thumbnail_text": "SON GECE",
            "description": (
                "Bu pilot bölüm, sakin tempolu bir tarih anlatısı otomasyonunun "
                "teknik denemesidir. Görseller Wikimedia Commons üzerinden serbest "
                "lisans bilgileri kontrol edilerek alınır."
            ),
            "narration": (
                "Şimdi, bulunduğun yerde rahatça uzan. Günün seslerini yavaşça geride "
                "bırakırken, zihnimizde binlerce yıl öncesine doğru sessiz bir yolculuğa "
                "çıkıyoruz. Bu gece Anadolu'nun yüksek ve rüzgârlı topraklarındayız. "
                "Karşımızda, güçlü taş duvarlarla çevrili Hattuşa yükseliyor. "
                "Güneş çoktan tepelerin arkasına çekilmiş. Şehrin dar sokaklarında "
                "ocakların son dumanları ağır ağır gökyüzüne karışıyor. "
                "Kapılardan içeri giren son yolcular acele etmiyor. Çünkü gece, eski "
                "dünyada yalnızca karanlık değil; sessizlik, belirsizlik ve bekleyiş "
                "anlamına da geliyor. Şehrin içinde taş döşeli yollar, depolar, "
                "tapınaklar ve krala ait yapılar bulunuyor. İnsanlar yarının bugünden "
                "farklı olacağını henüz bilmiyor olabilir. Uzak bölgelerden gelen "
                "haberler düzenin bozulduğunu söylüyor. Ticaret yolları eskisi kadar "
                "güvenli değil. Bazı ürünler pazara daha az ulaşıyor. Kuraklık, "
                "huzursuzluk ve savaş ihtimali, birbirinden ayrı görünen küçük "
                "işaretler gibi şehrin üzerine yerleşiyor. "
                "Yine de bu gecede hayat devam ediyor. Bir evde ekmek bölüşülüyor. "
                "Bir görevli kapıların kapanmasını izliyor. Bir çocuk, duvarların "
                "dışından gelen rüzgârı dinliyor. Biz ise kesin olarak bilinmeyen "
                "ayrıntıları uydurmadan, arkeolojik izlerin bize anlattığı kadarıyla "
                "bu dünyanın içinde kısa bir süre kalıyoruz. "
                "Taş duvarlara vuran rüzgârı hayal et. Uzakta bir ateşin solgun "
                "ışığını gör. Her nefeste, bugünün telaşından biraz daha uzaklaş. "
                "Şehir yavaşça sessizliğe gömülürken, sen de bu eski gecenin sakin "
                "ritmine bırak kendini. Tarihin büyük değişimleri bazen gürültüyle, "
                "bazen de kimsenin tam olarak fark etmediği sessiz gecelerle başlar."
            ),
            "search_queries": [
                "Hattusa ruins",
                "Hittite Lion Gate Hattusa",
                "Hittite artifacts museum",
                "Bronze Age Anatolia map",
                "ancient Anatolian ruins",
                "Hittite relief public domain",
            ],
            "chapters": [
                {"title": "Geceye giriş"},
                {"title": "Hattuşa'nın sokakları"},
                {"title": "Yaklaşan değişimin işaretleri"},
                {"title": "Sessiz kapanış"},
            ],
            "tags": ["uyku için tarih", "Hattuşa", "Hititler", "sakin anlatı"],
        }
    )


def generate_payload(topic: str, duration_minutes: int, test_mode: bool) -> dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if test_mode or not api_key:
        print("İçerik: test metni kullanılıyor.")
        return sample_payload(topic)

    from google import genai
    from google.genai import types

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
    target_words = max(450, int(duration_minutes * 115))

    prompt = f"""
Sen, kaynak belirsizliğini saklamayan, sakin ve güvenilir Türkçe tarih anlatıları
hazırlayan bir editörsün.

KONU: {topic}
HEDEF SÜRE: {duration_minutes} dakika
HEDEF KELİME: yaklaşık {target_words} kelime

Yalnızca geçerli JSON üret. Markdown veya açıklama ekleme.

JSON alanları:
{{
  "title": "YouTube başlığı; merak uyandırıcı ama abartısız",
  "thumbnail_text": "En fazla 4 kelimelik kapak metni",
  "description": "2 kısa paragraf video açıklaması",
  "narration": "Tek parça, sakin Türkçe seslendirme metni",
  "search_queries": ["Wikimedia Commons için İngilizce arama sorguları"],
  "chapters": [{{"title": "Bölüm adı"}}],
  "tags": ["etiket"]
}}

Kurallar:
- Dinleyiciye ikinci tekil şahısla, yumuşak biçimde hitap et.
- İlk 30 saniyede mekânı ve atmosferi kur.
- Ders anlatımı yerine dönemin içine girilen bir gece yolculuğu hissi ver.
- Cümleler doğal ve rahat okunabilir olsun.
- Sürekli dramatik zirveler, bağıran ifadeler ve tekrarlar kullanma.
- Kesin bilinmeyen tarihsel ayrıntıları kesinmiş gibi yazma.
- Mitleri gerçek bilgi gibi aktarma.
- Tarih ve sayıları mümkün olduğunda yazıyla yaz.
- Son bölüm dinleyiciyi rahatsız etmeden sakin biçimde kapansın.
- search_queries alanına görsel sonuç verecek 8-16 İngilizce sorgu yaz.
- narration alanı yaklaşık {target_words} kelime olsun.
"""

    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            print(f"Gemini içerik denemesi: {attempt}/3")
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.75,
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                ),
            )
            return _validate_payload(_clean_json(response.text or ""))
        except Exception as exc:
            last_error = exc
            print(f"Gemini yanıtı işlenemedi: {exc}")

    raise RuntimeError(f"Gemini içerik üretimi başarısız: {last_error}")
