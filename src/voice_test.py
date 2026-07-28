from __future__ import annotations

import base64
import html
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types


OUTPUT = Path("ses-test-sonuclari")

VOICES = [
    {
        "number": 1,
        "name": "Gacrux",
        "label": "Olgun",
        "filename": "01-gacrux.wav",
    },
    {
        "number": 2,
        "name": "Sadaltager",
        "label": "Bilgili",
        "filename": "02-sadaltager.wav",
    },
    {
        "number": 3,
        "name": "Charon",
        "label": "Bilgilendirici",
        "filename": "03-charon.wav",
    },
    {
        "number": 4,
        "name": "Sulafat",
        "label": "Sıcak",
        "filename": "04-sulafat.wav",
    },
]

MODEL_CANDIDATES = [
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
]

TEST_TEXT = (
    "Bin dört yüz elli üç yılının baharında, İstanbul yalnızca büyük bir "
    "kuşatmanın değil, uzun süredir biriken siyasi ve askerî kararların "
    "merkezindeydi. Osmanlı ordusu surların önünde hazırlık yaparken, şehirde "
    "yaşayan insanlar her gün daralan imkânlarla hayatlarını sürdürmeye "
    "çalışıyordu. Sonucu belirleyen şey tek bir saldırı değil; planlama, "
    "teknoloji, zamanlama ve karşılıklı kararların birbirini değiştirmesiydi."
)


def write_wav(
    path: Path,
    pcm: bytes,
    *,
    channels: int = 1,
    rate: int = 24000,
    sample_width: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


def extract_pcm(response: Any) -> bytes:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        raise RuntimeError("TTS yanıtında aday bulunamadı.")

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    if not parts:
        raise RuntimeError("TTS yanıtında ses parçası bulunamadı.")

    inline_data = getattr(parts[0], "inline_data", None)
    data = getattr(inline_data, "data", None)
    if not data:
        raise RuntimeError("TTS yanıtındaki ses verisi boş.")

    if isinstance(data, str):
        return base64.b64decode(data)
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, bytes):
        return data

    raise TypeError(f"Desteklenmeyen ses veri türü: {type(data)!r}")


def test_prompt(transcript: str) -> str:
    return f"""
# AUDIO PROFILE

A mature Turkish male historical documentary narrator.
Warm, credible, calm and intelligent. Close studio microphone.
Standard Turkey Turkish pronunciation.

# DIRECTOR'S NOTES

Read at a natural documentary pace of approximately 140 to 145 words per minute.
Do not sound sleepy, lethargic, theatrical, like a trailer, or like a newsreader.
Use short natural pauses only where punctuation requires them.
Keep the voice grounded and confident.
Pronounce Turkish names and numbers clearly.
Read the transcript exactly. Do not add, remove, summarize, or paraphrase words.
No music, no sound effects, no whispering.

# TRANSCRIPT

{transcript}
""".strip()


def generate_sample(
    client: genai.Client,
    *,
    model: str,
    voice_name: str,
    transcript: str,
    target: Path,
) -> None:
    response = client.models.generate_content(
        model=model,
        contents=test_prompt(transcript),
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
        ),
    )

    pcm = extract_pcm(response)
    if len(pcm) < 24000:
        raise RuntimeError(
            f"{voice_name} sesi beklenenden kısa veya boş döndü."
        )

    write_wav(target, pcm)


def choose_model_and_generate_first(
    client: genai.Client,
    transcript: str,
) -> tuple[str, dict[str, Any]]:
    first_voice = VOICES[0]
    errors: list[str] = []

    for model in MODEL_CANDIDATES:
        try:
            print(
                f"Model kontrolü: {model} / ses={first_voice['name']}"
            )
            target = OUTPUT / first_voice["filename"]
            generate_sample(
                client,
                model=model,
                voice_name=first_voice["name"],
                transcript=transcript,
                target=target,
            )
            return model, {
                **first_voice,
                "status": "success",
                "model": model,
                "error": "",
            }
        except Exception as exc:
            message = f"{model}: {exc}"
            errors.append(message)
            print("Model kullanılamadı:", message)
            target = OUTPUT / first_voice["filename"]
            target.unlink(missing_ok=True)

    raise RuntimeError(
        "Hiçbir Gemini TTS modeli ses üretemedi. "
        + " | ".join(errors)
    )


def make_html(
    results: list[dict[str, Any]],
    selected_model: str,
    transcript: str,
) -> None:
    cards: list[str] = []

    for item in results:
        number = int(item["number"])
        name = html.escape(str(item["name"]))
        label = html.escape(str(item["label"]))
        filename = html.escape(str(item["filename"]))
        status = str(item["status"])

        if status == "success":
            player = (
                f'<audio controls preload="metadata" src="{filename}"></audio>'
            )
            badge = '<span class="ok">Hazır</span>'
        else:
            error = html.escape(str(item.get("error", "Bilinmeyen hata")))
            player = f'<p class="error">{error}</p>'
            badge = '<span class="bad">Üretilemedi</span>'

        cards.append(
            f"""
            <section class="voice">
              <div class="title">
                <strong>{number}. {name}</strong>
                <span>{label}</span>
                {badge}
              </div>
              {player}
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Uyku ve Tarih — Ses Seçimi</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Arial, sans-serif;
      background: #0c1016;
      color: #ece7dc;
    }}
    body {{
      max-width: 900px;
      margin: 0 auto;
      padding: 36px 20px 60px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    .meta {{
      color: #aeb5c0;
      margin-bottom: 28px;
      line-height: 1.55;
    }}
    .voice {{
      border-top: 1px solid #29313d;
      padding: 22px 0;
    }}
    .title {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .title strong {{
      font-size: 20px;
    }}
    .title > span:not(.ok):not(.bad) {{
      color: #aeb5c0;
    }}
    .ok, .bad {{
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 12px;
    }}
    .ok {{
      background: #163a2b;
      color: #a8e5c4;
    }}
    .bad {{
      background: #472020;
      color: #ffc2c2;
    }}
    audio {{
      width: min(100%, 640px);
    }}
    .transcript {{
      margin-top: 32px;
      padding: 18px;
      background: #131a23;
      border-radius: 12px;
      line-height: 1.65;
      color: #cbd2dc;
    }}
    .error {{
      color: #ffc2c2;
      overflow-wrap: anywhere;
    }}
  </style>
</head>
<body>
  <h1>Uyku ve Tarih — Ses Seçim Testi</h1>
  <p class="meta">
    Dört ses aynı metin, aynı model ve aynı yönetmen talimatıyla üretildi.<br>
    Kullanılan model: <strong>{html.escape(selected_model)}</strong>
  </p>
  {''.join(cards)}
  <div class="transcript">
    <strong>Okunan test metni</strong>
    <p>{html.escape(transcript)}</p>
  </div>
</body>
</html>
"""

    (OUTPUT / "ses-secimi.html").write_text(
        document,
        encoding="utf-8",
    )


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY GitHub Repository Secret olarak bulunamadı."
        )

    transcript = os.getenv("VOICE_TEST_TEXT", "").strip() or TEST_TEXT
    OUTPUT.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)

    selected_model, first_result = choose_model_and_generate_first(
        client,
        transcript,
    )
    results = [first_result]

    # Aynı karşılaştırmada model değiştirilmez. Ses farkının tek değişkeni
    # voice_name olmalıdır.
    for voice in VOICES[1:]:
        target = OUTPUT / voice["filename"]
        try:
            print(
                f"Ses üretiliyor: {voice['number']}. {voice['name']} "
                f"/ model={selected_model}"
            )
            generate_sample(
                client,
                model=selected_model,
                voice_name=voice["name"],
                transcript=transcript,
                target=target,
            )
            results.append({
                **voice,
                "status": "success",
                "model": selected_model,
                "error": "",
            })
        except Exception as exc:
            target.unlink(missing_ok=True)
            results.append({
                **voice,
                "status": "failed",
                "model": selected_model,
                "error": str(exc),
            })
            print(f"{voice['name']} üretilemedi: {exc}")

        # Ücretsiz kota üzerinde ani arka arkaya yük oluşturmamak için.
        time.sleep(6)

    report = {
        "selected_model": selected_model,
        "test_text": transcript,
        "voices": results,
        "successful_count": sum(
            item["status"] == "success"
            for item in results
        ),
    }
    (OUTPUT / "ses-raporu.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    make_html(results, selected_model, transcript)

    successful = int(report["successful_count"])
    print(f"Ses testi tamamlandı: {successful}/4 örnek hazır.")

    # En az ilk model testi başarılıysa artifact mutlaka oluşturulur.
    # Bazı sesler geçici kota yüzünden eksikse workflow kırmızıya dönmez;
    # rapor hangi örneğin eksik olduğunu açıkça gösterir.
    if successful < 1:
        raise RuntimeError("Hiçbir ses örneği üretilemedi.")


if __name__ == "__main__":
    main()
