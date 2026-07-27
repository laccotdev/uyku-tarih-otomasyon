from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import wave
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output-v3"
WORK = ROOT / "work-v3"
WIDTH = 1920
HEIGHT = 1080
FPS = 25
USER_AGENT = "UykuTarihTopicToVideo/5.1"
CLOUDFLARE_API_BASE = "https://api.cloudflare.com/client/v4/accounts"
VOICE_NAME = "Schedar"

CLOUDFLARE_IMAGE_MODELS = [
    "@cf/black-forest-labs/flux-2-klein-4b",
    "@cf/black-forest-labs/flux-1-schnell",
    "@cf/lykon/dreamshaper-8-lcm",
    "@cf/stabilityai/stable-diffusion-xl-base-1.0",
]

STYLE_BIBLE = """
Photorealistic cinematic historical reconstruction for a premium late-night
documentary. Restrained realism, plausible period architecture, materials,
clothing and tools. One locked film palette across the whole video: deep
blue-black moonlit shadows, muted amber torch highlights, desaturated stone,
earth and bronze, gentle contrast and restrained saturation. Avoid bright
cyan, vivid orange, green casts and radically different color temperatures.
Realistic skin and anatomy, immersive 16:9 composition, no fantasy spectacle.
The visual must look like a frame from one cohesive historical film. No text,
no letters, no numbers, no captions, no logos, no watermarks, no borders,
no collage and no museum display.
""".strip()

GLOBAL_NEGATIVE = """
text, letters, numbers, captions, subtitle, watermark, logo, border, frame,
split screen, collage, infographic, museum display, object on white background,
modern clothing, modern technology, electricity, neon, fantasy armor,
fantasy castle, science fiction, cartoon, anime, illustration, oversaturated,
plastic skin, deformed face, malformed hands, extra fingers, extra limbs,
duplicate people, blurry, low resolution
""".strip()

TEXT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
]

TTS_MODELS = [
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
]

# Cloudflare FLUX.1 accepts at most 2048 characters in `prompt`.
# Keep a safety margin because fallback models receive exclusions in the same field.
MAX_IMAGE_PROMPT_CHARS = 1900
MAX_NEGATIVE_PROMPT_CHARS = 760

TRANSITIONS = [
    "fade",
    "dissolve",
    "smoothleft",
    "smoothright",
    "hblur",
    "fadeblack",
]

ALLOWED_TRANSITIONS = set(TRANSITIONS)
ALLOWED_AMBIENT_PROFILES = {
    "exterior_wind",
    "interior_room",
    "firelight",
    "archive_room",
    "night_silence",
    "distant_storm",
}


def run(command: list[str]) -> None:
    print("$", " ".join(str(x) for x in command))
    subprocess.run(command, check=True)


def reset_dirs() -> None:
    for path in (OUTPUT, WORK):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def safe_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Yanıtta geçerli JSON bulunamadı.")
    return json.loads(text[start:end + 1])


def model_names(client: genai.Client) -> set[str]:
    names: set[str] = set()
    try:
        for model in client.models.list():
            name = str(getattr(model, "name", "")).removeprefix("models/")
            actions = set(getattr(model, "supported_actions", None) or [])
            if name and (not actions or "generateContent" in actions):
                names.add(name)
    except Exception as exc:
        print("Model listesi alınamadı; sabit yedek zinciri kullanılacak:", exc)
    return names


def model_chain(client: genai.Client, preferred: list[str]) -> list[str]:
    configured = []
    for item in preferred:
        item = item.strip().removeprefix("models/")
        if item and item not in configured:
            configured.append(item)
    available = model_names(client)
    if not available:
        return configured
    filtered = [item for item in configured if item in available]
    return filtered or configured


def retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429", "500", "502", "503", "504", "unavailable", "high demand",
            "resource_exhausted", "timeout", "temporarily", "deadline",
            "connection reset", "service unavailable",
        )
    )


def generate_json(
    client: genai.Client,
    prompt: str,
    max_tokens: int = 8192,
) -> tuple[dict[str, Any], str]:
    configured = os.getenv("TEXT_MODEL", "").strip()
    chain = model_chain(client, [configured, *TEXT_MODELS])
    last_error: Exception | None = None

    for model in chain:
        for attempt, delay in enumerate((4, 14, 35), start=1):
            try:
                print(f"Metin üretimi: model={model}, deneme={attempt}/3")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.48,
                        max_output_tokens=max_tokens,
                    ),
                )
                return safe_json(response.text or ""), model
            except Exception as exc:
                last_error = exc
                print(f"Metin modeli başarısız: {model}: {exc}")
                message = str(exc).lower()
                if "404" in message or "not found" in message or "no longer available" in message:
                    break
                if retryable(exc) and attempt < 3:
                    time.sleep(delay)
                    continue
                break

    raise RuntimeError(f"Hiçbir metin modeli çalışmadı. Son hata: {last_error}")


def _word_count(value: Any) -> int:
    return len(re.findall(r"\b\w+\b", str(value), flags=re.UNICODE))


def _split_narration_into_scenes(narration: str, scene_count: int) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", narration.strip())
        if item.strip()
    ]
    if not sentences:
        return [narration.strip()] + [""] * max(0, scene_count - 1)

    total_words = max(1, sum(_word_count(item) for item in sentences))
    target = total_words / max(1, scene_count)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        sentence_words = _word_count(sentence)
        remaining_slots = scene_count - len(chunks)
        if current and current_words + sentence_words > target and remaining_slots > 1:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())

    while len(chunks) < scene_count:
        longest_index = max(range(len(chunks)), key=lambda idx: _word_count(chunks[idx]))
        words = chunks[longest_index].split()
        if len(words) < 8:
            chunks.append("")
            continue
        midpoint = len(words) // 2
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()
        chunks[longest_index] = left
        chunks.insert(longest_index + 1, right)

    if len(chunks) > scene_count:
        head = chunks[: scene_count - 1]
        tail = " ".join(chunks[scene_count - 1 :]).strip()
        chunks = head + [tail]
    return chunks[:scene_count]


def _ensure_scene_narration(payload: dict[str, Any]) -> None:
    scenes = list(payload.get("scenes", []))
    narration = re.sub(r"\s+", " ", str(payload.get("narration", ""))).strip()
    scene_texts = [
        re.sub(r"\s+", " ", str(scene.get("narration_text", ""))).strip()
        for scene in scenes
    ]
    covered_words = sum(_word_count(item) for item in scene_texts)
    narration_words = max(1, _word_count(narration))

    if len(scene_texts) != len(scenes) or covered_words < narration_words * 0.72:
        scene_texts = _split_narration_into_scenes(narration, len(scenes))

    for scene, scene_text in zip(scenes, scene_texts):
        scene["narration_text"] = scene_text

    joined = " ".join(item for item in scene_texts if item).strip()
    if joined:
        payload["narration"] = joined


def _world_bible_text(payload: dict[str, Any]) -> str:
    world = payload.get("world_bible", {})
    if isinstance(world, dict):
        return json.dumps(world, ensure_ascii=False, indent=2)
    return str(world).strip()


def _package_issues(
    payload: dict[str, Any],
    scene_count: int,
    minimum_words: int,
) -> list[str]:
    issues: list[str] = []
    required = (
        "video_title", "thumbnail_text", "description", "narration",
        "visual_identity", "world_bible", "thumbnail_prompt", "scenes",
    )
    missing = [key for key in required if not payload.get(key)]
    if missing:
        issues.append("Eksik alanlar: " + ", ".join(missing))

    narration = re.sub(r"\s+", " ", str(payload.get("narration", ""))).strip()
    count = _word_count(narration)
    if count < minimum_words:
        issues.append(
            f"Senaryo kısa: {count} kelime. En az {minimum_words} kelime olmalı."
        )

    scenes = payload.get("scenes")
    if not isinstance(scenes, list):
        issues.append("scenes alanı liste değil.")
        return issues
    if len(scenes) < scene_count:
        issues.append(
            f"Sahne sayısı yetersiz: {len(scenes)}. Tam {scene_count} sahne gerekli."
        )

    for index, scene in enumerate(scenes[:scene_count], start=1):
        if not isinstance(scene, dict):
            issues.append(f"Sahne {index} geçersiz.")
            continue
        for key in (
            "narration_idea", "narration_text", "visual_goal", "image_prompt",
            "transition", "transition_duration", "ambient_profile",
        ):
            if scene.get(key) in (None, ""):
                issues.append(f"Sahne {index} için {key} eksik.")
    return issues


def _normalize_package(payload: dict[str, Any], scene_count: int) -> None:
    payload["narration"] = re.sub(
        r"\s+", " ", str(payload.get("narration", ""))
    ).strip()
    payload["scenes"] = list(payload.get("scenes", []))[:scene_count]

    for index, scene in enumerate(payload["scenes"], start=1):
        scene["scene_id"] = index
        transition = str(scene.get("transition", "fade")).strip().lower()
        if transition not in ALLOWED_TRANSITIONS:
            transition = "fade"
        scene["transition"] = transition

        try:
            transition_duration = float(scene.get("transition_duration", 0.72))
        except (TypeError, ValueError):
            transition_duration = 0.72
        scene["transition_duration"] = max(0.45, min(1.25, transition_duration))

        ambient = str(scene.get("ambient_profile", "night_silence")).strip().lower()
        if ambient not in ALLOWED_AMBIENT_PROFILES:
            ambient = "night_silence"
        scene["ambient_profile"] = ambient

        try:
            weight = float(scene.get("duration_weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        scene["duration_weight"] = max(0.55, min(1.8, weight))

        if str(scene.get("importance", "normal")) not in {"high", "normal"}:
            scene["importance"] = "normal"

    _ensure_scene_narration(payload)


def _repair_video_package(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    issues: list[str],
    target_words: int,
    scene_count: int,
) -> tuple[dict[str, Any], str]:
    existing = json.dumps(payload, ensure_ascii=False)
    issue_text = "\n".join(f"- {item}" for item in issues)
    prompt = f"""
Yalnızca geçerli JSON üret. Markdown yazma.

Aşağıdaki tarih videosu paketini düzelt ve EKSİKSİZ TAM JSON olarak geri ver.
Kullanıcı yalnızca konuyu verdi; ondan ek bilgi istenemez.

KONU:
{topic}

TESPİT EDİLEN SORUNLAR:
{issue_text}

ZORUNLU HEDEFLER:
- narration alanı {max(120, target_words - 20)} ile {target_words + 35} Türkçe kelime arasında olsun.
- Metin sakin, doğal, tarihsel açıdan temkinli ve seslendirmeye uygun olsun.
- İlk cümle dinleyiciyi zamana ve mekâna taşısın.
- Son cümle yumuşak bir kapanış yapsın.
- Tam {scene_count} sahne olsun.
- world_bible alanı dönem, coğrafya, mimari, kıyafet, malzeme, ışık, palet ve yasaklanan medeniyetleri tanımlasın.
- Her sahnede narration_text alanı bulunsun ve tüm narration_text alanları birleştiğinde narration metnini oluştursun.
- Her sahne narration sırasındaki somut fikri doğrudan görselleştirsin.
- transition yalnızca fade, dissolve, smoothleft, smoothright, hblur veya fadeblack olsun.
- transition_duration 0.45 ile 1.25 saniye arasında olsun.
- ambient_profile yalnızca exterior_wind, interior_room, firelight, archive_room, night_silence veya distant_storm olsun.
- duration_weight 0.55 ile 1.8 arasında olsun.
- Her image_prompt İngilizce, ayrıntılı, 16:9 sinematik tarih karesi tarif etsin.
- Müze objesi, beyaz fon, kolaj, yazı, sayı, logo, modern veya fantastik unsur olmasın.
- video_title, thumbnail_text, description, visual_identity, world_bible,
  thumbnail_prompt, thumbnail_negative_prompt, chapters ve tags alanlarını koru veya iyileştir.
- thumbnail_text en fazla dört kelime olsun.

MEVCUT JSON:
{existing}
"""
    return generate_json(client, prompt, max_tokens=12000)


def build_video_package(
    client: genai.Client,
    topic: str,
    target_seconds: int,
    scene_count: int,
) -> tuple[dict[str, Any], str]:
    target_words = max(170, min(260, round(target_seconds * 2.35)))
    minimum_words = max(110, round(target_words * 0.62))
    prompt = f"""
Yalnızca geçerli JSON üret. Markdown yazma.

ROL:
Sen; tarih araştırması, sakin belgesel senaryosu, sinematografik sahne
planlama ve YouTube paketleme alanlarında çalışan kıdemli bir editörsün.

KULLANICININ VERDİĞİ TEK BİLGİ:
{topic}

Sistem bu konuyu kendi anlayacak. Kullanıcıdan prompt, sahne, görsel, ses,
başlık veya montaj tercihi istemeyecek.

HEDEF:
Yaklaşık {target_seconds} saniyelik, uyku öncesi dinlemeye uygun, yayın
kalitesinde Türkçe tarih videosu.
Narration alanı {max(120, target_words - 20)} ile {target_words + 35} kelime arasında olsun.
Tam {scene_count} temiz ve birbirini takip eden görsel sahne.

JSON ŞEMASI:
{{
  "topic_interpretation": "Konunun nasıl ele alındığının kısa özeti",
  "historical_scope": "Dönem, yer ve bağlam",
  "video_title": "Merak uyandıran fakat abartısız Türkçe YouTube başlığı",
  "thumbnail_text": "En fazla dört kelime",
  "description": "İki kısa paragraf Türkçe açıklama",
  "narration": "Tek parça final Türkçe seslendirme metni",
  "visual_identity": "Bu videoya özgü tek cümlelik sanat yönetimi",
  "world_bible": {{
    "period": "Dönem",
    "location": "Coğrafya",
    "architecture": "Tutarlı mimari tanımı",
    "clothing": "Tutarlı kıyafet tanımı",
    "materials": "Taş, kerpiç, ahşap, bronz gibi malzemeler",
    "lighting": "Ay ışığı, yağ kandili, meşale gibi ışık kuralları",
    "palette": "Bütün videoda korunacak renk paleti",
    "forbidden": ["Yanlış medeniyet ve dönem unsurları"]
  }},
  "thumbnail_prompt": "Yazısız 16:9 kapak arka planı için ayrıntılı İngilizce prompt. Ana özne sağ tarafta, sol taraf koyu ve boş.",
  "thumbnail_negative_prompt": "Kapakta kesinlikle olmaması gereken unsurlar",
  "scenes": [
    {{
      "scene_id": 1,
      "narration_idea": "Bu sahne sırasında anlatılan ana fikir",
      "narration_text": "Bu sahnede okunacak final Türkçe cümleler",
      "visual_goal": "Görüntünün açıkça göstermesi gereken olay, mekân veya durum",
      "image_prompt": "Tek bir sinematik kare üretmek için ayrıntılı İngilizce prompt",
      "negative_prompt": "Bu sahneye özgü kaçınılacak unsurlar",
      "duration_weight": 1.0,
      "transition": "fade | dissolve | smoothleft | smoothright | hblur | fadeblack",
      "transition_duration": 0.72,
      "ambient_profile": "exterior_wind | interior_room | firelight | archive_room | night_silence | distant_storm",
      "importance": "high | normal"
    }}
  ],
  "chapters": ["Bölüm adı"],
  "tags": ["etiket"]
}}

ZORUNLU SENARYO KURALLARI:
- İlk cümlede dinleyiciyi doğrudan zamana ve mekâna taşı.
- Standart Türkiye Türkçesi kullan.
- Sakin, tok, doğal ve güven veren anlatım kur.
- Fragman, reklam, haber spikeri ve tiyatro tonundan kaçın.
- Bilinmeyen ayrıntıları kesin gerçek gibi yazma.
- Tarihsel belirsizlikleri kısa ve doğal biçimde belirt.
- Sayıları seslendirmeye uygun biçimde yazıyla yaz.
- Metnin sonunda yumuşak ve düşünceli bir kapanış yap.
- Aynı cümle yapısını ve aynı bilgiyi tekrarlama.

ZORUNLU GÖRSEL KURALLARI:
- Her image_prompt doğrudan o sahnede anlatılan şeyi göstermeli.
- Görsel sırası anlatının sırasını izlemeli; rastgele obje kataloğu olmasın.
- Öncelik: yaşayan mekân, çevre, mimari, günlük hayat ve olay atmosferi.
- Beyaz fonda müze objesi, katalog fotoğrafı ve bilgi kartı üretme.
- Aynı video boyunca tek film, tek dönem ve tek renk dünyası hissi koru.
- Tarihsel olarak olası mimari, kıyafet, eşya ve malzemeleri tarif et.
- Fantastik kule, fantastik zırh, modern nesne veya dekor kullanma.
- Görselin üzerinde yazı, harf, sayı, logo, altyazı veya filigran olmasın.
- Yakın yüz planlarını sınırlı tut; atmosferik geniş ve orta planlar kullan.
- Her sahne promptu; konu, dönem, yer, ışık, kompozisyon ve kamera açısını içersin.
- Tam olarak {scene_count} sahne üret.

ZORUNLU EDITOR BRAIN KURALLARI:
- narration metnini sahnelere anlamlı biçimde böl; her sahnenin narration_text alanı gerçek konuşma sırasını izlesin.
- Sahne süreleri eşit olmasın; duration_weight ile anlatım yoğunluğuna göre ritim kur.
- Aynı mekân veya fikir devam ediyorsa fade ya da dissolve kullan.
- Yeni mekâna geçiliyorsa smoothleft veya smoothright kullan.
- Zaman sıçraması, bölüm değişimi veya önemli kırılmada fadeblack kullan.
- Hblur yalnızca sis, hatıra, belirsizlik veya zihinsel geçişlerde kullanılmalı.
- Geçiş efekti gösteriş için değil anlatının anlamı için seçilmeli.
- Her sahne için ortam sesini ambient_profile ile belirle.
- world_bible bütün sahne promptlarında korunacak tek tarihî dünya referansıdır.
"""

    payload, model = generate_json(client, prompt, max_tokens=12000)
    model_history = [model]

    # Kısa veya eksik model yanıtını hata sayıp tamamen bırakmak yerine,
    # aynı paketi otomatik olarak genişletip onar.
    for repair_round in range(1, 4):
        issues = _package_issues(payload, scene_count, minimum_words)
        if not issues:
            _normalize_package(payload, scene_count)
            return payload, " -> ".join(model_history)

        print(
            f"Video paketi doğrulama turu {repair_round}/3: "
            + " | ".join(issues)
        )
        payload, repair_model = _repair_video_package(
            client,
            topic,
            payload,
            issues,
            target_words,
            scene_count,
        )
        model_history.append(repair_model)

    final_issues = _package_issues(payload, scene_count, minimum_words)
    if final_issues:
        raise ValueError(
            "Video paketi üç otomatik onarımdan sonra hâlâ geçersiz: "
            + " | ".join(final_issues)
        )

    _normalize_package(payload, scene_count)
    return payload, " -> ".join(model_history)


def validate_package(
    payload: dict[str, Any],
    scene_count: int,
    minimum_words: int = 110,
) -> None:
    """Backward-compatible validator used by older calls/tests."""
    issues = _package_issues(payload, scene_count, minimum_words)
    if issues:
        raise ValueError(" | ".join(issues))
    _normalize_package(payload, scene_count)

def _compact_text(value: Any, limit: int) -> str:
    clean = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(clean) <= limit:
        return clean
    return clean[: max(1, limit - 1)].rstrip(" ,.;:-") + "…"


def _fit_prompt(parts: list[str], limit: int = MAX_IMAGE_PROMPT_CHARS) -> str:
    clean_parts = [re.sub(r"\s+", " ", p).strip() for p in parts if str(p).strip()]
    result = ". ".join(clean_parts)
    if len(result) <= limit:
        return result
    budgets = [900, 410, 260, 230]
    reduced = [
        _compact_text(part, budgets[i] if i < len(budgets) else 160)
        for i, part in enumerate(clean_parts)
    ]
    return _compact_text(". ".join(reduced), limit)


def combined_prompt(payload: dict[str, Any], scene: dict[str, Any]) -> str:
    world = _world_bible_text(payload)
    return _fit_prompt([
        str(scene.get("image_prompt", "")),
        f"Strict historical continuity: {world}",
        f"Visual identity: {payload.get('visual_identity', '')}",
        "Photorealistic premium late-night historical documentary, cohesive film still, plausible period architecture and materials, blue-black moonlit shadows, restrained amber firelight, 16:9, no text or collage.",
    ])


def combined_negative(scene: dict[str, Any]) -> str:
    extra = str(scene.get("negative_prompt", "")).strip()
    return _compact_text(
        f"{GLOBAL_NEGATIVE}, {extra}" if extra else GLOBAL_NEGATIVE,
        MAX_NEGATIVE_PROMPT_CHARS,
    )


def deterministic_seed(topic: str, scene_id: int, attempt: int) -> int:
    digest = hashlib.sha256(f"{topic}|{scene_id}|{attempt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_000_000_000


def cloudflare_credentials() -> tuple[str, str]:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID bulunamadı.")
    if not api_token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN bulunamadı.")
    return account_id, api_token


def cloudflare_model_chain() -> list[str]:
    configured = os.getenv("CLOUDFLARE_IMAGE_MODEL", "").strip()
    models = [configured, *CLOUDFLARE_IMAGE_MODELS] if configured else list(CLOUDFLARE_IMAGE_MODELS)
    return list(dict.fromkeys(model for model in models if model))


def _cloudflare_error(response: requests.Response) -> RuntimeError:
    try:
        payload = response.json()
        errors = payload.get("errors") or []
        detail = "; ".join(
            str(item.get("message") or item.get("code") or item)
            for item in errors
            if isinstance(item, dict)
        )
        if not detail:
            detail = str(payload)[:800]
    except Exception:
        detail = response.text[:800] if response.text else "Yanıt gövdesi boş."
    return RuntimeError(
        f"Cloudflare Workers AI HTTP {response.status_code}: {detail}"
    )


def _decode_cloudflare_image(response: requests.Response) -> bytes:
    content_type = response.headers.get("content-type", "").lower()
    if content_type.startswith("image/"):
        return response.content

    try:
        payload = response.json()
    except Exception as exc:
        raise RuntimeError(
            f"Cloudflare görsel yerine çözülemeyen yanıt döndürdü: {content_type}"
        ) from exc

    if payload.get("success") is False:
        errors = payload.get("errors") or []
        raise RuntimeError(f"Cloudflare API hatası: {errors}")

    result = payload.get("result", payload)
    encoded = None
    if isinstance(result, dict):
        encoded = result.get("image") or result.get("b64_json") or result.get("base64")
    elif isinstance(result, str):
        encoded = result

    if not encoded:
        raise RuntimeError(f"Cloudflare geçerli görsel döndürmedi: {str(payload)[:800]}")

    if isinstance(encoded, str) and encoded.startswith("data:image"):
        encoded = encoded.split(",", 1)[-1]
    try:
        return base64.b64decode(encoded)
    except Exception as exc:
        raise RuntimeError("Cloudflare görsel base64 verisi çözülemedi.") from exc


def _save_generated_image(raw_bytes: bytes, target: Path) -> None:
    temp = target.with_suffix(".download")
    temp.write_bytes(raw_bytes)
    try:
        with Image.open(temp) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
            if image.width < 700 or image.height < 390:
                raise ValueError(
                    f"Üretilen görsel çözünürlüğü düşük: {image.width}x{image.height}"
                )
            image.save(target, "JPEG", quality=94, optimize=True)
    finally:
        temp.unlink(missing_ok=True)


def _cloudflare_request_body(
    model: str,
    prompt: str,
    negative: str,
    seed: int,
) -> tuple[dict[str, str], dict[str, str] | None]:
    """Return multipart form data or JSON and hard-limit prompt lengths."""
    prompt = _compact_text(prompt, MAX_IMAGE_PROMPT_CHARS)
    negative = _compact_text(negative, MAX_NEGATIVE_PROMPT_CHARS)
    final_prompt = _fit_prompt(
        [prompt, f"Avoid: {negative}", "Landscape 16:9, clean cinematic composition."],
        limit=MAX_IMAGE_PROMPT_CHARS,
    )
    print(
        f"Cloudflare prompt length: model={model}, "
        f"prompt={len(final_prompt)}, negative={len(negative)}"
    )

    if "flux-2-klein" in model or "flux-2-dev" in model:
        form = {
            "prompt": final_prompt,
            "width": "1344",
            "height": "768",
            "guidance": "4.5",
            "seed": str(seed),
        }
        return form, None

    if "flux-1-schnell" in model:
        body = {"prompt": final_prompt, "seed": seed, "steps": 8}
        return {}, body

    body = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": 1024,
        "height": 576,
        "num_steps": 8 if "dreamshaper" in model else 20,
        "guidance": 7.5,
        "seed": seed,
    }
    return {}, body


def cloudflare_image_request(
    prompt: str,
    negative: str,
    seed: int,
    target: Path,
    model: str,
) -> None:
    account_id, api_token = cloudflare_credentials()
    endpoint = f"{CLOUDFLARE_API_BASE}/{account_id}/ai/run/{model}"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json, image/*",
    }
    form_data, json_body = _cloudflare_request_body(model, prompt, negative, seed)

    if form_data:
        response = requests.post(
            endpoint,
            data=form_data,
            headers=headers,
            timeout=(20, 180),
        )
    else:
        response = requests.post(
            endpoint,
            json=json_body,
            headers={**headers, "Content-Type": "application/json"},
            timeout=(20, 180),
        )

    if not response.ok:
        raise _cloudflare_error(response)
    _save_generated_image(_decode_cloudflare_image(response), target)

def image_review(
    client: genai.Client,
    scene: dict[str, Any],
    image_path: Path,
) -> tuple[bool, int, str]:
    data = image_path.read_bytes()
    prompt = f"""
Yalnızca JSON üret:
{{
  "pass": true,
  "score": 0,
  "reason": "Kısa açıklama"
}}

Bu yapay zekâ görselini aşağıdaki sahne amacıyla karşılaştır:
SAHNE AMACI: {scene['visual_goal']}
ANLATILAN FİKİR: {scene['narration_idea']}

Kriterler:
- Sahne amacını açıkça gösteriyor mu?
- Modern, fantastik veya dönem dışı unsur var mı?
- Görselde yazı, sayı, logo, filigran veya çerçeve var mı?
- Belirgin anatomi bozukluğu veya yapay katalog görünümü var mı?
- Premium tarih belgeseli karesi gibi görünüyor mu?

score 0-100 olsun. pass yalnızca score 64 veya üzerindeyse true olsun.
"""
    part = types.Part.from_bytes(data=data, mime_type="image/jpeg")
    try:
        payload, _ = generate_json_with_parts(client, prompt, part)
        score = int(payload.get("score", 0))
        passed = bool(payload.get("pass")) and score >= 64
        return passed, score, str(payload.get("reason", ""))
    except Exception as exc:
        # Görsel değerlendirme servisi geçici olarak çalışmazsa üretimi durdurma.
        print("Görsel kalite değerlendirmesi atlandı:", exc)
        return True, 70, "Otomatik inceleme geçici olarak atlandı."


def generate_json_with_parts(
    client: genai.Client,
    prompt: str,
    image_part: types.Part,
) -> tuple[dict[str, Any], str]:
    configured = os.getenv("TEXT_MODEL", "").strip()
    chain = model_chain(client, [configured, *TEXT_MODELS])
    last_error: Exception | None = None

    for model in chain:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt), image_part],
                    )
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.15,
                    max_output_tokens=1024,
                ),
            )
            return safe_json(response.text or ""), model
        except Exception as exc:
            last_error = exc
            print(f"Görsel inceleme modeli başarısız: {model}: {exc}")
            continue

    raise RuntimeError(f"Görsel inceleme başarısız: {last_error}")


def generate_scene_image(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    scene: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    models = cloudflare_model_chain()
    last_error: Exception | None = None
    max_attempts = min(4, len(models))
    best_score = -1
    best_reason = ""
    best_model = ""
    best_seed = 0
    best_file = WORK / f"best-scene-{int(scene['scene_id']):02d}.jpg"
    best_file.unlink(missing_ok=True)

    for attempt, model in enumerate(models[:max_attempts], start=1):
        seed = deterministic_seed(topic, int(scene["scene_id"]), attempt)
        try:
            print(
                f"Sahne görseli: {scene['scene_id']}, model={model}, "
                f"deneme={attempt}/{max_attempts}"
            )
            cloudflare_image_request(
                combined_prompt(payload, scene), combined_negative(scene), seed, target, model
            )
            passed, score, reason = image_review(client, scene, target)
            print(f"Görsel değerlendirmesi: pass={passed}; score={score}; {reason}")

            if score > best_score:
                shutil.copyfile(target, best_file)
                best_score = score
                best_reason = reason
                best_model = model
                best_seed = seed

            if passed:
                return {
                    "scene_id": scene["scene_id"],
                    "model": model,
                    "seed": seed,
                    "review_score": score,
                    "review": reason,
                    "file": target.name,
                }
            target.unlink(missing_ok=True)
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            print(f"Sahne görseli başarısız ({model}): {exc}")
            message = str(exc).lower()
            if "10,000 neurons" in message or "account limited" in message:
                raise RuntimeError(
                    "Cloudflare günlük ücretsiz görsel kotası dolmuş. "
                    "Kota 00:00 UTC'de yenilenir."
                ) from exc
            time.sleep(min(8, 2 * attempt))

    if best_file.exists() and best_score >= 52:
        shutil.copyfile(best_file, target)
        return {
            "scene_id": scene["scene_id"],
            "model": best_model,
            "seed": best_seed,
            "review_score": best_score,
            "review": f"En iyi aday kullanıldı: {best_reason}",
            "quality_fallback": True,
            "file": target.name,
        }

    raise RuntimeError(
        f"Sahne {scene['scene_id']} için uygun görsel üretilemedi: {last_error}"
    )


def generate_thumbnail_background(
    topic: str,
    payload: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    prompt = (
        f"{payload['thumbnail_prompt'].strip()}\n\n"
        f"VIDEO VISUAL IDENTITY:\n{payload['visual_identity'].strip()}\n\n"
        f"MASTER STYLE:\n{STYLE_BIBLE}\n"
        "YouTube thumbnail background, main focal subject on the right third, "
        "dark clean negative space on the left third for typography."
    )
    negative = (
        f"{GLOBAL_NEGATIVE}, "
        f"{str(payload.get('thumbnail_negative_prompt', '')).strip()}, "
        "words, title, typography"
    )

    last_error: Exception | None = None
    models = cloudflare_model_chain()
    max_attempts = min(4, len(models))
    for attempt, model in enumerate(models[:max_attempts], start=1):
        seed = deterministic_seed(topic, 999, attempt)
        try:
            print(f"Kapak arka planı: model={model}, deneme={attempt}/{max_attempts}")
            cloudflare_image_request(prompt, negative, seed, target, model)
            return {"model": model, "seed": seed, "file": target.name}
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            print(f"Kapak arka planı başarısız ({model}): {exc}")
            time.sleep(min(12, 3 * attempt))

    raise RuntimeError(f"Kapak arka planı üretilemedi: {last_error}")

def review_thumbnail_candidate(
    client: genai.Client,
    image_path: Path,
    title_text: str,
    topic: str,
    visual_identity: str,
) -> tuple[int, str]:
    data = image_path.read_bytes()
    prompt = f"""
Yalnızca JSON üret:
{{
  "score": 0,
  "reason": "Kısa açıklama"
}}

Bu görsel aşağıdaki YouTube tarih videosunun kapak arka planı adayıdır.
KONU: {topic}
BAŞLIK METNİ: {title_text}
GÖRSEL KİMLİK: {visual_identity}

Puanlama kriterleri:
- Konuyu ve doğru tarihsel dönemi belirgin biçimde çağrıştırıyor mu?
- Başka medeniyetlere ait jenerik bir yapı gibi görünmekten kaçınıyor mu?
- Premium, sinematik ve mobil ekranda güçlü mü?
- Sağ tarafta tek güçlü ana odak, sol tarafta temiz ve koyu başlık alanı var mı?
- Yazı, logo, filigran, kolaj, aşırı kalabalık veya dönem dışı unsur var mı?
- Videonun mavi-siyah ay ışığı ve kısık amber meşale paletiyle uyumlu mu?

0-100 arasında puan ver. 72 altı zayıf kabul edilir.
"""
    part = types.Part.from_bytes(data=data, mime_type="image/jpeg")
    try:
        payload, _ = generate_json_with_parts(client, prompt, part)
        return int(payload.get("score", 0)), str(payload.get("reason", ""))
    except Exception as exc:
        print("Kapak değerlendirmesi atlandı:", exc)
        return 65, "Kapak değerlendirmesi atlandı."


def generate_thumbnail_candidates(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    target_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    base_prompt = str(payload["thumbnail_prompt"]).strip()
    visual_identity = str(payload["visual_identity"]).strip()
    variants = [
        "A single iconic period-specific gateway or landmark, low camera angle, monumental but historically plausible, main subject on right.",
        "A wide atmospheric night view of the exact historical place, restrained torchlight, deep perspective, strong subject on right.",
        "A closer architectural detail unique to the topic, dramatic moonlight, clear silhouette, cinematic scale, dark left third.",
        "A quiet human-scale moment at the historical location with one small silhouette for scale, architecture dominant, clean left third.",
    ]

    best_path: Path | None = None
    best_score = -1
    best_info: dict[str, Any] = {}

    for idx, extra in enumerate(variants, start=1):
        target = target_dir / f"thumbnail_candidate_{idx}.jpg"
        prompt = (
            f"{base_prompt}\n\nTOPIC: {topic}\n\n{extra}\n\n"
            f"VIDEO VISUAL IDENTITY:\n{visual_identity}\n\n"
            f"MASTER STYLE:\n{STYLE_BIBLE}\n"
            "YouTube thumbnail background only. Topic-specific architecture, clothing and iconography. "
            "Main focal subject on the right third. Left third darker, calmer and uncluttered for typography. "
            "Single cohesive environment, no collage, no written symbols and no generic Egyptian, Greek or Roman substitution unless the topic requires it."
        )
        negative = (
            f"{GLOBAL_NEGATIVE}, "
            f"{str(payload.get('thumbnail_negative_prompt', '')).strip()}, "
            "words, title, typography, collage, multiple panels, busy left side, generic ancient temple, wrong civilization"
        )
        info = generate_thumbnail_background_with_prompt(
            topic, prompt, negative, target, candidate_id=idx
        )
        score, reason = review_thumbnail_candidate(
            client,
            target,
            str(payload.get("thumbnail_text", "")),
            topic,
            visual_identity,
        )
        info["review_score"] = score
        info["review_reason"] = reason
        if score > best_score:
            best_score = score
            best_path = target
            best_info = info

    if best_path is None:
        raise RuntimeError("Kapak adayı seçilemedi.")
    best_info["selected_file"] = best_path.name
    return best_path, best_info


def generate_thumbnail_background_with_prompt(
    topic: str,
    prompt: str,
    negative: str,
    target: Path,
    candidate_id: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    models = cloudflare_model_chain()
    max_attempts = min(4, len(models))
    for attempt, model in enumerate(models[:max_attempts], start=1):
        seed = deterministic_seed(topic, 900 + candidate_id, attempt)
        try:
            print(f"Kapak adayı {candidate_id}: model={model}, deneme={attempt}/{max_attempts}")
            cloudflare_image_request(prompt, negative, seed, target, model)
            return {"model": model, "seed": seed, "file": target.name, "candidate_id": candidate_id}
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            print(f"Kapak adayı başarısız ({model}): {exc}")
            time.sleep(min(12, 3 * attempt))

    raise RuntimeError(f"Kapak adayı üretilemedi: {last_error}")


def video_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold else
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def clean_video_frame(source: Path, target: Path) -> None:
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

    frame = ImageOps.fit(
        image,
        (WIDTH, HEIGHT),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )

    # Match every scene to a common night-documentary exposure.
    gray = ImageOps.grayscale(frame)
    mean_luma = sum(ImageStat.Stat(gray).mean) / 1.0
    target_luma = 72.0
    exposure_gain = max(0.78, min(1.22, target_luma / max(1.0, mean_luma)))
    frame = ImageEnhance.Brightness(frame).enhance(exposure_gain)
    frame = ImageEnhance.Color(frame).enhance(0.78)
    frame = ImageEnhance.Contrast(frame).enhance(1.07)

    # Fixed blue-shadow / amber-highlight grade. This is static, not animated.
    luminance = ImageOps.grayscale(frame)
    grade = ImageOps.colorize(
        luminance,
        black=(13, 22, 36),
        mid=(91, 88, 91),
        white=(216, 190, 151),
    ).convert("RGB")
    frame = Image.blend(frame, grade, 0.30)

    # Fixed vignette and subtle matte. No per-frame grain or animated noise.
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(7, 9, 13, 20))
    for inset, alpha in ((0, 38), (55, 25), (120, 14)):
        draw.rounded_rectangle(
            (inset, inset, WIDTH - inset, HEIGHT - inset),
            radius=70,
            outline=(0, 0, 0, alpha),
            width=90,
        )

    frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    frame.save(target, "JPEG", quality=95, optimize=True)


def make_storyboard(
    frames: list[Path],
    scenes: list[dict[str, Any]],
    target: Path,
) -> None:
    # Storyboard yalnızca kontrol dosyasıdır; final videoya girmez.
    cols = 3
    cell_w, image_h, caption_h = 640, 360, 66
    rows = math.ceil(len(frames) / cols)
    board = Image.new("RGB", (cols * cell_w, rows * (image_h + caption_h)), (18, 16, 14))
    draw = ImageDraw.Draw(board)

    for index, frame_path in enumerate(frames):
        row, col = divmod(index, cols)
        x = col * cell_w
        y = row * (image_h + caption_h)
        with Image.open(frame_path) as raw:
            image = ImageOps.fit(raw.convert("RGB"), (cell_w, image_h), Image.Resampling.LANCZOS)
        board.paste(image, (x, y))
        title = str(scenes[index].get("visual_goal", f"Sahne {index + 1}"))
        title = textwrap.shorten(title, width=56, placeholder="…")
        draw.rectangle((x, y + image_h, x + cell_w, y + image_h + caption_h), fill=(23, 20, 17))
        draw.text(
            (x + 18, y + image_h + 19),
            title,
            font=video_font(18, bold=True),
            fill=(228, 219, 201),
        )
    board.save(target, "JPEG", quality=92, optimize=True)


def wrap_title(draw: ImageDraw.ImageDraw, text: str, max_width: int, font_obj) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().upper().split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=font_obj)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def make_thumbnail(background: Path, text: str, target: Path) -> None:
    with Image.open(background) as raw:
        image = ImageOps.fit(raw.convert("RGB"), (1280, 720), Image.Resampling.LANCZOS)

    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = ImageEnhance.Color(image).enhance(0.90)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    for x in range(780):
        alpha = int(215 * (1 - x / 780) ** 1.6)
        draw_overlay.line((x, 0, x, 720), fill=(9, 8, 7, alpha))
    draw_overlay.rectangle((0, 0, 1280, 720), fill=(0, 0, 0, 18))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(image)
    title_font = video_font(82, bold=True)
    lines = wrap_title(draw, text, 600, title_font)
    total_h = len(lines) * 94
    start_y = max(135, (720 - total_h) // 2 - 20)

    for line_index, line in enumerate(lines):
        y = start_y + line_index * 94
        draw.text(
            (68, y),
            line,
            font=title_font,
            fill=(247, 239, 222),
            stroke_width=3,
            stroke_fill=(12, 10, 8),
        )

    draw.rectangle((70, start_y - 34, 160, start_y - 26), fill=(204, 174, 127, 255))
    draw.text(
        (70, 632),
        "UYKU VE TARİH",
        font=video_font(24, bold=True),
        fill=(210, 188, 151),
    )
    image.convert("RGB").save(target, "JPEG", quality=95, optimize=True)


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)


def ffprobe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(1.0, float(result.stdout.strip()))


def tts_directive(narration: str) -> str:
    return f"""
Read only the Turkish transcript after the separator.

VOICE:
Mature Turkish male documentary narrator. Calm, grounded, warm and natural.
Standard Turkey Turkish pronunciation. Close studio microphone.

PERFORMANCE:
Late-night historical documentary for relaxed listening.
Slow but conversational. Stable low energy. Gentle sentence endings.
Natural short pauses. Clear articulation without theatrical emphasis.
Never sound like an advertisement, trailer, newsreader or stage actor.
Do not whisper. Do not add or remove words. No music or sound effects.

--- TRANSCRIPT ---
{narration}
--- END ---
""".strip()


def synthesize_narration(
    client: genai.Client,
    narration: str,
    target: Path,
) -> str:
    configured = os.getenv("TTS_MODEL", "").strip()
    chain = model_chain(client, [configured, *TTS_MODELS])
    last_error: Exception | None = None

    for model in chain:
        for attempt, delay in enumerate((8, 25, 55), start=1):
            try:
                print(f"TTS: model={model}, deneme={attempt}/3")
                response = client.models.generate_content(
                    model=model,
                    contents=tts_directive(narration),
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=VOICE_NAME
                                )
                            )
                        ),
                    ),
                )
                candidates = response.candidates or []
                if not candidates or not candidates[0].content.parts:
                    raise ValueError("TTS ses parçası döndürmedi.")
                inline = candidates[0].content.parts[0].inline_data
                data = inline.data if inline else None
                if not data or len(data) < 48000:
                    raise ValueError("TTS verisi boş veya çok kısa.")
                write_wav(target, data)
                if ffprobe_duration(target) < 35:
                    raise ValueError("TTS sesi beklenenden kısa.")
                return model
            except Exception as exc:
                last_error = exc
                target.unlink(missing_ok=True)
                print(f"TTS hatası: {model}: {exc}")
                message = str(exc).lower()
                if "404" in message or "not found" in message or "no longer available" in message:
                    break
                if retryable(exc) and attempt < 3:
                    time.sleep(delay)
                    continue
                break

    raise RuntimeError(f"Ses üretilemedi. Son hata: {last_error}")


def normalize_audio(source: Path, target: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-af",
            "highpass=f=55,lowpass=f=14500,"
            "acompressor=threshold=-21dB:ratio=2.0:attack=24:release=190,"
            "loudnorm=I=-17:TP=-2:LRA=7",
            "-ar", "48000", "-c:a", "pcm_s16le",
            str(target),
        ]
    )


def allocate_scene_durations(
    scenes: list[dict[str, Any]],
    total_duration: float,
) -> list[float]:
    if not scenes:
        return []

    weights: list[float] = []
    for scene in scenes:
        words = max(4, _word_count(scene.get("narration_text", "")))
        punctuation = len(re.findall(r"[,:;.!?]", str(scene.get("narration_text", ""))))
        importance = 1.12 if scene.get("importance") == "high" else 1.0
        explicit = float(scene.get("duration_weight", 1.0))
        weights.append((words + punctuation * 0.55) * importance * explicit)

    minimum = 4.2
    maximum = 15.0
    durations = [total_duration * weight / sum(weights) for weight in weights]

    for _ in range(5):
        durations = [max(minimum, min(maximum, item)) for item in durations]
        difference = total_duration - sum(durations)
        if abs(difference) < 0.02:
            break
        adjustable = [
            idx for idx, item in enumerate(durations)
            if (difference > 0 and item < maximum - 0.01)
            or (difference < 0 and item > minimum + 0.01)
        ]
        if not adjustable:
            break
        share = difference / len(adjustable)
        for idx in adjustable:
            durations[idx] += share

    scale = total_duration / max(0.001, sum(durations))
    return [max(0.5, item * scale) for item in durations]


def scene_transition_plan(scenes: list[dict[str, Any]]) -> tuple[list[str], list[float]]:
    names: list[str] = []
    durations: list[float] = []
    for scene in scenes[:-1]:
        name = str(scene.get("transition", "fade")).strip().lower()
        if name not in ALLOWED_TRANSITIONS:
            name = "fade"
        try:
            duration = float(scene.get("transition_duration", 0.72))
        except (TypeError, ValueError):
            duration = 0.72
        names.append(name)
        durations.append(max(0.45, min(1.25, duration)))
    return names, durations


def ambient_filter(profile: str, duration: float) -> str:
    fade_out = max(0.0, duration - 1.0)
    filters = {
        "exterior_wind": "highpass=f=75,lowpass=f=950,volume=0.065",
        "interior_room": "highpass=f=35,lowpass=f=310,volume=0.035",
        "firelight": "highpass=f=140,lowpass=f=2600,tremolo=f=0.55:d=0.22,volume=0.038",
        "archive_room": "highpass=f=45,lowpass=f=520,volume=0.032",
        "distant_storm": "highpass=f=35,lowpass=f=780,volume=0.060",
        "night_silence": "highpass=f=35,lowpass=f=240,volume=0.022",
    }
    selected = filters.get(profile, filters["night_silence"])
    return (
        f"{selected},"
        f"afade=t=in:st=0:d=0.9,"
        f"afade=t=out:st={fade_out:.3f}:d=1.0,"
        "aformat=sample_rates=48000:channel_layouts=stereo"
    )


def generate_ambient_segment(profile: str, duration: float, target: Path) -> None:
    color = "brown" if profile in {"interior_room", "archive_room", "distant_storm"} else "pink"
    run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"anoisesrc=color={color}:amplitude=0.18:sample_rate=48000:d={duration:.3f}",
            "-af", ambient_filter(profile, duration),
            "-c:a", "pcm_s16le",
            str(target),
        ]
    )


def build_ambient_track(
    scenes: list[dict[str, Any]],
    visible_durations: list[float],
    transition_durations: list[float],
    target: Path,
) -> None:
    ambient_dir = WORK / "ambient-scenes"
    ambient_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Path] = []

    for index, (scene, visible) in enumerate(zip(scenes, visible_durations)):
        extra = transition_durations[index] if index < len(transition_durations) else 0.0
        segment_duration = visible + extra
        segment = ambient_dir / f"ambient_{index + 1:03d}.wav"
        generate_ambient_segment(
            str(scene.get("ambient_profile", "night_silence")),
            segment_duration,
            segment,
        )
        segments.append(segment)

    if len(segments) == 1:
        shutil.copy2(segments[0], target)
        return

    command = ["ffmpeg", "-y"]
    for segment in segments:
        command += ["-i", str(segment)]

    filters: list[str] = []
    current = "[0:a]"
    for index in range(1, len(segments)):
        out = f"[a{index}]"
        crossfade = transition_durations[index - 1]
        filters.append(
            f"{current}[{index}:a]acrossfade=d={crossfade:.3f}:c1=tri:c2=tri{out}"
        )
        current = out
    final = "[ambient]"
    filters.append(f"{current}volume=0.72{final}")

    command += [
        "-filter_complex", ";".join(filters),
        "-map", final,
        "-c:a", "pcm_s16le",
        str(target),
    ]
    run(command)


def mix_narration_and_ambient(
    narration: Path,
    ambient: Path,
    target: Path,
) -> None:
    run(
        [
            "ffmpeg", "-y",
            "-i", str(narration),
            "-i", str(ambient),
            "-filter_complex",
            "[1:a]volume=0.34[amb];"
            "[amb][0:a]sidechaincompress=threshold=0.020:ratio=8:attack=25:release=360[ducked];"
            "[0:a][ducked]amix=inputs=2:weights='1 1':normalize=0,"
            "alimiter=limit=0.95,loudnorm=I=-17:TP=-2:LRA=7[aout]",
            "-map", "[aout]",
            "-ar", "48000",
            "-c:a", "pcm_s16le",
            str(target),
        ]
    )


def write_edit_timeline(
    scenes: list[dict[str, Any]],
    visible_durations: list[float],
    transitions: list[str],
    transition_durations: list[float],
    target: Path,
) -> None:
    cursor = 0.0
    timeline: list[dict[str, Any]] = []
    for index, (scene, duration) in enumerate(zip(scenes, visible_durations)):
        item = {
            "scene_id": scene.get("scene_id", index + 1),
            "start": round(cursor, 3),
            "end": round(cursor + duration, 3),
            "duration": round(duration, 3),
            "narration_text": scene.get("narration_text", ""),
            "ambient_profile": scene.get("ambient_profile", "night_silence"),
            "transition_to_next": transitions[index] if index < len(transitions) else None,
            "transition_duration": transition_durations[index] if index < len(transition_durations) else 0.0,
        }
        timeline.append(item)
        cursor += duration
    target.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")


def scene_filter(motion: str, frames: int, seconds: float) -> str:
    # ZERO-JITTER MODE: no zoom, pan, grain, noise or animated sharpening.
    # All perceived motion comes only from controlled scene transitions.
    return (
        f"scale={WIDTH}:{HEIGHT}:flags=lanczos,"
        "setsar=1,format=yuv420p"
    )



def make_editorial_shots(
    frames: list[Path],
    scenes: list[dict[str, Any]],
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Create a richer timeline from each generated image without adding motion.

    The final video remains completely static inside each shot. We only create
    different crops: wide, detail, and atmosphere. This gives editor-like rhythm
    without shimmer or zoom jitter.
    """
    shot_dir = WORK / "editorial-shots"
    if shot_dir.exists():
        shutil.rmtree(shot_dir)
    shot_dir.mkdir(parents=True, exist_ok=True)

    shots: list[Path] = []
    shot_meta: list[dict[str, Any]] = []

    for index, (frame_path, scene) in enumerate(zip(frames, scenes), start=1):
        with Image.open(frame_path) as raw:
            base = ImageOps.exif_transpose(raw).convert("RGB")

        # 1) wide shot
        wide = ImageOps.fit(base, (WIDTH, HEIGHT), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        wide_path = shot_dir / f"s{index:02d}_a_wide.jpg"
        wide.save(wide_path, "JPEG", quality=94, optimize=True)
        shots.append(wide_path)
        shot_meta.append({
            "source_scene": index,
            "shot_type": "wide",
            "transition": scene.get("transition", "dissolve"),
            "audio_bed": scene.get("audio_bed", "night_wind"),
            "importance": scene.get("importance", "normal"),
            "duration_weight": 1.25 if scene.get("importance") == "high" else 1.0,
        })

        # 2) detail crop, alternating focal direction
        centering = (0.34, 0.50) if index % 2 else (0.66, 0.50)
        detail = ImageOps.fit(base, (WIDTH, HEIGHT), Image.Resampling.LANCZOS, centering=centering)
        detail = ImageEnhance.Contrast(detail).enhance(1.03)
        detail_path = shot_dir / f"s{index:02d}_b_detail.jpg"
        detail.save(detail_path, "JPEG", quality=94, optimize=True)
        shots.append(detail_path)
        shot_meta.append({
            "source_scene": index,
            "shot_type": "detail",
            "transition": "cut" if index % 3 else "dissolve",
            "audio_bed": scene.get("audio_bed", "room_tone"),
            "importance": scene.get("importance", "normal"),
            "duration_weight": 0.62,
        })

        # 3) atmosphere crop for important/closing scenes only
        if index == 1 or index == len(frames) or scene.get("importance") == "high":
            center_y = 0.42 if index % 2 else 0.56
            atmosphere = ImageOps.fit(base, (WIDTH, HEIGHT), Image.Resampling.LANCZOS, centering=(0.5, center_y))
            atmosphere = ImageEnhance.Brightness(atmosphere).enhance(0.92)
            atmos_path = shot_dir / f"s{index:02d}_c_atmosphere.jpg"
            atmosphere.save(atmos_path, "JPEG", quality=94, optimize=True)
            shots.append(atmos_path)
            shot_meta.append({
                "source_scene": index,
                "shot_type": "atmosphere",
                "transition": "fadeblack" if index == len(frames) else "dissolve",
                "audio_bed": scene.get("audio_bed", "night_wind"),
                "importance": scene.get("importance", "high"),
                "duration_weight": 0.95 if index != len(frames) else 1.45,
            })

    return shots, shot_meta


def make_intro_card(thumbnail: Path, target: Path, topic_title: str) -> None:
    with Image.open(thumbnail) as raw:
        image = ImageOps.fit(raw.convert("RGB"), (WIDTH, HEIGHT), Image.Resampling.LANCZOS)

    image = ImageEnhance.Brightness(image).enhance(0.55)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 88))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(image)
    title = re.sub(r"\s+", " ", topic_title).strip().upper()
    if len(title) > 44:
        title = textwrap.shorten(title, width=44, placeholder="…").upper()

    title_font = video_font(58, bold=True)
    small_font = video_font(25, bold=True)

    # Minimal editorial intro, no noisy animation. Fade is applied in ffmpeg.
    draw.rectangle((120, 375, 265, 385), fill=(215, 184, 134, 255))
    draw.text(
        (120, 415),
        "UYKU VE TARİH",
        font=small_font,
        fill=(216, 197, 163),
        stroke_width=1,
        stroke_fill=(8, 7, 6),
    )
    lines = []
    current = ""
    for word in title.split():
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=title_font)[2] <= 1180:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    for i, line in enumerate(lines[:2]):
        draw.text(
            (120, 470 + i * 68),
            line,
            font=title_font,
            fill=(248, 240, 224),
            stroke_width=2,
            stroke_fill=(10, 9, 8),
        )

    image.convert("RGB").save(target, "JPEG", quality=95, optimize=True)


def audio_bed_filter(kind: str, duration: float) -> str:
    # Procedural atmosphere beds. Very low level, designed to be felt, not heard.
    if kind == "fire":
        return (
            f"anoisesrc=color=brown:amplitude=0.020:duration={duration:.3f},"
            "highpass=f=450,lowpass=f=2400,volume=0.026"
        )
    if kind == "interior":
        return (
            f"anoisesrc=color=pink:amplitude=0.012:duration={duration:.3f},"
            "lowpass=f=420,volume=0.018"
        )
    if kind == "storm":
        return (
            f"anoisesrc=color=blue:amplitude=0.018:duration={duration:.3f},"
            "lowpass=f=900,volume=0.024"
        )
    if kind == "archive":
        return (
            f"anoisesrc=color=pink:amplitude=0.010:duration={duration:.3f},"
            "highpass=f=80,lowpass=f=650,volume=0.016"
        )
    return (
        f"anoisesrc=color=pink:amplitude=0.014:duration={duration:.3f},"
        "highpass=f=120,lowpass=f=1000,volume=0.020"
    )


def create_sound_design(shot_meta: list[dict[str, Any]], duration: float, target: Path) -> None:
    # Create a continuous restrained atmosphere bed with slow fades.
    bed = WORK / "sound-bed.wav"
    base = (
        f"anoisesrc=color=pink:amplitude=0.013:duration={duration:.3f},"
        "highpass=f=85,lowpass=f=950,"
        "afade=t=in:st=0:d=3,"
        f"afade=t=out:st={max(0, duration-4):.3f}:d=4,"
        "volume=0.022"
    )
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", base,
        "-ar", "48000", "-c:a", "pcm_s16le",
        str(bed),
    ])
    target.write_bytes(bed.read_bytes())


def render_intro_clip(intro_frame: Path, target: Path, seconds: float = 4.2) -> None:
    frames_count = max(1, math.ceil(seconds * FPS))
    vf = (
        "scale=1920:1080,"
        "fade=t=in:st=0:d=1.05,"
        f"fade=t=out:st={max(0, seconds-0.9):.3f}:d=0.9,"
        "format=yuv420p"
    )
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(intro_frame),
        "-vf", vf,
        "-frames:v", str(frames_count),
        "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "19", "-pix_fmt", "yuv420p",
        str(target),
    ])


def render_static_clip(frame: Path, target: Path, seconds: float) -> None:
    frames_count = max(1, math.ceil(seconds * FPS))
    vf = (
        "scale=1920:1080,"
        "eq=saturation=0.94:contrast=1.01:brightness=-0.006,"
        "vignette=PI/8.8,"
        "format=yuv420p"
    )
    run([
        "ffmpeg", "-y",
        "-loop", "1", "-i", str(frame),
        "-vf", vf,
        "-frames:v", str(frames_count),
        "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "19", "-pix_fmt", "yuv420p",
        str(target),
    ])


def editorial_transition(prev_type: str, next_type: str, index: int) -> tuple[str, float]:
    # Most transitions are cuts/dissolves; no presentation-like slide effects.
    if next_type == "detail":
        return "cut", 0.0
    if index % 6 == 0:
        return "fadeblack", 0.45
    return "fade", 0.38

def render_video(
    frames: list[Path],
    scenes: list[dict[str, Any]],
    audio: Path,
    target: Path,
    visible_durations: list[float],
    transitions: list[str],
    transition_durations: list[float],
) -> float:
    duration = ffprobe_duration(audio)
    clips_dir = WORK / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for index, (frame, visible) in enumerate(zip(frames, visible_durations), start=1):
        extra = transition_durations[index - 1] if index - 1 < len(transition_durations) else 0.0
        clip_duration = visible + extra
        clip = clips_dir / f"clip_{index:03d}.mp4"
        run(
            [
                "ffmpeg", "-y",
                "-loop", "1",
                "-framerate", str(FPS),
                "-i", str(frame),
                "-t", f"{clip_duration:.3f}",
                "-vf", scene_filter("static", 0, clip_duration),
                "-an",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-tune", "stillimage",
                "-pix_fmt", "yuv420p",
                "-r", str(FPS),
                str(clip),
            ]
        )
        clips.append(clip)

    if len(clips) == 1:
        run(
            [
                "ffmpeg", "-y",
                "-i", str(clips[0]),
                "-i", str(audio),
                "-filter_complex",
                f"[0:v]fade=t=in:st=0:d=1.0,fade=t=out:st={max(0.0, duration - 1.2):.3f}:d=1.2[vout]",
                "-map", "[vout]",
                "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest", "-movflags", "+faststart",
                str(target),
            ]
        )
        return duration

    filter_parts: list[str] = []
    current_label = "[0:v]"
    offset = visible_durations[0]

    for index in range(1, len(clips)):
        out_label = f"[vx{index}]"
        transition = transitions[index - 1]
        transition_duration = transition_durations[index - 1]
        filter_parts.append(
            f"{current_label}[{index}:v]"
            f"xfade=transition={transition}:duration={transition_duration:.3f}:"
            f"offset={offset:.3f}{out_label}"
        )
        current_label = out_label
        offset += visible_durations[index]

    final_label = "[vfinal]"
    filter_parts.append(
        f"{current_label}"
        f"fade=t=in:st=0:d=1.0,"
        f"fade=t=out:st={max(0.0, duration - 1.25):.3f}:d=1.25"
        f"{final_label}"
    )

    command = ["ffmpeg", "-y"]
    for clip in clips:
        command += ["-i", str(clip)]
    command += ["-i", str(audio)]
    command += [
        "-filter_complex", ";".join(filter_parts),
        "-map", final_label,
        "-map", f"{len(clips)}:a:0",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(target),
    ]
    run(command)
    return duration


def timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def create_srt(narration: str, duration: float, target: Path) -> None:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", narration.strip())
        if item.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= 145:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    weights = [max(1, len(chunk)) for chunk in chunks]
    total = sum(weights)
    cursor = 0.0
    blocks = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
        piece = duration * weight / total
        end = min(duration, cursor + piece)
        blocks.append(
            f"{index}\n{timestamp(cursor)} --> {timestamp(end)}\n{chunk}\n"
        )
        cursor = end
    target.write_text("\n".join(blocks), encoding="utf-8")


def chapter_text(chapters: list[str], duration: float) -> list[str]:
    clean = [str(item).strip() for item in chapters if str(item).strip()]
    if not clean:
        clean = ["Geceye giriş", "Tarihî dünyanın içinde", "Sessiz kapanış"]
    spacing = duration / len(clean)
    lines = []
    for index, title in enumerate(clean):
        sec = int(index * spacing)
        minute, second = divmod(sec, 60)
        lines.append(f"{minute:02d}:{second:02d} {title}")
    return lines


def failure_file(exc: BaseException) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "HATA-V5-1.txt").write_text(
        f"{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )


def main() -> None:
    reset_dirs()

    topic = os.getenv("VIDEO_TOPIC", "").strip()
    if not topic:
        raise RuntimeError("Video konusu boş.")
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
    cloudflare_api_token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")
    if not cloudflare_account_id:
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID bulunamadı.")
    if not cloudflare_api_token:
        raise RuntimeError("CLOUDFLARE_API_TOKEN bulunamadı.")

    target_seconds = int(os.getenv("TARGET_SECONDS", "90"))
    scene_count = int(os.getenv("SCENE_COUNT", "12"))
    client = genai.Client(api_key=gemini_key)

    print("=" * 72)
    print("UYKU VE TARİH V5.1 — EDITOR BRAIN")
    print("Konu:", topic)
    print("=" * 72)

    payload, text_model = build_video_package(
        client, topic, target_seconds, scene_count
    )
    (OUTPUT / "video-paketi.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "senaryo.txt").write_text(
        payload["narration"], encoding="utf-8"
    )
    (OUTPUT / "baslik.txt").write_text(
        str(payload["video_title"]), encoding="utf-8"
    )

    raw_dir = WORK / "raw-images"
    frame_dir = WORK / "video-frames"
    raw_dir.mkdir()
    frame_dir.mkdir()

    frames: list[Path] = []
    image_manifest: list[dict[str, Any]] = []

    for scene in payload["scenes"]:
        scene_id = int(scene["scene_id"])
        raw = raw_dir / f"scene_{scene_id:02d}.jpg"
        info = generate_scene_image(client, topic, payload, scene, raw)
        frame = frame_dir / f"frame_{scene_id:02d}.jpg"
        clean_video_frame(raw, frame)
        frames.append(frame)
        image_manifest.append(info)

    make_storyboard(
        frames, payload["scenes"], OUTPUT / "storyboard-kontrol.jpg"
    )

    thumb_dir = WORK / "thumbnail-candidates"
    selected_thumb_background, thumb_info = generate_thumbnail_candidates(
        client, topic, payload, thumb_dir
    )
    make_thumbnail(
        selected_thumb_background,
        str(payload["thumbnail_text"]),
        OUTPUT / "kapak.jpg",
    )

    raw_audio = WORK / "narration-raw.wav"
    tts_model = synthesize_narration(
        client, payload["narration"], raw_audio
    )
    narration_audio = OUTPUT / "seslendirme.wav"
    normalize_audio(raw_audio, narration_audio)

    narration_duration = ffprobe_duration(narration_audio)
    visible_durations = allocate_scene_durations(
        payload["scenes"], narration_duration
    )
    transitions, transition_durations = scene_transition_plan(payload["scenes"])
    write_edit_timeline(
        payload["scenes"],
        visible_durations,
        transitions,
        transition_durations,
        OUTPUT / "edit-timeline.json",
    )

    ambient_track = WORK / "ambient-track.wav"
    build_ambient_track(
        payload["scenes"],
        visible_durations,
        transition_durations,
        ambient_track,
    )
    final_audio = OUTPUT / "ses-tasarim.wav"
    mix_narration_and_ambient(
        narration_audio, ambient_track, final_audio
    )

    video = OUTPUT / "pilot-video-v5-1-editorial-cut.mp4"
    actual_duration = render_video(
        frames,
        payload["scenes"],
        final_audio,
        video,
        visible_durations,
        transitions,
        transition_durations,
    )
    create_srt(
        payload["narration"], actual_duration, OUTPUT / "altyazi.srt"
    )

    chapters = chapter_text(payload.get("chapters", []), actual_duration)
    tags = ", ".join(
        str(tag).strip()
        for tag in payload.get("tags", [])
        if str(tag).strip()
    )
    description = [
        str(payload["description"]).strip(),
        "",
        "BÖLÜMLER",
        *chapters,
    ]
    if tags:
        description.extend(["", f"Etiketler: {tags}"])
    (OUTPUT / "youtube-aciklamasi.txt").write_text(
        "\n".join(description).strip() + "\n",
        encoding="utf-8",
    )

    manifest = {
        "topic": topic,
        "text_model": text_model,
        "tts_model": tts_model,
        "image_engine": "Cloudflare Workers AI",
        "image_model_default": os.getenv(
            "CLOUDFLARE_IMAGE_MODEL",
            "@cf/black-forest-labs/flux-2-klein-4b",
        ),
        "actual_duration_seconds": round(actual_duration, 2),
        "scene_count": len(frames),
        "editor_brain": {
            "zero_jitter": True,
            "semantic_scene_timing": True,
            "semantic_transitions": True,
            "procedural_ambience": True,
            "automatic_audio_ducking": True,
            "world_bible_continuity": True,
        },
        "images": image_manifest,
        "thumbnail": thumb_info,
        "final_video_contains_scene_numbers": False,
        "final_video_contains_scene_titles": False,
        "zero_jitter_mode": True,
        "camera_motion_inside_scenes": False,
        "transition_duration_seconds": 0.8,
        "transition_style": "restrained editorial xfade",
    }
    (OUTPUT / "uretim-raporu.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT / "ONCE-BUNU-OKU.txt").write_text(
        (
            "V5 Editorial Cut Engine yalnızca konu girdisiyle üretildi.\n\n"
            "Önce pilot-video-v5-1-editorial-cut.mp4 dosyasını izle.\n"
            "kapak.jpg dosyasını mobil boyutta kontrol et.\n"
            "storyboard-kontrol.jpg yalnızca kalite kontrol dosyasıdır; "
            "üzerindeki açıklamalar final videoda bulunmaz.\n"
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("V4 EDITOR BRAIN TAMAMLANDI")
    print("Video:", video)
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failure_file(exc)
        raise
