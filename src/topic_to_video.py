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
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output-v3"
WORK = ROOT / "work-v3"
WIDTH = 1920
HEIGHT = 1080
FPS = 25
USER_AGENT = "UykuTarihTopicToVideo/3.3"
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
clothing and tools. Quiet atmospheric lighting, earthy natural colors,
subtle filmic contrast, realistic skin and anatomy, immersive 16:9
composition, no fantasy spectacle. The visual must look like a frame from one
cohesive historical film. No text, no letters, no numbers, no captions,
no logos, no watermarks, no borders, no collage and no museum display.
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


def _package_issues(
    payload: dict[str, Any],
    scene_count: int,
    minimum_words: int,
) -> list[str]:
    issues: list[str] = []
    required = (
        "video_title", "thumbnail_text", "description", "narration",
        "visual_identity", "thumbnail_prompt", "scenes",
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
        for key in ("narration_idea", "visual_goal", "image_prompt"):
            if not str(scene.get(key, "")).strip():
                issues.append(f"Sahne {index} için {key} eksik.")
    return issues


def _normalize_package(payload: dict[str, Any], scene_count: int) -> None:
    payload["narration"] = re.sub(
        r"\s+", " ", str(payload.get("narration", ""))
    ).strip()
    payload["scenes"] = list(payload.get("scenes", []))[:scene_count]

    for index, scene in enumerate(payload["scenes"], start=1):
        scene["scene_id"] = index
        motion = str(scene.get("motion", "slow_zoom_in"))
        if motion not in {
            "slow_zoom_in", "slow_zoom_out", "pan_left", "pan_right"
        }:
            scene["motion"] = "slow_zoom_in"
        if str(scene.get("importance", "normal")) not in {"high", "normal"}:
            scene["importance"] = "normal"


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
- Her sahne narration sırasındaki somut fikri doğrudan görselleştirsin.
- Her image_prompt İngilizce, ayrıntılı, 16:9 sinematik tarih karesi tarif etsin.
- Müze objesi, beyaz fon, kolaj, yazı, sayı, logo, modern veya fantastik unsur olmasın.
- video_title, thumbnail_text, description, visual_identity,
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
  "thumbnail_prompt": "Yazısız 16:9 kapak arka planı için ayrıntılı İngilizce prompt. Ana özne sağ tarafta, sol taraf koyu ve boş.",
  "thumbnail_negative_prompt": "Kapakta kesinlikle olmaması gereken unsurlar",
  "scenes": [
    {{
      "scene_id": 1,
      "narration_idea": "Bu sahne sırasında anlatılan ana fikir",
      "visual_goal": "Görüntünün açıkça göstermesi gereken olay, mekân veya durum",
      "image_prompt": "Tek bir sinematik kare üretmek için ayrıntılı İngilizce prompt",
      "negative_prompt": "Bu sahneye özgü kaçınılacak unsurlar",
      "motion": "slow_zoom_in | slow_zoom_out | pan_left | pan_right",
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

def combined_prompt(payload: dict[str, Any], scene: dict[str, Any]) -> str:
    return (
        f"{scene['image_prompt'].strip()}\n\n"
        f"VIDEO VISUAL IDENTITY:\n{payload['visual_identity'].strip()}\n\n"
        f"MASTER STYLE:\n{STYLE_BIBLE}"
    )


def combined_negative(scene: dict[str, Any]) -> str:
    extra = str(scene.get("negative_prompt", "")).strip()
    return f"{GLOBAL_NEGATIVE}, {extra}" if extra else GLOBAL_NEGATIVE


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
    """Return form data for multipart models or JSON for classic models."""
    final_prompt = (
        f"{prompt}\n\n"
        f"STRICT EXCLUSIONS: {negative}\n"
        "Landscape 16:9 frame, clean cinematic composition."
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
        body = {
            "prompt": final_prompt,
            "seed": seed,
            "steps": 8,
        }
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
) -> tuple[bool, str]:
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
        return passed, str(payload.get("reason", ""))
    except Exception as exc:
        # Görsel değerlendirme servisi geçici olarak çalışmazsa üretimi durdurma.
        print("Görsel kalite değerlendirmesi atlandı:", exc)
        return True, "Otomatik inceleme geçici olarak atlandı."


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

    for attempt, model in enumerate(models[:max_attempts], start=1):
        seed = deterministic_seed(topic, int(scene["scene_id"]), attempt)
        try:
            print(
                f"Sahne görseli: {scene['scene_id']}, model={model}, "
                f"deneme={attempt}/{max_attempts}"
            )
            cloudflare_image_request(
                combined_prompt(payload, scene),
                combined_negative(scene),
                seed,
                target,
                model,
            )
            passed, reason = image_review(client, scene, target)
            print(f"Görsel değerlendirmesi: pass={passed}; {reason}")
            if passed:
                return {
                    "scene_id": scene["scene_id"],
                    "model": model,
                    "seed": seed,
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
            time.sleep(min(12, 3 * attempt))

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
        image, (WIDTH, HEIGHT),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    frame = ImageEnhance.Color(frame).enhance(0.90)
    frame = ImageEnhance.Contrast(frame).enhance(1.06)
    frame = ImageEnhance.Brightness(frame).enhance(0.95)

    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(13, 10, 7, 16))
    frame = Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")
    frame.save(target, "JPEG", quality=94, optimize=True)


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


def scene_filter(motion: str, frames: int, seconds: float) -> str:
    fade_out = max(0.0, seconds - 0.6)
    if motion == "slow_zoom_out":
        zoom = "if(eq(on,1),1.075,max(1.0,zoom-0.00020))"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_left":
        zoom = "1.045"
        x = "(iw-iw/zoom)*(1-on/{frames})".format(frames=max(1, frames))
        y = "ih/2-(ih/zoom/2)"
    elif motion == "pan_right":
        zoom = "1.045"
        x = "(iw-iw/zoom)*(on/{frames})".format(frames=max(1, frames))
        y = "ih/2-(ih/zoom/2)"
    else:
        zoom = "min(zoom+0.00018,1.075)"
        x = "iw/2-(iw/zoom/2)"
        y = "ih/2-(ih/zoom/2)"

    return (
        f"zoompan=z='{zoom}':x='{x}':y='{y}':"
        f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"fade=t=in:st=0:d=0.6,"
        f"fade=t=out:st={fade_out:.3f}:d=0.6,"
        "vignette=PI/5.5,"
        "noise=alls=1.4:allf=t,"
        "format=yuv420p"
    )


def render_video(
    frames: list[Path],
    scenes: list[dict[str, Any]],
    audio: Path,
    target: Path,
) -> float:
    duration = ffprobe_duration(audio)
    scene_seconds = duration / len(frames)
    clips_dir = WORK / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for index, (frame, scene) in enumerate(zip(frames, scenes), start=1):
        frames_count = max(1, math.ceil(scene_seconds * FPS))
        clip = clips_dir / f"clip_{index:03d}.mp4"
        motion = str(scene.get("motion", "slow_zoom_in"))
        run(
            [
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(frame),
                "-vf", scene_filter(motion, frames_count, scene_seconds),
                "-frames:v", str(frames_count),
                "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "21", "-pix_fmt", "yuv420p",
                str(clip),
            ]
        )
        clips.append(clip)

    concat = WORK / "clips.txt"
    concat.write_text(
        "\n".join(f"file '{clip.resolve().as_posix()}'" for clip in clips),
        encoding="utf-8",
    )

    run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(target),
        ]
    )
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
    (OUTPUT / "HATA-V3-3.txt").write_text(
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
    print("UYKU VE TARİH V3.3 — KONUDAN VİDEOYA")
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

    thumb_raw = WORK / "thumbnail-background.jpg"
    thumb_info = generate_thumbnail_background(topic, payload, thumb_raw)
    make_thumbnail(
        thumb_raw,
        str(payload["thumbnail_text"]),
        OUTPUT / "kapak.jpg",
    )

    raw_audio = WORK / "narration-raw.wav"
    tts_model = synthesize_narration(
        client, payload["narration"], raw_audio
    )
    audio = OUTPUT / "seslendirme.wav"
    normalize_audio(raw_audio, audio)

    video = OUTPUT / "pilot-video-v3-3.mp4"
    actual_duration = render_video(
        frames, payload["scenes"], audio, video
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
        "images": image_manifest,
        "thumbnail": thumb_info,
        "final_video_contains_scene_numbers": False,
        "final_video_contains_scene_titles": False,
    }
    (OUTPUT / "uretim-raporu.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT / "ONCE-BUNU-OKU.txt").write_text(
        (
            "V3 yalnızca konu girdisiyle üretildi.\n\n"
            "Önce pilot-video-v3-3.mp4 dosyasını izle.\n"
            "kapak.jpg dosyasını mobil boyutta kontrol et.\n"
            "storyboard-kontrol.jpg yalnızca kalite kontrol dosyasıdır; "
            "üzerindeki açıklamalar final videoda bulunmaz.\n"
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("V3 ÜRETİM TAMAMLANDI")
    print("Video:", video)
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failure_file(exc)
        raise
