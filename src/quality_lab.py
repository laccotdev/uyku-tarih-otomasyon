from __future__ import annotations

import html
import io
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output-v2"
WORK = ROOT / "work-v2"
WIDTH = 1920
HEIGHT = 1080
FPS = 25
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MET_API = "https://collectionapi.metmuseum.org/public/collection/v1"
USER_AGENT = "UykuTarihProfessionalLab/2.0 (educational-history-video)"
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
VOICE_OPTIONS = [
    ("Charon", "Bilgilendirici, tok ve kontrollü"),
    ("Schedar", "Dengeli, sakin ve nötr"),
    ("Sulafat", "Sıcak, yumuşak ve gece anlatısına uygun"),
]


@dataclass
class MediaItem:
    title: str
    image_url: str
    source_page: str
    author: str
    license_name: str
    license_url: str
    description: str
    source: str
    width: int = 0
    height: int = 0
    score: float = 0.0


def clean_html(value: str | None) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def slug_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def run(command: list[str]) -> None:
    print("$", " ".join(str(x) for x in command))
    subprocess.run(command, check=True)


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


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for item in candidates:
        if Path(item).exists():
            return ImageFont.truetype(item, size=size)
    return ImageFont.load_default()


def json_from_response(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Gemini yanıtında JSON bulunamadı.")
    return json.loads(text[start:end + 1])


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip().removeprefix("models/")
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def available_generate_models(client: genai.Client) -> set[str]:
    """Return models exposed to this API key that support generateContent."""
    available: set[str] = set()
    try:
        for model in client.models.list():
            actions = set(getattr(model, "supported_actions", None) or [])
            name = str(getattr(model, "name", "")).removeprefix("models/")
            if name and (not actions or "generateContent" in actions):
                available.add(name)
    except Exception as exc:
        # Model discovery is helpful but must never block the render.
        print("Model listesi alınamadı; sabit yedek zinciri kullanılacak:", exc)
    return available


def is_retryable_service_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "429", "500", "502", "503", "504",
        "resource_exhausted", "unavailable", "high demand",
        "temporarily", "timeout", "timed out", "deadline exceeded",
        "connection reset", "connection aborted", "service unavailable",
    )
    return any(marker in message for marker in markers)


def ordered_candidates(
    available: set[str],
    preferred: list[str],
) -> list[str]:
    candidates = _unique(preferred)
    if not available:
        return candidates
    filtered = [model for model in candidates if model in available]
    # Keep fixed candidates as a last resort because list() can lag aliases.
    return filtered or candidates


def build_editorial_package(
    client: genai.Client,
    topic: str,
    preview_seconds: int,
    visual_count: int,
) -> dict[str, Any]:
    target_words = max(145, min(230, int(preview_seconds * 2.35)))
    prompt = f"""
Yalnızca geçerli JSON üret. Markdown ekleme.

Sen; sakin tarih belgeseli, uyku anlatısı ve görsel araştırma konusunda çalışan
kıdemli bir Türkçe editörsün.

KONU: {topic}
SES ÖNİZLEMESİ: yaklaşık {preview_seconds} saniye
HEDEF KELİME: {target_words}
SAHNE SAYISI: {visual_count}

JSON şeması:
{{
  "video_title": "Merak uyandırıcı ama abartısız YouTube başlığı",
  "thumbnail_text": "En fazla üç kelime",
  "sample_narration": "Doğal, sakin ve yayın kalitesinde Türkçe anlatım",
  "visual_direction": "Tek paragraf görsel sanat yönetimi",
  "scenes": [
    {{
      "scene_title": "Kısa sahne adı",
      "purpose": "Bu görsel anlatıda neyi destekliyor",
      "commons_query": "Wikimedia Commons için İngilizce, somut ve kesin sorgu",
      "backup_query": "Daha geniş İngilizce alternatif sorgu",
      "required_terms": ["İngilizce", "anahtar", "kelimeler"],
      "avoid_terms": ["logo", "modern reconstruction gibi istenmeyen kelimeler"]
    }}
  ]
}}

Yazım kuralları:
- İlk cümle dinleyiciyi doğrudan tarihî mekâna taşısın.
- Standart Türkiye Türkçesi kullan.
- Reklam, fragman ve yapay dramatizasyon tonu kullanma.
- Kısa ve doğal cümleler yaz; art arda aynı kalıbı tekrarlama.
- Kesin bilinmeyen bilgileri kesinmiş gibi sunma.
- Sakin ama merak uyandıran bir gece belgeseli hissi ver.
- Sayıları ve tarihleri seslendirmeye uygun biçimde yazıyla yaz.
- Metnin sonunda yumuşak bir kapanış olsun.
- Her sahne birbirinden farklı bir görsel ihtiyacını karşılasın.
- Sorgularda yer, eser, kapı, tablet, harita, kazı alanı gibi somut özel adlar kullan.
- Genel "ancient history" sorgularından kaçın.
- Görsel üretme; yalnızca arşivde bulunabilecek gerçek mekân, eser, harita,
  gravür, kazı veya müze objeleri planla.
"""

    available = available_generate_models(client)
    configured = os.getenv("TEXT_MODEL", "").strip()
    models = ordered_candidates(
        available,
        [
            # This task is structured JSON; Lite models are faster and usually
            # less congested than the flagship model.
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            configured,
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
        ],
    )
    print("Metin modeli yedek zinciri:", " -> ".join(models))

    last_error: Exception | None = None
    delays = (3, 12, 30)
    for model in models:
        for attempt, delay in enumerate(delays, start=1):
            try:
                print(f"Editoryal paket: model={model}, deneme={attempt}/{len(delays)}")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.55,
                        max_output_tokens=8192,
                    ),
                )
                payload = json_from_response(response.text or "")
                scenes = payload.get("scenes")
                if not isinstance(scenes, list) or len(scenes) < visual_count:
                    raise ValueError("Yeterli sahne üretilmedi.")
                payload["scenes"] = scenes[:visual_count]
                narration = slug_text(str(payload.get("sample_narration", "")))
                if len(narration.split()) < 100:
                    raise ValueError("Ses metni çok kısa.")
                payload["sample_narration"] = narration
                payload["text_model_used"] = model
                print("Editoryal paket başarılı. Kullanılan model:", model)
                return payload
            except Exception as exc:
                last_error = exc
                message = str(exc)
                print(f"Model başarısız: {model}: {message}")
                # Missing/unsupported models should immediately move to fallback.
                if "404" in message or "not found" in message.lower() or "no longer available" in message.lower():
                    break
                if not is_retryable_service_error(exc):
                    # A malformed JSON response can still succeed on another attempt.
                    if attempt == len(delays):
                        break
                if attempt < len(delays):
                    print(f"Geçici hata; {delay} saniye sonra yeniden denenecek.")
                    time.sleep(delay)
        print("Sıradaki metin modeline geçiliyor.")

    raise RuntimeError(f"Editoryal paket üretilemedi. Son hata: {last_error}")

def license_allowed(name: str) -> bool:
    lower = name.lower()
    if not lower:
        return False
    if any(term in lower for term in ("noncommercial", "no derivatives", "fair use")):
        return False
    return any(
        term in lower
        for term in (
            "public domain", "cc0", "cc by", "cc-by",
            "creative commons attribution",
        )
    )


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key, {})
    if isinstance(value, dict):
        return clean_html(str(value.get("value", "")))
    return clean_html(str(value))


def tokenize(value: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zÀ-ž0-9]+", value)
        if len(token) >= 3
    }


def score_item(
    item: MediaItem,
    query: str,
    required_terms: list[str],
    avoid_terms: list[str],
) -> float:
    haystack = f"{item.title} {item.description}".lower()
    query_tokens = tokenize(query)
    required_tokens = tokenize(" ".join(required_terms))
    overlap = len((query_tokens | required_tokens) & tokenize(haystack))
    score = overlap * 7.0

    for term in required_terms:
        if term.lower() in haystack:
            score += 8.0
    for term in avoid_terms:
        if term.lower() in haystack:
            score -= 30.0

    if item.width >= 1600:
        score += 8
    elif item.width >= 1000:
        score += 4
    if item.height >= 800:
        score += 4

    ratio = item.width / item.height if item.height else 1.0
    if 1.25 <= ratio <= 2.1:
        score += 5
    if item.source == "Wikimedia Commons":
        score += 2
    if "public domain" in item.license_name.lower() or "cc0" in item.license_name.lower():
        score += 4
    return score


def commons_search(query: str, limit: int = 20) -> list[MediaItem]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|size|extmetadata",
        "iiurlwidth": 1920,
        "format": "json",
        "formatversion": 2,
    }
    response = requests.get(
        COMMONS_API,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=40,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    items: list[MediaItem] = []
    for page in pages:
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        if info.get("mime") not in ALLOWED_MIMES:
            continue
        meta = info.get("extmetadata") or {}
        license_name = metadata_value(meta, "LicenseShortName")
        if not license_allowed(license_name):
            continue
        title = str(page.get("title", "")).removeprefix("File:").strip()
        url = str(info.get("thumburl") or info.get("url") or "").strip()
        if not title or not url:
            continue
        items.append(
            MediaItem(
                title=title,
                image_url=url,
                source_page=(
                    "https://commons.wikimedia.org/wiki/"
                    + quote(str(page.get("title", "")).replace(" ", "_"))
                ),
                author=metadata_value(meta, "Artist") or "Belirtilmemiş",
                license_name=license_name,
                license_url=metadata_value(meta, "LicenseUrl"),
                description=(
                    metadata_value(meta, "ImageDescription")
                    or metadata_value(meta, "ObjectName")
                ),
                source="Wikimedia Commons",
                width=int(info.get("thumbwidth") or info.get("width") or 0),
                height=int(info.get("thumbheight") or info.get("height") or 0),
            )
        )
    return items


def met_search(query: str, limit: int = 12) -> list[MediaItem]:
    try:
        search = requests.get(
            f"{MET_API}/search",
            params={"hasImages": "true", "q": query},
            headers={"User-Agent": USER_AGENT},
            timeout=30,
        )
        search.raise_for_status()
        ids = (search.json().get("objectIDs") or [])[:limit]
    except Exception:
        return []

    items: list[MediaItem] = []
    for object_id in ids:
        try:
            response = requests.get(
                f"{MET_API}/objects/{object_id}",
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if not data.get("isPublicDomain") or not data.get("primaryImageSmall"):
                continue
            title = str(data.get("title") or "The Met eseri")
            description = " ".join(
                str(data.get(key) or "")
                for key in (
                    "objectName", "culture", "period", "dynasty",
                    "reign", "medium", "geographyType",
                    "city", "country", "region",
                )
            )
            items.append(
                MediaItem(
                    title=title,
                    image_url=str(data["primaryImageSmall"]),
                    source_page=str(data.get("objectURL") or ""),
                    author=str(data.get("artistDisplayName") or "The Metropolitan Museum of Art"),
                    license_name="Public Domain / Open Access",
                    license_url="https://www.metmuseum.org/about-the-met/policies-and-documents/open-access",
                    description=description,
                    source="The Metropolitan Museum of Art",
                    width=1200,
                    height=1200,
                )
            )
        except Exception:
            continue
    return items


def select_media_for_scene(
    scene: dict[str, Any],
    used_urls: set[str],
) -> MediaItem | None:
    primary = str(scene.get("commons_query", "")).strip()
    backup = str(scene.get("backup_query", "")).strip()
    required = [str(x) for x in scene.get("required_terms", [])]
    avoid = [str(x) for x in scene.get("avoid_terms", [])]
    candidates: list[MediaItem] = []

    for query in (primary, backup):
        if not query:
            continue
        try:
            candidates.extend(commons_search(query))
        except Exception as exc:
            print("Commons araması başarısız:", exc)
        if len(candidates) < 4:
            candidates.extend(met_search(query, limit=8))

    unique: dict[str, MediaItem] = {}
    for item in candidates:
        unique.setdefault(item.image_url, item)

    ranked = []
    for item in unique.values():
        item.score = score_item(item, primary, required, avoid)
        if item.image_url in used_urls:
            item.score -= 35
        ranked.append(item)

    ranked.sort(key=lambda x: x.score, reverse=True)
    return ranked[0] if ranked else None


def download_image(item: MediaItem, target: Path) -> bool:
    try:
        response = requests.get(
            item.image_url,
            headers={"User-Agent": USER_AGENT},
            timeout=60,
        )
        response.raise_for_status()
        target.write_bytes(response.content)
        with Image.open(target) as image:
            image.verify()
        return True
    except Exception as exc:
        target.unlink(missing_ok=True)
        print("Görsel indirilemedi:", exc)
        return False


def make_frame(source: Path, target: Path, scene_no: int) -> None:
    with Image.open(source) as raw:
        original = ImageOps.exif_transpose(raw).convert("RGB")

    bg = ImageOps.fit(original, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(34))
    bg = ImageEnhance.Brightness(bg).enhance(0.42)
    bg = ImageEnhance.Color(bg).enhance(0.65)

    fg = ImageOps.contain(original, (1760, 960), method=Image.Resampling.LANCZOS)
    fg = ImageEnhance.Contrast(fg).enhance(1.05)
    fg = ImageEnhance.Color(fg).enhance(0.78)

    canvas = bg.copy()
    x = (WIDTH - fg.width) // 2
    y = (HEIGHT - fg.height) // 2
    canvas.paste(fg, (x, y))

    shade = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shade)
    draw.rectangle((0, 0, WIDTH, 90), fill=(10, 9, 8, 45))
    draw.rectangle((0, HEIGHT - 160, WIDTH, HEIGHT), fill=(10, 9, 8, 70))
    draw.ellipse((-260, -180, 900, 980), fill=(79, 58, 33, 18))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shade).convert("RGB")

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (70, HEIGHT - 95),
        f"{scene_no:02d}",
        font=font(28, bold=True),
        fill=(213, 197, 165),
    )
    canvas.save(target, "JPEG", quality=93, optimize=True)


def make_storyboard(
    frames: list[Path],
    scenes: list[dict[str, Any]],
    target: Path,
) -> None:
    cols = 3
    cell_w, cell_h = 600, 390
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (18, 17, 15))
    draw = ImageDraw.Draw(sheet)
    for index, frame_path in enumerate(frames):
        row, col = divmod(index, cols)
        x, y = col * cell_w, row * cell_h
        with Image.open(frame_path) as raw:
            img = ImageOps.fit(raw.convert("RGB"), (cell_w, 338), Image.Resampling.LANCZOS)
        sheet.paste(img, (x, y))
        draw.rectangle((x, y + 338, x + cell_w, y + cell_h), fill=(24, 22, 19))
        title = str(scenes[index].get("scene_title", f"Sahne {index + 1}"))
        caption = textwrap.shorten(title, width=48, placeholder="…")
        draw.text(
            (x + 18, y + 350),
            f"{index + 1:02d}  {caption}",
            font=font(19, bold=True),
            fill=(229, 221, 204),
        )
    sheet.save(target, "JPEG", quality=92, optimize=True)


def make_thumbnail(frame: Path, text: str, target: Path) -> None:
    with Image.open(frame) as raw:
        image = ImageOps.fit(raw.convert("RGB"), (1280, 720), Image.Resampling.LANCZOS)
    image = ImageEnhance.Brightness(image).enhance(0.76)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle((0, 0, 760, 720), fill=(8, 8, 7, 150))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)
    draw = ImageDraw.Draw(image)
    clean = re.sub(r"\s+", " ", text).strip().upper()
    wrapped = textwrap.fill(clean, width=13)
    draw.multiline_text(
        (72, 190),
        wrapped,
        font=font(86, bold=True),
        fill=(241, 232, 211),
        spacing=7,
        stroke_width=2,
        stroke_fill=(15, 13, 10),
    )
    draw.text(
        (77, 600),
        "UYKU İÇİN TARİH",
        font=font(28, bold=True),
        fill=(193, 173, 138),
    )
    image.convert("RGB").save(target, "JPEG", quality=94, optimize=True)


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm)


def tts_prompt(narration: str) -> str:
    return f"""
Synthesize speech only. Never read the directions, labels, or section names aloud.

AUDIO PROFILE:
A mature Turkish documentary narrator with a calm, grounded presence.
Natural standard Turkey Turkish. Close-microphone studio sound.

SCENE:
A quiet late-night historical documentary intended for relaxed listening.

DIRECTOR'S NOTES:
Speak slowly but naturally. Keep a stable low-energy delivery.
Use gentle sentence endings and short natural pauses.
Do not sound like an advertisement, trailer, news bulletin, or theatre actor.
Do not exaggerate emotion. Keep pronunciation clear and intimate.
Do not whisper. Do not add music, sound effects, commentary, or extra words.

THE SPOKEN TRANSCRIPT STARTS AFTER THIS LINE:
{narration}
THE SPOKEN TRANSCRIPT ENDS HERE.
""".strip()


def synthesize_voice(
    client: genai.Client,
    narration: str,
    voice_name: str,
    target: Path,
) -> str:
    available = available_generate_models(client)
    configured = os.getenv("TTS_MODEL", "").strip()
    models = ordered_candidates(
        available,
        [
            configured,
            "gemini-3.1-flash-tts-preview",
            "gemini-2.5-flash-preview-tts",
        ],
    )
    print(f"{voice_name} TTS yedek zinciri:", " -> ".join(models))

    last_error: Exception | None = None
    delays = (8, 25, 55)
    for model in models:
        for attempt, delay in enumerate(delays, start=1):
            try:
                print(
                    f"Ses üretiliyor: voice={voice_name}, model={model}, "
                    f"deneme={attempt}/{len(delays)}"
                )
                response = client.models.generate_content(
                    model=model,
                    contents=tts_prompt(narration),
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
                candidates = response.candidates or []
                if not candidates or not candidates[0].content.parts:
                    raise ValueError("TTS yanıtında ses parçası bulunamadı.")
                inline = candidates[0].content.parts[0].inline_data
                data = inline.data if inline else None
                if not data or len(data) < 48000:
                    raise ValueError("Ses verisi boş veya çok kısa.")
                write_wav(target, data)
                if ffprobe_duration(target) < 20:
                    raise ValueError("Üretilen ses beklenenden kısa.")
                print(f"Ses başarılı: {voice_name}, model={model}")
                return model
            except Exception as exc:
                last_error = exc
                target.unlink(missing_ok=True)
                message = str(exc)
                print(f"TTS başarısız: {voice_name}, model={model}: {message}")
                if "404" in message or "not found" in message.lower() or "no longer available" in message.lower():
                    break
                if attempt < len(delays):
                    print(f"Geçici TTS hatası; {delay} saniye beklenecek.")
                    time.sleep(delay)
        print(f"{voice_name} için sıradaki TTS modeline geçiliyor.")

    raise RuntimeError(f"{voice_name} sesi üretilemedi. Son hata: {last_error}")

def normalize_audio(source: Path, target: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-af",
            "highpass=f=55,lowpass=f=14500,"
            "acompressor=threshold=-20dB:ratio=2.2:attack=20:release=180,"
            "loudnorm=I=-17:TP=-2:LRA=8",
            "-ar", "48000",
            "-c:a", "pcm_s16le",
            str(target),
        ]
    )


def create_preview(
    frames: list[Path],
    audio: Path,
    target: Path,
    work_dir: Path,
) -> None:
    duration = ffprobe_duration(audio)
    scene_duration = duration / len(frames)
    clip_dir = work_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for index, frame in enumerate(frames):
        clip = clip_dir / f"clip_{index:03d}.mp4"
        frame_count = max(1, math.ceil(scene_duration * FPS))
        fade_out_start = max(0.0, scene_duration - 0.75)
        zoom = "0.00016" if index % 2 == 0 else "0.00011"
        vf = (
            f"zoompan=z='min(zoom+{zoom},1.075)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frame_count}:s={WIDTH}x{HEIGHT}:fps={FPS},"
            f"fade=t=in:st=0:d=0.65,"
            f"fade=t=out:st={fade_out_start:.3f}:d=0.75,"
            "vignette=PI/5,"
            "noise=alls=2:allf=t,"
            "format=yuv420p"
        )
        run(
            [
                "ffmpeg", "-y", "-loop", "1", "-i", str(frame),
                "-vf", vf,
                "-frames:v", str(frame_count),
                "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "21", "-pix_fmt", "yuv420p",
                str(clip),
            ]
        )
        clips.append(clip)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{clip.resolve().as_posix()}'" for clip in clips),
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart",
            str(target),
        ]
    )


def save_sources(items: list[MediaItem], scenes: list[dict[str, Any]], target: Path) -> None:
    lines = ["GÖRSEL KAYNAKLARI VE LİSANSLAR", ""]
    for index, item in enumerate(items, start=1):
        scene_title = scenes[index - 1].get("scene_title", f"Sahne {index}")
        lines.extend(
            [
                f"{index}. {scene_title}",
                f"Eser: {item.title}",
                f"Kaynak: {item.source}",
                f"Üretici: {item.author}",
                f"Lisans: {item.license_name}",
                f"Sayfa: {item.source_page}",
                f"Lisans bağlantısı: {item.license_url or 'Belirtilmemiş'}",
                "",
            ]
        )
    target.write_text("\n".join(lines), encoding="utf-8")


def create_report(
    topic: str,
    payload: dict[str, Any],
    items: list[MediaItem],
    target: Path,
) -> None:
    voice_lines = [
        f"- {voice}: {description}"
        for voice, description in VOICE_OPTIONS
    ]
    report = f"""
PROFESYONEL KALİTE TESTİ

Konu:
{topic}

Amaç:
Bu çalışma tam video değildir. Yayınlanabilir uzun videoya geçmeden önce ses
karakterini ve görsel sanat yönetimini ayrı ayrı onaylamak için hazırlanmıştır.

SES DOSYALARI:
{chr(10).join(voice_lines)}

Önerilen ilk dinleme sırası:
1. Schedar
2. Sulafat
3. Charon

GÖRSEL DOSYALAR:
- storyboard.jpg: seçilen bütün sahnelerin tek sayfalık kontrolü
- onizleme-schedar.mp4: varsayılan ses ve sahnelerle kısa kurgu
- kapak-v2.jpg: yeni kapak yönü

Görsel yön:
{payload.get("visual_direction", "")}

Seçilen gerçek/lisanslı görsel sayısı:
{len(items)}

Sonraki kalite kapısı:
Bir ses seçildikten sonra aynı ses; cümle bölme, telaffuz sözlüğü ve bölüm bazlı
üretimle sabitlenir. Ardından iki ila beş dakikalık gerçek pilot hazırlanır.
""".strip()
    target.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    if WORK.exists():
        shutil.rmtree(WORK)
    OUTPUT.mkdir(parents=True)
    WORK.mkdir(parents=True)

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")

    topic = os.getenv("VIDEO_TOPIC", "MÖ 1200'de Hattuşa'nın son gecesi").strip()
    preview_seconds = max(45, min(100, int(os.getenv("PREVIEW_SECONDS", "75"))))
    visual_count = max(8, min(16, int(os.getenv("VISUAL_COUNT", "12"))))
    client = genai.Client(api_key=api_key)

    print("=" * 72)
    print("UYKU VE TARİH — PROFESYONEL KALİTE LABORATUVARI")
    print("Konu:", topic)
    print("=" * 72)

    payload = build_editorial_package(client, topic, preview_seconds, visual_count)
    (OUTPUT / "editoryal-paket.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    narration = payload["sample_narration"]
    (OUTPUT / "ses-test-metni.txt").write_text(narration, encoding="utf-8")
    (OUTPUT / "video-basligi.txt").write_text(
        str(payload.get("video_title", topic)),
        encoding="utf-8",
    )

    source_dir = WORK / "source-images"
    frame_dir = WORK / "frames"
    source_dir.mkdir()
    frame_dir.mkdir()

    used_urls: set[str] = set()
    selected_items: list[MediaItem] = []
    frames: list[Path] = []
    successful_scenes: list[dict[str, Any]] = []

    for index, scene in enumerate(payload["scenes"], start=1):
        print(f"Sahne seçiliyor {index}/{visual_count}: {scene.get('scene_title')}")
        item = select_media_for_scene(scene, used_urls)
        if item is None:
            print("Sahne için uygun lisanslı görsel bulunamadı; sahne atlandı.")
            continue
        suffix = Path(item.image_url.split("?", 1)[0]).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            suffix = ".jpg"
        source_path = source_dir / f"source_{index:02d}{suffix}"
        if not download_image(item, source_path):
            continue
        frame_path = frame_dir / f"frame_{index:02d}.jpg"
        make_frame(source_path, frame_path, index)
        used_urls.add(item.image_url)
        selected_items.append(item)
        frames.append(frame_path)
        successful_scenes.append(scene)

    if len(frames) < 6:
        raise RuntimeError(
            f"Yeterli kaliteli görsel bulunamadı. Yalnızca {len(frames)} sahne seçildi."
        )

    make_storyboard(frames, successful_scenes, OUTPUT / "storyboard.jpg")
    make_thumbnail(
        frames[0],
        str(payload.get("thumbnail_text", "SON GECE")),
        OUTPUT / "kapak-v2.jpg",
    )
    save_sources(
        selected_items,
        successful_scenes,
        OUTPUT / "gorsel-kaynaklari-v2.txt",
    )

    voice_dir = WORK / "voices"
    voice_dir.mkdir()
    normalized_voices: dict[str, Path] = {}
    voice_manifest = []

    for voice_name, description in VOICE_OPTIONS:
        raw = voice_dir / f"{voice_name.lower()}-raw.wav"
        final = OUTPUT / f"ses-{voice_name.lower()}.wav"
        try:
            model_used = synthesize_voice(client, narration, voice_name, raw)
            normalize_audio(raw, final)
            normalized_voices[voice_name] = final
            voice_manifest.append(
                {
                    "voice": voice_name,
                    "description": description,
                    "status": "success",
                    "tts_model_used": model_used,
                    "duration_seconds": round(ffprobe_duration(final), 2),
                }
            )
        except Exception as exc:
            print(f"{voice_name} sesi atlandı: {exc}")
            voice_manifest.append(
                {
                    "voice": voice_name,
                    "description": description,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    if not normalized_voices:
        raise RuntimeError("Hiçbir TTS sesi üretilemedi.")

    (OUTPUT / "ses-secenekleri.json").write_text(
        json.dumps(voice_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_voice = "Schedar" if "Schedar" in normalized_voices else next(iter(normalized_voices))
    preview_audio = normalized_voices[preview_voice]
    print("Önizleme sesi:", preview_voice)
    create_preview(
        frames,
        preview_audio,
        OUTPUT / "onizleme-schedar.mp4",
        WORK,
    )
    create_report(
        topic,
        payload,
        selected_items,
        OUTPUT / "ONCE-BUNU-OKU.txt",
    )

    print("=" * 72)
    print("KALİTE TESTİ TAMAMLANDI")
    print("Çıktı klasörü:", OUTPUT)
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "HATA-V2.txt").write_text(
            f"{type(exc).__name__}: {exc}\n",
            encoding="utf-8",
        )
        raise
