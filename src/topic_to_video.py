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
USER_AGENT = "UykuTarihTopicToVideo/8.2"
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
    "gemini-2.5-flash-preview-tts",
    "gemini-3.1-flash-tts-preview",
]

# Cloudflare FLUX.1 accepts at most 2048 characters in `prompt`.
# Keep a safety margin because fallback models receive exclusions in the same field.
MAX_IMAGE_PROMPT_CHARS = 1320
MAX_NEGATIVE_PROMPT_CHARS = 360
INTRO_VISIBLE_SECONDS = 8.0
INTRO_TRANSITION_SECONDS = 0.70
SCENE_PAUSE_SECONDS = 0.18
DEFAULT_TARGET_SECONDS = 300
DEFAULT_SCENE_COUNT = 12
CHAPTER_COUNT = 3
MIN_FINAL_VIDEO_SECONDS = 260
MAX_FINAL_VIDEO_SECONDS = 330

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
    try:
        subprocess.run(
            command,
            check=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Harici işlem 15 dakikalık güvenlik sınırını aştı ve durduruldu."
        ) from exc


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
                if retryable(exc) and attempt < 2:
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
            "chapter_index", "chapter_title", "beat_type",
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

        default_chapter = min(
            CHAPTER_COUNT,
            1 + ((index - 1) * CHAPTER_COUNT // max(1, scene_count)),
        )
        try:
            chapter_index = int(scene.get("chapter_index", default_chapter))
        except (TypeError, ValueError):
            chapter_index = default_chapter
        scene["chapter_index"] = max(1, min(CHAPTER_COUNT, chapter_index))

        chapter_title = re.sub(
            r"\s+", " ", str(scene.get("chapter_title", "")).strip()
        )
        if not chapter_title:
            chapter_title = [
                "Geceye Giriş",
                "Yaşayan Dünya",
                "Kırılma Anı",
                "Sessiz Miras",
            ][scene["chapter_index"] - 1]
        scene["chapter_title"] = chapter_title

        allowed_beats = {
            "hook", "orientation", "setting",
            "routine", "craft", "community",
            "evidence", "tension", "turning_point",
            "aftermath", "legacy", "reflection",
        }
        beat_type = str(scene.get("beat_type", "")).strip().lower()
        if beat_type not in allowed_beats:
            beat_order = [
                "hook", "orientation", "setting",
                "routine", "craft", "community",
                "evidence", "tension", "turning_point",
                "aftermath", "legacy", "reflection",
            ]
            beat_type = beat_order[min(index - 1, len(beat_order) - 1)]
        scene["beat_type"] = beat_type

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
  intro_hook, thumbnail_prompt, thumbnail_negative_prompt, chapters ve tags alanlarını koru veya iyileştir.
- thumbnail_text en fazla dört kelime olsun.
- intro_hook en fazla sekiz kelime olsun.
- Sahnelere chapter_index, chapter_title ve beat_type alanlarını ekle.
- Üç perde ve perde başına dört sahne kullan.

MEVCUT JSON:
{existing}
"""
    return generate_json(client, prompt, max_tokens=12000)


CHAPTER_BEATS = [
    (
        "Gerilim Birikiyor",
        ["hook", "context", "goal", "obstacle"],
    ),
    (
        "Kararlar ve Çatışma",
        ["first_move", "counter_move", "escalation", "breakthrough"],
    ),
    (
        "Sonuç ve Dönüşüm",
        ["climax", "immediate_result", "human_cost", "legacy"],
    ),
]


def _safe_title_from_topic(topic: str) -> str:
    clean = re.sub(r"\s+", " ", str(topic)).strip()
    return textwrap.shorten(clean, width=68, placeholder="…") or "Tarihten Bir Gece"


def _safe_thumbnail_text(topic: str) -> str:
    words = re.findall(r"\w+", str(topic), flags=re.UNICODE)
    selected = words[:4] or ["TARİHTEN", "BİR", "GECE"]
    return " ".join(selected).upper()


def _local_package_skeleton(
    topic: str,
    scene_count: int,
) -> dict[str, Any]:
    title = _safe_title_from_topic(topic)
    scenes: list[dict[str, Any]] = []

    for index in range(1, scene_count + 1):
        chapter_index = min(
            CHAPTER_COUNT,
            1 + ((index - 1) * CHAPTER_COUNT // max(1, scene_count)),
        )
        chapter_title, beats = CHAPTER_BEATS[chapter_index - 1]
        beat_position = (index - 1) % max(1, len(beats))
        beat_type = beats[min(beat_position, len(beats) - 1)]

        scenes.append({
            "scene_id": index,
            "chapter_index": chapter_index,
            "chapter_title": chapter_title,
            "beat_type": beat_type,
            "narration_idea": (
                f"{topic} konusunun {chapter_title.lower()} bölümündeki "
                f"{beat_type} anlatı adımı"
            ),
            "narration_text": "",
            "visual_goal": (
                f"{topic} ile doğrudan ilişkili, dönem ve coğrafyaya uygun "
                f"{beat_type} sahnesi"
            ),
            "image_prompt": (
                f"Photorealistic historical documentary reconstruction directly "
                f"related to {topic}. Show the exact period, location and daily "
                f"environment implied by the topic. Cinematic 16:9 frame, "
                f"historically plausible architecture, clothing, tools and "
                f"materials, restrained moonlight and amber practical light."
            ),
            "negative_prompt": (
                "wrong century, wrong civilization, modern objects, fantasy, "
                "text, logo, collage, museum display"
            ),
            "duration_weight": 1.0,
            "transition": "fadeblack" if beat_type in {
                "turning_point", "aftermath"
            } else "fade",
            "transition_duration": 0.62,
            "ambient_profile": (
                "interior_room"
                if beat_type in {"craft", "evidence"}
                else "exterior_wind"
            ),
            "importance": (
                "high"
                if beat_type in {"hook", "turning_point", "reflection"}
                else "normal"
            ),
            "continuity_bridge": (
                "Önceki bölümde kurulan düşüncenin doğal devamı."
            ),
        })

    return {
        "topic_interpretation": (
            f"{topic} konusu, insan deneyimi ile tarihsel bağlamı "
            "birlikte ele alan sakin bir gece belgeseli olarak yorumlandı."
        ),
        "historical_scope": (
            f"{topic} başlığının işaret ettiği dönem, coğrafya ve toplumsal bağlam."
        ),
        "video_title": title,
        "thumbnail_text": _safe_thumbnail_text(topic),
        "intro_hook": "Gece çökerken hayat nasıl görünüyordu?",
        "description": (
            f"Bu bölümde {topic} başlığının tarihsel dünyasına sakin ve "
            "insan odaklı bir anlatımla yaklaşıyoruz.\n\n"
            "Kesin olarak bilinenlerle olası ayrıntılar birbirinden ayrılarak, "
            "mekânın ve gündelik hayatın izleri takip ediliyor."
        ),
        "narration": "",
        "visual_identity": (
            "Premium late-night historical documentary with restrained "
            "blue-black shadows and subtle amber practical light."
        ),
        "world_bible": {
            "period": "Konudan çıkarılan tarihsel dönem",
            "location": "Konudan çıkarılan coğrafya",
            "architecture": "Döneme ve coğrafyaya uygun gerçekçi mimari",
            "clothing": "Döneme uygun doğal kumaşlar ve işlevsel kıyafetler",
            "materials": "Taş, ahşap, toprak, metal ve yerel malzemeler",
            "lighting": "Ay ışığı, kandil, ocak veya meşale gibi dönem ışıkları",
            "palette": "Lacivert gölgeler, düşük doygunluk, kısık amber vurgular",
            "forbidden": [
                "modern teknoloji",
                "yanlış medeniyet",
                "yanlış yüzyıl",
                "fantastik mimari",
                "yazı ve logo",
            ],
        },
        "thumbnail_prompt": (
            f"Single cohesive cinematic historical scene directly related to "
            f"{topic}, iconic setting on the right third, dark clean negative "
            "space on the left, photorealistic, premium documentary, 16:9."
        ),
        "thumbnail_negative_prompt": (
            "text, logo, collage, split screen, wrong civilization, modern objects"
        ),
        "scenes": scenes,
        "chapters": [item[0] for item in CHAPTER_BEATS],
        "tags": ["tarih", "belgesel", "uyku için tarih"],
    }


def _merge_package_defaults(
    payload: dict[str, Any] | None,
    topic: str,
    scene_count: int,
) -> dict[str, Any]:
    defaults = _local_package_skeleton(topic, scene_count)
    if not isinstance(payload, dict):
        return defaults

    for key, value in defaults.items():
        if key == "scenes":
            continue
        if payload.get(key) in (None, "", [], {}):
            payload[key] = value

    supplied_scenes = payload.get("scenes")
    if not isinstance(supplied_scenes, list):
        supplied_scenes = []

    merged_scenes: list[dict[str, Any]] = []
    for index in range(scene_count):
        base = dict(defaults["scenes"][index])
        if index < len(supplied_scenes) and isinstance(
            supplied_scenes[index], dict
        ):
            candidate = supplied_scenes[index]
            for key, value in candidate.items():
                if value not in (None, ""):
                    base[key] = value
        merged_scenes.append(base)

    payload["scenes"] = merged_scenes
    _normalize_package(payload, scene_count)
    return payload


def _safe_chapter_index(chapter_index: Any) -> int:
    try:
        value = int(chapter_index)
    except (TypeError, ValueError):
        value = CHAPTER_COUNT

    available = max(1, len(CHAPTER_BEATS))
    return max(1, min(available, value))


def _local_expand_chapter(
    seed_text: str,
    topic: str,
    chapter_index: int,
    target_words: int,
) -> str:
    seed = re.sub(r"\s+", " ", str(seed_text)).strip()
    safe_index = _safe_chapter_index(chapter_index)
    chapter_title = CHAPTER_BEATS[safe_index - 1][0]

    # This is only an emergency length guard. It expands cause-and-effect
    # reasoning without inventing dates, names or visual atmosphere.
    additions = [
        (
            f"{chapter_title} aşamasında asıl belirleyici olan, önceki kararların "
            f"yeni bir zorunluluk yaratmasıydı. {topic} çerçevesindeki gelişmeler "
            "tek bir anda ortaya çıkmadı; birbirini etkileyen tercihler, "
            "beklentiler ve karşı hamleler sonucunda biçimlendi."
        ),
        (
            "Bu noktada tarafların hedefleri aynı değildi. Bir grubun güvenlik, "
            "düzen veya iktidar için attığı adım, diğer tarafın seçeneklerini "
            "daralttı ve yeni bir karar alınmasını zorunlu hâle getirdi."
        ),
        (
            "Kaynaklar her ayrıntıyı aynı açıklıkta aktarmadığı için, kesin olarak "
            "bilinen gelişmelerle daha sonra yapılan yorumları birbirinden ayırmak "
            "gerekir. Buna rağmen olayların sırası, kararların sonuçlarını "
            "anlamaya yetecek kadar belirgindir."
        ),
        (
            "Alınan karar yalnızca o anı değiştirmedi. Yönetim, ekonomi, güvenlik "
            "ve gündelik yaşam üzerinde birbirine bağlı sonuçlar doğurdu; sonraki "
            "adımlar da bu yeni koşullara cevap vermek zorunda kaldı."
        ),
        (
            "Bu gelişmenin insanlara yansıması aynı ölçüde önemliydi. Büyük "
            "siyasi veya askerî değişimler, barınma, çalışma, ticaret ve toplumsal "
            "düzen gibi gündelik alanlarda somut karşılıklar üretti."
        ),
        (
            "Sonuçta ortaya çıkan dönüşüm, tek bir kişinin veya tek bir kararın "
            "ürünü değildi. Önceki gerilimler, mevcut imkânlar ve verilen "
            "karşılıklar birleşerek olayın yönünü belirledi."
        ),
    ]

    parts = [seed] if seed else []
    cursor = 0
    while _word_count(" ".join(parts)) < target_words:
        parts.append(additions[cursor % len(additions)])
        cursor += 1
        if cursor >= 18:
            break

    combined = re.sub(r"\s+", " ", " ".join(parts)).strip()
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", combined)
        if item.strip()
    ]

    selected: list[str] = []
    for sentence in sentences:
        proposed = " ".join(selected + [sentence])
        if selected and _word_count(proposed) > target_words + 16:
            break
        selected.append(sentence)

    result = " ".join(selected).strip()
    return result or combined


def _chapter_generation_prompt(
    topic: str,
    payload: dict[str, Any],
    chapter_index: int,
    scenes: list[dict[str, Any]],
    target_words: int,
) -> str:
    chapter_title = CHAPTER_BEATS[chapter_index - 1][0]
    scene_context = [
        {
            "scene_id": scene.get("scene_id"),
            "beat_type": scene.get("beat_type"),
            "narration_idea": scene.get("narration_idea"),
            "visual_goal": scene.get("visual_goal"),
        }
        for scene in scenes
    ]
    scene_count = len(scenes)
    previous_tail = str(
        payload.get("_previous_chapter_tail", "")
    ).strip()

    continuation_rule = (
        "Bu ilk bölüm. İlk 25 kelime içinde ana çatışmayı veya soruyu kur."
        if chapter_index == 1
        else (
            "Yeni bir giriş yapma. Aşağıdaki önceki bölümün son düşüncesinden "
            "doğrudan devam et:\n" + previous_tail
        )
    )

    return f"""
Yalnızca geçerli JSON üret:
{{
  "chapter_title": "Kısa Türkçe bölüm adı",
  "chapter_text": "Tek parça akıcı Türkçe anlatım",
  "scene_texts": [
    "Sahne metni 1"
  ]
}}

KONU:
{topic}

BÖLÜM:
{chapter_index}/{CHAPTER_COUNT} — {chapter_title}

ANLATIM MİMARİSİ:
- Bu bir görsel betimleme metni değil, olay anlatısıdır.
- Metnin en az yüzde 65'i olay, karar, neden ve sonuç içermeli.
- Tarihsel bağlam yaklaşık yüzde 20 olabilir.
- Duyusal veya atmosferik betimleme en fazla yüzde 15 olabilir.
- Her paragraf hikâyeyi ileri götürmeli.
- Kim ne istedi, ne engelledi, hangi karar alındı ve bunun sonucu ne oldu
  sorularından en az ikisini her sahnede cevapla.
- "Gece çökerken", "taş duvarlar", "sessizlik", "ay ışığı",
  "karanlığın içinde" gibi kalıpları bölüm boyunca toplam bir defadan
  fazla kullanma.
- Kamera, görüntü, sahne, kadraj veya izleyici kelimelerini kullanma.
- Bilgi listesi yazma.
- Aynı fikri farklı kelimelerle tekrar etme.
- Yapay dramatizasyon, fragman tonu ve şiirsel dolgu kullanma.
- Tarihsel belirsizliği kesin gerçek gibi sunma.
- Özel isim, tarih ve kararlar anlatının omurgasını oluştursun.
- Bölüm yaklaşık {target_words} kelime olsun.
- Tam {scene_count} scene_texts üret.
- scene_texts birleşince chapter_text ile aynı anlatıyı oluştursun.
- {continuation_rule}

VİDEO BAĞLAMI:
{json.dumps({
    "historical_scope": payload.get("historical_scope", ""),
    "video_title": payload.get("video_title", ""),
    "topic_interpretation": payload.get("topic_interpretation", ""),
}, ensure_ascii=False)}

BU BÖLÜMÜN ANLATI ADIMLARI:
{json.dumps(scene_context, ensure_ascii=False)}
"""


def _generate_long_chapter(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    chapter_index: int,
    scenes: list[dict[str, Any]],
    target_words: int,
) -> tuple[str, list[str], str]:
    prompt = _chapter_generation_prompt(
        topic,
        payload,
        chapter_index,
        scenes,
        target_words,
    )

    model_name = "local-fallback"
    chapter_text = ""
    scene_texts: list[str] = []
    scene_count = len(scenes)

    try:
        result, model_name = generate_json(
            client,
            prompt,
            max_tokens=4300,
        )
        chapter_text = re.sub(
            r"\s+", " ", str(result.get("chapter_text", ""))
        ).strip()
        supplied = result.get("scene_texts", [])
        if isinstance(supplied, list):
            scene_texts = [
                re.sub(r"\s+", " ", str(item)).strip()
                for item in supplied[:scene_count]
            ]
    except Exception as exc:
        print(
            f"Bölüm {chapter_index} üretimi atlandı; "
            f"güvenli anlatı iskeleti kullanılacak: {exc}"
        )

    if len(scene_texts) == scene_count:
        joined = " ".join(
            item for item in scene_texts if item
        ).strip()
        if _word_count(joined) >= max(
            80,
            _word_count(chapter_text) * 0.72,
        ):
            chapter_text = joined

    # No atmospheric filler. A shorter but coherent act is preferred over
    # repetitive generic description.
    if _word_count(chapter_text) < 95:
        seed = " ".join(
            str(scene.get("narration_text", "")).strip()
            for scene in scenes
        ).strip()
        chapter_text = seed or (
            f"{topic} başlığındaki bu aşamada tarafların hedefleri, "
            "kararları ve bu kararların doğrudan sonuçları belirleyici oldu. "
            "Gelişmeler birbirinden bağımsız değildi; her hamle bir sonraki "
            "kararı zorladı ve dengeleri değiştirdi."
        )
        model_name = f"{model_name} + compact-local-recovery"

    scene_texts = _split_narration_into_scenes(
        chapter_text,
        scene_count,
    )
    while len(scene_texts) < scene_count:
        scene_texts.append("")

    return chapter_text, scene_texts[:scene_count], model_name


def _force_long_form_package(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any] | None,
    target_words: int,
    minimum_words: int,
    scene_count: int,
) -> tuple[dict[str, Any], list[str]]:
    payload = _merge_package_defaults(
        payload,
        topic,
        scene_count,
    )

    chapter_groups: list[list[dict[str, Any]]] = [
        [] for _ in range(CHAPTER_COUNT)
    ]
    for index, scene in enumerate(payload["scenes"]):
        chapter_index = min(
            CHAPTER_COUNT,
            1 + index * CHAPTER_COUNT // max(1, scene_count),
        )
        scene["chapter_index"] = chapter_index
        scene["chapter_title"] = CHAPTER_BEATS[chapter_index - 1][0]
        chapter_beats = CHAPTER_BEATS[chapter_index - 1][1]
        scene["beat_type"] = chapter_beats[
            len(chapter_groups[chapter_index - 1]) % len(chapter_beats)
        ]
        chapter_groups[chapter_index - 1].append(scene)

    per_chapter_target = max(
        205,
        math.ceil((target_words + 15) / CHAPTER_COUNT),
    )
    chapter_models: list[str] = []
    chapter_texts: list[str] = []

    for chapter_index, scenes in enumerate(
        chapter_groups,
        start=1,
    ):
        chapter_text, scene_texts, model = _generate_long_chapter(
            client,
            topic,
            payload,
            chapter_index,
            scenes,
            per_chapter_target,
        )
        chapter_models.append(model)
        chapter_texts.append(chapter_text)
        previous_sentences = [
            item.strip()
            for item in re.split(
                r"(?<=[.!?])\s+",
                chapter_text,
            )
            if item.strip()
        ]
        payload["_previous_chapter_tail"] = " ".join(
            previous_sentences[-2:]
        )

        for scene, scene_text in zip(scenes, scene_texts):
            scene["narration_text"] = scene_text
            scene["narration_idea"] = textwrap.shorten(
                scene_text,
                width=180,
                placeholder="…",
            )

    payload["narration"] = re.sub(
        r"\s+",
        " ",
        " ".join(chapter_texts),
    ).strip()

    if _word_count(payload["narration"]) < minimum_words:
        deficit_target = max(
            target_words,
            minimum_words + 45,
        )
        payload["narration"] = _local_expand_chapter(
            payload["narration"],
            topic,
            CHAPTER_COUNT,
            deficit_target,
        )
        all_scene_texts = _split_narration_into_scenes(
            payload["narration"],
            scene_count,
        )
        for scene, scene_text in zip(
            payload["scenes"],
            all_scene_texts,
        ):
            scene["narration_text"] = scene_text
            scene["narration_idea"] = textwrap.shorten(
                scene_text,
                width=180,
                placeholder="…",
            )
        chapter_models.append("final-local-length-guard")

    payload.pop("_previous_chapter_tail", None)
    _normalize_package(payload, scene_count)
    return payload, chapter_models


def build_video_package(
    client: genai.Client,
    topic: str,
    target_seconds: int,
    scene_count: int,
) -> tuple[dict[str, Any], str]:
    target_words = max(
        640,
        min(710, round(target_seconds * 2.28)),
    )
    minimum_words = max(
        580,
        round(target_words * 0.88),
    )

    prompt = f"""
Yalnızca geçerli JSON üret. Markdown yazma.

ROL:
Sen tarih araştırması, sakin belgesel senaryosu, sinematografik sahne
planlama ve YouTube paketleme alanlarında çalışan kıdemli bir editörsün.

KONU:
{topic}

ÖNEMLİ:
Bu ilk çağrıda önce güçlü ve eksiksiz bir video planı kur.
Metin beklenenden kısa kalırsa sistem üç perdeyi ayrı ayrı tamamlayacaktır.
Kullanıcıdan ek bilgi isteme.

HEDEF:
Yaklaşık {target_seconds} saniyelik, uyku öncesi dinlemeye uygun,
üç perdeli Türkçe tarih videosu.
Tam {scene_count} sahne üret.

JSON:
{{
  "topic_interpretation": "Kısa yorum",
  "historical_scope": "Dönem, yer ve bağlam",
  "video_title": "Türkçe başlık",
  "thumbnail_text": "En fazla dört kelime",
  "intro_hook": "En fazla sekiz kelime",
  "description": "İki kısa paragraf",
  "narration": "Mevcut olabildiğince uzun Türkçe anlatım",
  "visual_identity": "Tek film sanat yönetimi",
  "world_bible": {{
    "period": "Dönem",
    "location": "Coğrafya",
    "architecture": "Mimari",
    "clothing": "Kıyafet",
    "materials": "Malzemeler",
    "lighting": "Işık",
    "palette": "Renk paleti",
    "forbidden": ["Yanlış unsurlar"]
  }},
  "thumbnail_prompt": "İngilizce kapak promptu",
  "thumbnail_negative_prompt": "Kaçınılacak unsurlar",
  "scenes": [
    {{
      "scene_id": 1,
      "chapter_index": 1,
      "chapter_title": "Bölüm adı",
      "beat_type": "hook",
      "narration_idea": "Sahnenin ana fikri",
      "narration_text": "Sahne metni",
      "visual_goal": "Somut görsel hedef",
      "image_prompt": "Ayrıntılı İngilizce 16:9 prompt",
      "negative_prompt": "Sahneye özgü negatifler",
      "duration_weight": 1.0,
      "transition": "fade",
      "transition_duration": 0.62,
      "ambient_profile": "exterior_wind",
      "importance": "normal",
      "continuity_bridge": "Önceki sahneden doğal geçiş"
    }}
  ],
  "chapters": ["Üç perde adı"],
  "tags": ["etiket"]
}}

KURALLAR:
- Üç perde ve perde başına dört sahne kullan.
- Sahne sırası: hook, context, goal, obstacle, first_move, counter_move,
  escalation, breakthrough, climax, immediate_result, human_cost, legacy.
- Anlatımın en az yüzde 65'i olay, karar, neden ve sonuç içersin.
- Betimleme en fazla yüzde 15 olsun.
- Görsel promptları ayrıntılı olabilir; narration_text görsel tarif etmesin.
- Görseller konu, dönem ve coğrafyayla doğrudan ilişkili olsun.
- Yanlış medeniyet, modern nesne, fantastik mimari, yazı, logo ve kolaj olmasın.
- Aynı film paleti bütün sahnelerde korunsun.
- Tarihsel belirsizlikleri kesin gerçek gibi sunma.
"""

    payload: dict[str, Any] | None = None
    model_history: list[str] = []

    try:
        payload, model = generate_json(
            client,
            prompt,
            max_tokens=12000,
        )
        model_history.append(model)
    except Exception as exc:
        print(
            "İlk video paketi üretilemedi; yerel güvenli iskelet "
            f"kullanılacak: {exc}"
        )
        payload = _local_package_skeleton(
            topic,
            scene_count,
        )
        model_history.append("local-package-skeleton")

    payload = _merge_package_defaults(
        payload,
        topic,
        scene_count,
    )

    structural_issues = [
        issue
        for issue in _package_issues(
            payload,
            scene_count,
            minimum_words=0,
        )
        if not issue.startswith("Senaryo kısa:")
    ]

    # Only one whole-package repair is allowed. Repeating the same giant JSON
    # three times caused the previous failure and wasted time.
    if structural_issues:
        print(
            "Yapısal paket onarımı: "
            + " | ".join(structural_issues)
        )
        try:
            payload, repair_model = _repair_video_package(
                client,
                topic,
                payload,
                structural_issues,
                target_words,
                scene_count,
            )
            model_history.append(repair_model)
            payload = _merge_package_defaults(
                payload,
                topic,
                scene_count,
            )
        except Exception as exc:
            print(
                "Yapısal yapay zekâ onarımı atlandı; yerel varsayılanlar "
                f"korunacak: {exc}"
            )
            model_history.append("local-structural-recovery")

    current_words = _word_count(
        payload.get("narration", "")
    )
    print(
        f"Uzun format kontrolü: {current_words} kelime; "
        f"hedef en az {minimum_words}."
    )

    if current_words < minimum_words:
        print(
            "FAIL-SOFT STORY RECOVERY: Senaryo üç ayrı perde "
            "olarak tamamlanıyor."
        )
        payload, chapter_models = _force_long_form_package(
            client,
            topic,
            payload,
            target_words,
            minimum_words,
            scene_count,
        )
        model_history.extend(chapter_models)

    # Absolute local guard. A short script is never allowed to terminate the
    # workflow again.
    final_words = _word_count(
        payload.get("narration", "")
    )
    if final_words < minimum_words:
        print(
            "LOCAL LENGTH GUARD: kalan kelime açığı yerel olarak "
            "tamamlanıyor."
        )
        payload, chapter_models = _force_long_form_package(
            client,
            topic,
            payload,
            max(target_words, minimum_words + 60),
            minimum_words,
            scene_count,
        )
        model_history.extend(chapter_models)

    payload = _merge_package_defaults(
        payload,
        topic,
        scene_count,
    )

    final_words = _word_count(
        payload.get("narration", "")
    )
    print(
        f"FAIL-SOFT STORY READY: {final_words} kelime, "
        f"{len(payload.get('scenes', []))} sahne."
    )

    # Remaining non-fatal warnings are logged, never raised here.
    payload, continuity_model = narrative_continuity_pass(
        client,
        topic,
        payload,
        scene_count,
    )
    model_history.append(continuity_model)

    warnings = _package_issues(
        payload,
        scene_count,
        minimum_words,
    )
    if warnings:
        print(
            "Paket uyarıları yerel güvenli değerlerle devam ediyor: "
            + " | ".join(warnings)
        )

    _normalize_package(payload, scene_count)
    return payload, " -> ".join(model_history)


def _atmosphere_sentence_count(narration: str) -> int:
    markers = (
        "gece",
        "sessiz",
        "taş duvar",
        "ay ış",
        "karanlık",
        "rüzgâr",
        "meşale",
    )
    sentences = [
        item.strip().lower()
        for item in re.split(
            r"(?<=[.!?])\s+",
            narration,
        )
        if item.strip()
    ]
    return sum(
        1
        for sentence in sentences
        if sum(marker in sentence for marker in markers) >= 2
    )


def narrative_continuity_pass(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    scene_count: int,
) -> tuple[dict[str, Any], str]:
    source_narration = re.sub(
        r"\s+",
        " ",
        str(payload.get("narration", "")),
    ).strip()

    prompt = f"""
Yalnızca geçerli JSON üret:
{{
  "narration": "Tek parça, başlıksız Türkçe anlatım"
}}

KONU:
{topic}

MEVCUT METİN:
{source_narration}

GÖREV:
Metni tek bir belgesel hikâyesi gibi yeniden kurgula.

KURALLAR:
- 620 ile 730 kelime arasında kal.
- Bilgi ve olay sırasını koru; yeni kesin bilgi uydurma.
- Üç perde net hissedilsin: gerilim, çatışma, sonuç.
- Her paragraf bir neden, karar, eylem veya sonuç eklesin.
- Betimleme toplam metnin yüzde 15'ini geçmesin.
- İlk 25 kelimede ana soruyu veya çatışmayı kur.
- Yeni bölüm başlatan tekrar girişlerini kaldır.
- Aynı olayı veya fikri tekrar etme.
- "gece", "sessizlik", "taş duvar", "ay ışığı" gibi atmosfer
  kalıplarını yalnızca gerçekten gerekli olduğunda kullan.
- Son bölüm özet listesi gibi değil, olayın etkisini anlatarak bitsin.
- Başlık, madde işareti, bölüm adı ve kamera tarifi kullanma.
"""

    try:
        result, model = generate_json(
            client,
            prompt,
            max_tokens=5600,
        )
        candidate = re.sub(
            r"\s+",
            " ",
            str(result.get("narration", "")),
        ).strip()
        words = _word_count(candidate)
        if not 580 <= words <= 760:
            raise ValueError(
                f"Süreklilik metni uzunluğu uygun değil: {words}"
            )
        if _atmosphere_sentence_count(candidate) > 5:
            raise ValueError(
                "Süreklilik metni hâlâ fazla betimleyici."
            )

        payload["narration"] = candidate
        scene_texts = _split_narration_into_scenes(
            candidate,
            scene_count,
        )
        for scene, scene_text in zip(
            payload["scenes"],
            scene_texts,
        ):
            scene["narration_text"] = scene_text
            scene["narration_idea"] = textwrap.shorten(
                scene_text,
                width=170,
                placeholder="…",
            )
        return payload, model
    except Exception as exc:
        print(
            "Süreklilik editörü mevcut güçlü metni korudu: "
            f"{exc}"
        )
        return payload, "continuity-pass-skipped"



def story_director_pass(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    scene_count: int,
) -> tuple[dict[str, Any], str]:
    current = json.dumps(payload, ensure_ascii=False)
    original_words = _word_count(str(payload.get("narration", "")))
    minimum_words = max(540, round(original_words * 0.86))

    prompt = f"""
Yalnızca geçerli ve eksiksiz JSON üret. Markdown yazma.

Sen uzun format tarih belgeseli hikâye yönetmenisin.
Aşağıdaki paketi yaklaşık beş dakikalık, tek nefeste dinlenen,
neden-sonuç ilişkisi güçlü bir gece belgeseline dönüştür.

KONU:
{topic}

UZUN FORMAT YAPI:
- Tam {scene_count} sahne ve tam üç perde kullan.
- Her perdede dört sahne bulunsun.
- Perde 1: hook, context, goal, obstacle.
- Perde 2: first_move, counter_move, escalation, breakthrough.
- Perde 3: climax, immediate_result, human_cost, legacy.

- İlk otuz saniyede merkezî soruyu kur.
- Her sahne önceki sahnenin düşüncesini sürdürsün.
- Bilgi listesi, maddeleme, tekrar ve her sahnede yeniden giriş yapma.
- İnsan deneyimi, mekân, gündelik ayrıntı ve tarihsel bağlam dengeli olsun.
- Bilinmeyen ayrıntıları kesin gerçek gibi sunma.
- Son bölüm, konuyu günümüze kalan izlerle sakin biçimde bağlasın.
- narration en az {minimum_words} Türkçe kelime olsun.
- narration_text alanları birleştiğinde narration ile aynı metni oluştursun.
- intro_hook en fazla sekiz kelime ve video başlığından farklı olsun.
- Her sahnede chapter_index, chapter_title, beat_type ve continuity_bridge bulunsun.
- Görsel promptu yalnızca o sahnenin somut anlatımına odaklansın.
- Aynı film paleti, dönem, coğrafya ve mimari bütün sahnelerde korunsun.
- chapters alanı üç kısa Türkçe perde başlığı içersin.

MEVCUT JSON:
{current}
"""
    try:
        refined, model = generate_json(client, prompt, max_tokens=16000)
        issues = _package_issues(refined, scene_count, minimum_words)
        if issues:
            print("Long-form Story Director doğrulama başarısız:", issues)
            return payload, "fallback-original"
        _normalize_package(refined, scene_count)
        _ensure_scene_narration(refined)
        return refined, model
    except Exception as exc:
        print("Long-form Story Director geçici olarak atlandı:", exc)
        return payload, "fallback-original"


def scene_tts_directive(text: str, chapter_index: int, chapter_count: int) -> str:
    return f"""
Read only the Turkish transcript after the separator.

Use the exact same mature Turkish male documentary narrator in every chapter.
This is chapter {chapter_index} of {chapter_count}; it must sound like one
continuous five-minute story, not a fresh introduction.

Calm, grounded, warm and intimate. Standard Turkey Turkish.
Late-night historical documentary. Natural documentary pace.
Natural pauses, clear articulation and gentle sentence endings.
No trailer voice, no advertisement, no newsreader tone.
Do not add, remove or paraphrase words. No music or effects.

--- TRANSCRIPT ---
{text}
--- END ---
""".strip()


def _is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
        )
    )


def _ensure_edge_tts() -> Any:
    try:
        import edge_tts
        return edge_tts
    except ImportError:
        print("Ücretsiz yedek ses motoru kuruluyor: edge-tts")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--quiet",
                "edge-tts>=7.0,<8.0",
            ],
            check=True,
            timeout=180,
        )
        import edge_tts
        return edge_tts


def synthesize_edge_tts(
    narration: str,
    target: Path,
) -> str:
    edge_tts = _ensure_edge_tts()
    mp3_target = target.with_suffix(".edge.mp3")
    mp3_target.unlink(missing_ok=True)

    async def _save() -> str:
        voice_name = "tr-TR-AhmetNeural"
        try:
            communicator = edge_tts.Communicate(
                narration,
                voice_name,
                rate="+8%",
                pitch="-1Hz",
                volume="+0%",
            )
            await communicator.save(str(mp3_target))
            return voice_name
        except Exception:
            voices = await edge_tts.list_voices()
            turkish = [
                item
                for item in voices
                if str(item.get("Locale", "")).lower() == "tr-tr"
            ]
            male = [
                item
                for item in turkish
                if str(item.get("Gender", "")).lower() == "male"
            ]
            candidates = male or turkish
            if not candidates:
                raise RuntimeError("Edge TTS içinde Türkçe ses bulunamadı.")
            selected = str(candidates[0]["ShortName"])
            communicator = edge_tts.Communicate(
                narration,
                selected,
                rate="+8%",
                pitch="-1Hz",
                volume="+0%",
            )
            await communicator.save(str(mp3_target))
            return selected

    import asyncio
    voice = asyncio.run(_save())

    if not mp3_target.exists() or mp3_target.stat().st_size < 8000:
        raise RuntimeError("Edge TTS ses dosyası üretmedi.")

    run([
        "ffmpeg", "-y",
        "-i", str(mp3_target),
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(target),
    ])
    mp3_target.unlink(missing_ok=True)

    if not target.exists() or ffprobe_duration(target) < 8.0:
        raise RuntimeError("Edge TTS sesi beklenenden kısa.")
    return f"edge-tts/{voice}"


def synthesize_short_segment(
    client: genai.Client,
    text: str,
    scene_index: int,
    scene_count: int,
    target: Path,
) -> str:
    configured = os.getenv("TTS_MODEL", "").strip()
    chain = model_chain(client, [configured, *TTS_MODELS])
    last_error: Exception | None = None

    # One request per Gemini model. A quota error must never trigger repeated
    # waiting and consume the remaining free-tier quota.
    for model in chain:
        try:
            print(
                f"Chapter TTS {scene_index}/{scene_count}: "
                f"model={model}, attempt=1/1"
            )
            response = client.models.generate_content(
                model=model,
                contents=scene_tts_directive(
                    text, scene_index, scene_count
                ),
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
                raise ValueError("TTS bölüm sesi döndürmedi.")
            inline = candidates[0].content.parts[0].inline_data
            data = inline.data if inline else None
            if not data or len(data) < 12000:
                raise ValueError("TTS bölüm sesi boş veya çok kısa.")
            write_wav(target, data)
            if ffprobe_duration(target) < 8.0:
                raise ValueError("TTS bölüm sesi beklenenden kısa.")
            return model
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            print(f"Gemini bölüm TTS atlandı: {model}: {exc}")
            continue

    print(
        f"Gemini bölüm TTS kullanılamadı; ücretsiz yedek ses motoruna "
        f"geçiliyor. Son hata: {last_error}"
    )
    return synthesize_edge_tts(text, target)


def append_scene_pause(
    source: Path,
    target: Path,
    pause_seconds: float,
) -> None:
    pause_seconds = max(0.0, float(pause_seconds))
    if pause_seconds <= 0.001:
        target.unlink(missing_ok=True)
        shutil.copy2(source, target)
        return

    run([
        "ffmpeg", "-y",
        "-i", str(source),
        "-af", f"apad=pad_dur={pause_seconds:.3f}",
        "-ar", "48000",
        "-c:a", "pcm_s16le",
        str(target),
    ])


def _verify_media_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} bulunamadı: {path}")
    if path.stat().st_size < 128:
        raise RuntimeError(f"{label} boş veya bozuk: {path}")


def concat_audio_files(files: list[Path], target: Path) -> None:
    if not files:
        raise ValueError("Birleştirilecek ses bulunamadı.")

    for index, path in enumerate(files, start=1):
        _verify_media_file(path, f"Ses parçası {index}")

    target.unlink(missing_ok=True)
    if len(files) == 1:
        shutil.copy2(files[0], target)
        _verify_media_file(target, "Birleştirilmiş ses")
        return

    command = ["ffmpeg", "-y"]
    filters: list[str] = []
    labels: list[str] = []

    for index, path in enumerate(files):
        command.extend(["-i", str(path)])
        label = f"a{index}"
        filters.append(
            f"[{index}:a]"
            "aresample=48000,"
            "aformat=sample_fmts=s16:channel_layouts=mono,"
            f"asetpts=N/SR/TB[{label}]"
        )
        labels.append(f"[{label}]")

    filters.append(
        "".join(labels)
        + f"concat=n={len(files)}:v=0:a=1[outa]"
    )

    command.extend([
        "-filter_complex", ";".join(filters),
        "-map", "[outa]",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(target),
    ])
    run(command)
    _verify_media_file(target, "Birleştirilmiş ses")

    expected = sum(ffprobe_duration(path) for path in files)
    actual = ffprobe_duration(target)
    if abs(actual - expected) > 0.75:
        raise RuntimeError(
            "Ses birleştirme süre kontrolünü geçemedi: "
            f"{actual:.2f}s / {expected:.2f}s"
        )


def write_concat_manifest(
    files: list[Path],
    target: Path,
) -> None:
    if not files:
        raise ValueError("Manifest için dosya bulunamadı.")

    lines: list[str] = []
    for index, path in enumerate(files, start=1):
        _verify_media_file(path, f"Video parçası {index}")
        safe_path = path.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{safe_path}'")

    # Real newline characters. Never write a literal backslash+n sequence.
    content = "\n".join(lines) + "\n"
    target.write_text(content, encoding="utf-8")

    readback = target.read_text(encoding="utf-8")
    if "\\nfile" in readback or readback.count("\n") != len(files):
        raise RuntimeError("FFmpeg concat manifest satır sonu kontrolü başarısız.")


def _atempo_chain(factor: float) -> str:
    factor = max(0.25, min(4.0, factor))
    values: list[float] = []
    while factor < 0.5:
        values.append(0.5)
        factor /= 0.5
    while factor > 2.0:
        values.append(2.0)
        factor /= 2.0
    values.append(factor)
    return ",".join(f"atempo={value:.6f}" for value in values)


def fit_audio_duration(
    source: Path,
    target: Path,
    target_seconds: float,
) -> None:
    target_seconds = max(1.0, float(target_seconds))
    current = ffprobe_duration(source)
    factor = current / target_seconds
    filter_chain = (
        f"{_atempo_chain(factor)},"
        "aresample=48000,"
        "apad=pad_dur=3,"
        f"atrim=duration={target_seconds:.3f},"
        "asetpts=N/SR/TB"
    )
    run([
        "ffmpeg", "-y",
        "-i", str(source),
        "-af", filter_chain,
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(target),
    ])
    final_duration = ffprobe_duration(target)
    if abs(final_duration - target_seconds) > 0.8:
        raise RuntimeError(
            "Ses hedef süreye getirilemedi: "
            f"{final_duration:.2f}s / {target_seconds:.2f}s"
        )


def prepare_natural_narration(
    source: Path,
    target: Path,
    preferred_seconds: float,
) -> tuple[float, float]:
    """Preserve natural speech: max 2% slower, max 8% faster."""
    preferred_seconds = max(1.0, float(preferred_seconds))
    current = ffprobe_duration(source)
    if current <= 0:
        raise RuntimeError("Seslendirme süresi okunamadı.")
    ideal_factor = current / preferred_seconds
    tempo_factor = max(0.98, min(1.08, ideal_factor))
    target.unlink(missing_ok=True)
    if abs(tempo_factor - 1.0) < 0.006:
        shutil.copy2(source, target)
    else:
        run([
            "ffmpeg", "-y", "-i", str(source),
            "-af", f"atempo={tempo_factor:.6f},aresample=48000,asetpts=N/SR/TB",
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
        ])
    actual = ffprobe_duration(target)
    if actual <= 0:
        raise RuntimeError("Doğal hızdaki seslendirme oluşturulamadı.")
    print(
        "NATURAL VOICE TIMING: "
        f"raw={current:.2f}s, tempo={tempo_factor:.3f}, "
        f"final={actual:.2f}s, preferred={preferred_seconds:.2f}s"
    )
    return actual, tempo_factor


def _allocate_exact_duration(
    total_seconds: float,
    weights: list[float],
    minimum: float,
) -> list[float]:
    if not weights:
        return []
    count = len(weights)
    minimum = min(minimum, total_seconds / count)
    base_total = minimum * count
    remaining = max(0.0, total_seconds - base_total)
    safe_weights = [max(1.0, float(value)) for value in weights]
    weight_sum = sum(safe_weights)
    result = [
        minimum + remaining * value / weight_sum
        for value in safe_weights
    ]
    result[-1] += total_seconds - sum(result)
    return result


def _chapter_groups(
    scenes: list[dict[str, Any]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    groups: list[list[tuple[int, dict[str, Any]]]] = [
        [] for _ in range(CHAPTER_COUNT)
    ]
    for index, scene in enumerate(scenes):
        try:
            chapter = int(scene.get("chapter_index", 1))
        except (TypeError, ValueError):
            chapter = 1 + index * CHAPTER_COUNT // max(1, len(scenes))
        chapter = max(1, min(CHAPTER_COUNT, chapter))
        groups[chapter - 1].append((index, scene))

    if any(not group for group in groups):
        groups = [[] for _ in range(CHAPTER_COUNT)]
        for index, scene in enumerate(scenes):
            chapter = min(
                CHAPTER_COUNT - 1,
                index * CHAPTER_COUNT // max(1, len(scenes)),
            )
            groups[chapter].append((index, scene))
    return groups


def synthesize_scene_narration(
    client: genai.Client,
    scenes: list[dict[str, Any]],
    output_audio: Path,
    target_seconds: float,
) -> tuple[list[float], str]:
    chapter_dir = WORK / "chapter-audio"
    if chapter_dir.exists():
        shutil.rmtree(chapter_dir)
    chapter_dir.mkdir(parents=True)

    groups = _chapter_groups(scenes)
    chapter_word_counts = [
        sum(
            _word_count(str(scene.get("narration_text", "")))
            for _, scene in group
        )
        for group in groups
    ]
    chapter_targets = _allocate_exact_duration(
        target_seconds,
        chapter_word_counts,
        minimum=48.0,
    )

    fitted_chapters: list[Path] = []
    scene_durations = [0.0] * len(scenes)
    models: list[str] = []

    for chapter_index, (group, chapter_target) in enumerate(
        zip(groups, chapter_targets),
        start=1,
    ):
        chapter_text = " ".join(
            re.sub(
                r"\\s+", " ",
                str(scene.get("narration_text", ""))
            ).strip()
            for _, scene in group
        ).strip()
        if not chapter_text:
            raise RuntimeError(f"Bölüm {chapter_index} metni boş.")

        raw = chapter_dir / f"chapter_{chapter_index:02d}_raw.wav"
        normalized = chapter_dir / f"chapter_{chapter_index:02d}_norm.wav"
        fitted = chapter_dir / f"chapter_{chapter_index:02d}_fitted.wav"

        model = synthesize_short_segment(
            client,
            chapter_text,
            chapter_index,
            len(groups),
            raw,
        )
        normalize_audio(raw, normalized)
        fit_audio_duration(normalized, fitted, chapter_target)
        fitted_chapters.append(fitted)
        models.append(model)

        scene_weights = [
            max(1, _word_count(str(scene.get("narration_text", ""))))
            for _, scene in group
        ]
        allocated = _allocate_exact_duration(
            chapter_target,
            scene_weights,
            minimum=14.0,
        )
        for (scene_position, _), duration in zip(group, allocated):
            scene_durations[scene_position] = duration

    concat_audio_files(fitted_chapters, output_audio)
    final_duration = ffprobe_duration(output_audio)
    if abs(final_duration - target_seconds) > 1.0:
        corrected = chapter_dir / "story-corrected.wav"
        fit_audio_duration(output_audio, corrected, target_seconds)
        shutil.copy2(corrected, output_audio)

    return scene_durations, " -> ".join(dict.fromkeys(models))


def create_scene_srt(
    scenes: list[dict[str, Any]],
    durations: list[float],
    intro_seconds: float,
    target: Path,
) -> None:
    cursor = intro_seconds
    blocks: list[str] = []
    for index, (scene, duration) in enumerate(zip(scenes, durations), start=1):
        scene_text = re.sub(r"\s+", " ", str(scene.get("narration_text", ""))).strip()
        blocks.append(
            f"{index}\n{timestamp(cursor)} --> {timestamp(cursor + duration)}\n"
            f"{scene_text}\n"
        )
        cursor += duration
    target.write_text("\n".join(blocks), encoding="utf-8")


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
    """Build a Cloudflare-safe request with a strict character guard."""
    prompt = _compact_text(prompt, 1040)
    negative = _compact_text(negative, MAX_NEGATIVE_PROMPT_CHARS)

    # FLUX fallback models count exclusions in the same prompt field.
    final_prompt = _compact_text(
        f"{prompt}. Avoid: {negative}. Landscape 16:9, cinematic historical film still, no text.",
        MAX_IMAGE_PROMPT_CHARS,
    )
    print(
        f"V5.2 prompt guard: model={model}, "
        f"prompt_chars={len(final_prompt)}, negative_chars={len(negative)}"
    )

    if len(final_prompt) > MAX_IMAGE_PROMPT_CHARS:
        raise RuntimeError(
            f"Internal prompt guard failed: {len(final_prompt)} characters"
        )

    if "flux-2-klein" in model or "flux-2-dev" in model:
        return {
            "prompt": final_prompt,
            "width": "1344",
            "height": "768",
            "guidance": "4.5",
            "seed": str(seed),
        }, None

    if "flux-1-schnell" in model:
        return {}, {
            "prompt": final_prompt,
            "seed": seed,
            "steps": 8,
        }

    return {}, {
        "prompt": _compact_text(prompt, 1100),
        "negative_prompt": negative,
        "width": 1024,
        "height": 576,
        "num_steps": 8 if "dreamshaper" in model else 20,
        "guidance": 7.5,
        "seed": seed,
    }


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

    def send(form_data: dict[str, str], json_body: dict[str, Any] | None):
        if form_data:
            return requests.post(
                endpoint,
                data=form_data,
                headers=headers,
                timeout=(20, 180),
            )
        return requests.post(
            endpoint,
            json=json_body,
            headers={**headers, "Content-Type": "application/json"},
            timeout=(20, 180),
        )

    form_data, json_body = _cloudflare_request_body(
        model, prompt, negative, seed
    )
    response = send(form_data, json_body)

    # Last-resort retry for Cloudflare prompt-length validation.
    if response.status_code == 400 and "prompt" in response.text.lower() and "2048" in response.text:
        emergency_prompt = _compact_text(prompt, 760)
        print(
            "Cloudflare rejected prompt length; emergency retry active: "
            f"{len(emergency_prompt)} chars"
        )
        if "flux-2-klein" in model or "flux-2-dev" in model:
            form_data = {
                "prompt": emergency_prompt,
                "width": "1344",
                "height": "768",
                "guidance": "4.0",
                "seed": str(seed),
            }
            json_body = None
        elif "flux-1-schnell" in model:
            form_data = {}
            json_body = {
                "prompt": emergency_prompt,
                "seed": seed,
                "steps": 8,
            }
        else:
            form_data = {}
            json_body = {
                "prompt": emergency_prompt,
                "negative_prompt": "text, logo, modern objects, fantasy, distortion",
                "width": 1024,
                "height": 576,
                "num_steps": 8 if "dreamshaper" in model else 20,
                "guidance": 7.0,
                "seed": seed,
            }
        response = send(form_data, json_body)

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


def _fatal_candidate_reason(reason: str, score: int) -> bool:
    normalized = re.sub(r"\s+", " ", str(reason or "")).lower()
    fatal_markers = (
        "tamamen siyah",
        "completely black",
        "boş görüntü",
        "blank image",
        "hiçbir sahne unsuru",
        "geçerli sahne yok",
        "yazı veya logo",
        "text or logo",
        "filigran",
        "watermark",
        "ağır anatomi",
        "severe anatomy",
    )
    return score < 28 or any(marker in normalized for marker in fatal_markers)


def _deterministic_repair_prompt(
    topic: str,
    payload: dict[str, Any],
    scene: dict[str, Any],
    critic_reason: str,
) -> tuple[str, str]:
    world = _compact_text(_world_bible_text(payload), 480)
    goal = _compact_text(scene.get("visual_goal", ""), 330)
    narration = _compact_text(
        scene.get("narration_text") or scene.get("narration_idea", ""),
        260,
    )
    reason = _compact_text(critic_reason, 230)

    prompt = _fit_prompt([
        f"Historically accurate reconstruction for this topic: {topic}",
        f"Show exactly this scene: {goal}",
        f"Narrative context: {narration}",
        f"Historical production bible: {world}",
        f"Correct the previous generation problem: {reason}",
        (
            "One clear cinematic 16:9 documentary frame, physically plausible "
            "period architecture, clothing, tools and geography. No generic "
            "fantasy, no later-era architecture, no modern urban features."
        ),
    ])
    negative = _compact_text(
        (
            f"{combined_negative(scene)}, modern city, medieval town, "
            "renaissance palace, fantasy set, generic catalog composition, "
            "wrong civilization, wrong century"
        ),
        MAX_NEGATIVE_PROMPT_CHARS,
    )
    return prompt, negative


def _critic_guided_repair_prompt(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    scene: dict[str, Any],
    critic_reason: str,
) -> tuple[str, str]:
    world = _compact_text(_world_bible_text(payload), 1500)
    prompt = f"""
Yalnızca geçerli JSON üret:
{{
  "image_prompt": "English prompt",
  "negative_prompt": "English negative prompt"
}}

You are repairing a rejected historical documentary image prompt.

TOPIC:
{topic}

SCENE GOAL:
{scene.get("visual_goal", "")}

SCENE NARRATION:
{scene.get("narration_text") or scene.get("narration_idea", "")}

HISTORICAL WORLD BIBLE:
{world}

CRITIC FEEDBACK ABOUT THE FAILED IMAGE:
{critic_reason}

Write one concise English image prompt that fixes the critic feedback.
It must identify the correct era, location, architecture, materials, clothing,
lighting, camera distance and the exact action or setting.
Do not write a generic ancient city.
Keep image_prompt below 900 characters.
Keep negative_prompt below 260 characters.
No text, logos, modern objects, fantasy, later-era architecture or wrong civilization.
"""
    try:
        repaired, _ = generate_json(client, prompt, max_tokens=1600)
        image_prompt = _fit_prompt([
            str(repaired.get("image_prompt", "")),
            (
                "Photorealistic premium historical documentary frame, "
                "restrained blue-black moonlight and subtle amber practical light, "
                "16:9, coherent with the same film."
            ),
        ])
        negative = _compact_text(
            str(repaired.get("negative_prompt", "")) + ", " + GLOBAL_NEGATIVE,
            MAX_NEGATIVE_PROMPT_CHARS,
        )
        if len(image_prompt) >= 80:
            return image_prompt, negative
    except Exception as exc:
        print("Critic-guided prompt rewrite failed; deterministic repair used:", exc)

    return _deterministic_repair_prompt(
        topic, payload, scene, critic_reason
    )


def generate_scene_image(
    client: genai.Client,
    topic: str,
    payload: dict[str, Any],
    scene: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
    models = cloudflare_model_chain()
    last_error: Exception | None = None
    max_attempts = min(2, len(models))
    best_score = -1
    best_reason = ""
    best_model = ""
    best_seed = 0
    best_file = WORK / f"best-scene-{int(scene['scene_id']):02d}.jpg"
    best_file.unlink(missing_ok=True)

    # First pass: normal scene prompt and model fallbacks.
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

    # Second pass: use the critic's reason to rewrite only this scene prompt.
    repair_reason = best_reason or str(last_error or "The image did not match the scene.")
    repair_prompt, repair_negative = _critic_guided_repair_prompt(
        client, topic, payload, scene, repair_reason
    )
    preferred_models = list(dict.fromkeys([
        "@cf/black-forest-labs/flux-2-klein-4b",
        *models,
    ]))[:1]

    for repair_attempt, model in enumerate(preferred_models, start=1):
        seed = deterministic_seed(
            topic,
            int(scene["scene_id"]),
            100 + repair_attempt,
        )
        try:
            print(
                f"QUALITY RECOVERY scene={scene['scene_id']}, "
                f"model={model}, repair={repair_attempt}/{len(preferred_models)}"
            )
            cloudflare_image_request(
                repair_prompt,
                repair_negative,
                seed,
                target,
                model,
            )
            passed, score, reason = image_review(client, scene, target)
            print(
                f"QUALITY RECOVERY review: pass={passed}; "
                f"score={score}; {reason}"
            )

            if score > best_score:
                shutil.copyfile(target, best_file)
                best_score = score
                best_reason = reason
                best_model = model
                best_seed = seed

            # Repair result receives a slightly lower acceptance threshold,
            # because the critic-guided prompt has already addressed the failure.
            if passed or (
                score >= 58
                and not _fatal_candidate_reason(reason, score)
            ):
                return {
                    "scene_id": scene["scene_id"],
                    "model": model,
                    "seed": seed,
                    "review_score": score,
                    "review": reason,
                    "quality_recovery": True,
                    "file": target.name,
                }
            target.unlink(missing_ok=True)
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            print(f"QUALITY RECOVERY failed ({model}): {exc}")

    # Do not throw away the entire video for a usable but imperfect image.
    if (
        best_file.exists()
        and best_score >= 40
        and not _fatal_candidate_reason(best_reason, best_score)
    ):
        shutil.copyfile(best_file, target)
        print(
            f"QUALITY RECOVERY fallback accepted: scene={scene['scene_id']}, "
            f"score={best_score}"
        )
        return {
            "scene_id": scene["scene_id"],
            "model": best_model,
            "seed": best_seed,
            "review_score": best_score,
            "review": f"En iyi kullanılabilir aday: {best_reason}",
            "quality_fallback": True,
            "needs_manual_review": best_score < 52,
            "file": target.name,
        }

    detail = (
        f"best_score={best_score}; best_reason={best_reason}; "
        f"last_error={last_error}"
    )
    raise RuntimeError(
        f"Sahne {scene['scene_id']} için kullanılabilir görsel üretilemedi: "
        f"{detail}"
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
            "Single cohesive environment, no collage, no split screen, no diptych, no visible seam, no written symbols and no generic Egyptian, Greek or Roman substitution unless the topic requires it."
        )
        negative = (
            f"{GLOBAL_NEGATIVE}, "
            f"{str(payload.get('thumbnail_negative_prompt', '')).strip()}, "
            "words, title, typography, collage, multiple panels, split screen, diptych, divided image, vertical seam, busy left side, generic ancient temple, wrong civilization"
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
    max_attempts = min(2, len(models))
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
Premium historical documentary for relaxed listening.
Natural conversational documentary pace, approximately 138–148 words per minute.
Warm and controlled, but never sleepy, dragged out or lethargic.
Use short natural pauses only where punctuation requires them.
Keep sentences moving forward with clear cause-and-effect emphasis.
Clear articulation without theatrical emphasis.
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
        try:
            print(f"TTS: model={model}, deneme=1/1")
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
            print(f"Gemini TTS atlandı: {model}: {exc}")

    print(
        "Gemini tek parça TTS kullanılamadı; ücretsiz yedek ses "
        f"motoruna geçiliyor. Son hata: {last_error}"
    )
    return synthesize_edge_tts(narration, target)


def normalize_audio(source: Path, target: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-i", str(source),
            "-af",
            "aresample=48000,volume=-6dB,highpass=f=55,lowpass=f=11000,"
            "acompressor=threshold=-23dB:ratio=1.8:attack=28:release=210,"
            "alimiter=limit=0.90,loudnorm=I=-17:TP=-2:LRA=7",
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
            "[1:a]volume=0.18[amb];"
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


def _intro_background(source: Path) -> Image.Image:
    with Image.open(source) as raw:
        image = ImageOps.fit(
            ImageOps.exif_transpose(raw).convert("RGB"),
            (WIDTH, HEIGHT),
            Image.Resampling.LANCZOS,
        )
    image = ImageEnhance.Brightness(image).enhance(0.62)
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = ImageEnhance.Color(image).enhance(0.82)
    return image


def _left_cinematic_gradient(image: Image.Image, strength: int = 220) -> Image.Image:
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for x in range(1180):
        alpha = int(strength * (1 - x / 1180) ** 1.55)
        draw.line((x, 0, x, HEIGHT), fill=(5, 8, 14, alpha))
    draw.rectangle((0, 0, WIDTH, HEIGHT), fill=(4, 7, 12, 22))
    return Image.alpha_composite(base, overlay)


def make_intro_frames(
    hero_frame: Path,
    second_frame: Path,
    clean_target: Path,
    brand_target: Path,
    title_target: Path,
    exit_target: Path,
    video_title: str,
    intro_hook: str,
) -> None:
    hero = _intro_background(hero_frame)
    second = _intro_background(second_frame)

    clean = ImageEnhance.Brightness(hero).enhance(0.86)
    clean.save(clean_target, "JPEG", quality=95, optimize=True)

    brand = hero.convert("RGBA")
    brand = Image.alpha_composite(
        brand,
        Image.new("RGBA", brand.size, (0, 0, 0, 70)),
    )
    brand_draw = ImageDraw.Draw(brand)
    brand_font = video_font(54, bold=True)
    sub_font = video_font(21, bold=True)
    brand_text = "UYKU VE TARİH"
    brand_box = brand_draw.textbbox((0, 0), brand_text, font=brand_font)
    brand_width = brand_box[2] - brand_box[0]
    brand_draw.rectangle(
        (WIDTH // 2 - 85, 428, WIDTH // 2 + 85, 436),
        fill=(207, 174, 122, 255),
    )
    brand_draw.text(
        ((WIDTH - brand_width) // 2, 466),
        brand_text,
        font=brand_font,
        fill=(244, 238, 225),
        stroke_width=1,
        stroke_fill=(5, 7, 10),
    )
    sub_text = "GECE BELGESELİ"
    sub_box = brand_draw.textbbox((0, 0), sub_text, font=sub_font)
    brand_draw.text(
        ((WIDTH - (sub_box[2] - sub_box[0])) // 2, 548),
        sub_text,
        font=sub_font,
        fill=(202, 184, 154),
    )
    brand.convert("RGB").save(brand_target, "JPEG", quality=95, optimize=True)

    title_image = _left_cinematic_gradient(second)
    title_draw = ImageDraw.Draw(title_image)
    small_font = video_font(23, bold=True)
    title_font = video_font(58, bold=True)
    hook_font = video_font(27, bold=False)

    title = re.sub(r"\\s+", " ", video_title).strip().upper()
    title = textwrap.shorten(title, width=70, placeholder="…").upper()
    title_lines: list[str] = []
    current = ""
    for word in title.split():
        candidate = f"{current} {word}".strip()
        width = title_draw.textbbox((0, 0), candidate, font=title_font)[2]
        if width <= 1000:
            current = candidate
        else:
            if current:
                title_lines.append(current)
            current = word
    if current:
        title_lines.append(current)

    title_draw.text(
        (120, 320),
        "UYKU VE TARİH  /  YENİ BÖLÜM",
        font=small_font,
        fill=(210, 184, 142),
    )
    title_draw.rectangle(
        (120, 372, 320, 380),
        fill=(210, 178, 126, 255),
    )
    for line_index, line in enumerate(title_lines[:3]):
        title_draw.text(
            (120, 420 + line_index * 73),
            line,
            font=title_font,
            fill=(246, 240, 228),
            stroke_width=2,
            stroke_fill=(5, 7, 10),
        )

    hook = re.sub(r"\\s+", " ", intro_hook).strip()
    if hook:
        title_draw.text(
            (123, 666),
            hook,
            font=hook_font,
            fill=(207, 199, 184),
        )
    title_image.convert("RGB").save(
        title_target, "JPEG", quality=95, optimize=True
    )

    exit_image = hero.filter(ImageFilter.GaussianBlur(radius=1.1))
    exit_image = ImageEnhance.Brightness(exit_image).enhance(0.82)
    exit_image.save(exit_target, "JPEG", quality=95, optimize=True)


def render_intro_sequence(
    clean_frame: Path,
    brand_frame: Path,
    title_frame: Path,
    exit_frame: Path,
    target: Path,
) -> None:
    # 8-second editorial cold open. The brand is present but never stalls
    # the story with a long ident.
    durations = [1.5, 1.7, 3.7, 2.3]
    transition = 0.36

    command = ["ffmpeg", "-y"]
    for duration, frame in zip(
        durations,
        [
            clean_frame,
            brand_frame,
            title_frame,
            exit_frame,
        ],
    ):
        command += [
            "-loop", "1",
            "-framerate", str(FPS),
            "-t", f"{duration:.3f}",
            "-i", str(frame),
        ]

    filters = (
        "[0:v]scale=1920:1080,setsar=1,"
        "fade=t=in:st=0:d=0.45,"
        "format=yuv420p[v0];"
        "[1:v]scale=1920:1080,setsar=1,"
        "format=yuv420p[v1];"
        "[2:v]scale=1920:1080,setsar=1,"
        "format=yuv420p[v2];"
        "[3:v]scale=1920:1080,setsar=1,"
        "format=yuv420p[v3];"
        f"[v0][v1]xfade=transition=fade:duration={transition}:"
        "offset=1.140[x1];"
        f"[x1][v2]xfade=transition=fade:duration={transition}:"
        "offset=2.480[x2];"
        f"[x2][v3]xfade=transition=fade:duration={transition}:"
        "offset=5.820,"
        "fade=t=out:st=7.55:d=0.45,"
        "format=yuv420p[v]"
    )

    command += [
        "-filter_complex", filters,
        "-map", "[v]",
        "-t", f"{INTRO_VISIBLE_SECONDS:.3f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        str(target),
    ]
    run(command)


def create_intro_audio(
    target: Path,
    duration: float = INTRO_VISIBLE_SECONDS,
) -> None:
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i",
        f"anoisesrc=color=pink:amplitude=0.018:sample_rate=48000:d={duration:.3f}",
        "-f", "lavfi", "-i",
        f"sine=frequency=48:sample_rate=48000:duration={duration:.3f}",
        "-f", "lavfi", "-i",
        "anoisesrc=color=white:amplitude=0.08:sample_rate=48000:d=1.5",
        "-f", "lavfi", "-i",
        "sine=frequency=196:sample_rate=48000:duration=2.8",
        "-f", "lavfi", "-i",
        "anoisesrc=color=pink:amplitude=0.07:sample_rate=48000:d=1.2",
        "-filter_complex",
        "[0:a]highpass=f=70,lowpass=f=820,volume=0.035,"
        "afade=t=in:st=0:d=1.4,afade=t=out:st=10.0:d=2.0[air];"
        "[1:a]lowpass=f=90,volume=0.022,"
        "afade=t=in:st=0:d=2.0,afade=t=out:st=10.0:d=2.0[rumble];"
        "[2:a]highpass=f=700,lowpass=f=6500,"
        "afade=t=in:st=0:d=0.15,afade=t=out:st=0.45:d=0.95,"
        "adelay=2650|2650,volume=0.035[whoosh1];"
        "[3:a]afade=t=in:st=0:d=0.08,"
        "afade=t=out:st=0.8:d=2.0,"
        "aecho=0.5:0.25:230|460:0.18|0.09,"
        "adelay=5200|5200,volume=0.030[chime];"
        "[4:a]highpass=f=900,lowpass=f=7500,"
        "afade=t=in:st=0:d=0.12,afade=t=out:st=0.35:d=0.75,"
        "adelay=8650|8650,volume=0.030[whoosh2];"
        "[air][rumble][whoosh1][chime][whoosh2]"
        "amix=inputs=5:duration=longest:normalize=0,"
        "alimiter=limit=0.88[a]",
        "-map", "[a]",
        "-t", f"{duration:.3f}",
        "-ar", "48000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(target),
    ])


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

def _chapter_label_frame(
    image: Image.Image,
    chapter_index: int,
    chapter_title: str,
) -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new(
        "RGBA",
        canvas.size,
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle(
        (84, 82, 720, 184),
        radius=14,
        fill=(5, 8, 12, 158),
    )
    draw.rectangle(
        (112, 108, 188, 114),
        fill=(208, 176, 126, 255),
    )
    draw.text(
        (112, 130),
        "UYKU VE TARİH",
        font=video_font(17, bold=True),
        fill=(202, 182, 150),
    )
    title = textwrap.shorten(
        re.sub(
            r"\s+",
            " ",
            chapter_title,
        ).strip().upper(),
        width=38,
        placeholder="…",
    )
    draw.text(
        (292, 122),
        title,
        font=video_font(27, bold=True),
        fill=(243, 237, 225),
    )
    return Image.alpha_composite(
        canvas,
        overlay,
    ).convert("RGB")


def _scene_shot_frames(
    frame: Path,
    scene: dict[str, Any],
    index: int,
    target_dir: Path,
    chapter_start: bool,
) -> list[Path]:
    with Image.open(frame) as raw:
        base = ImageOps.exif_transpose(raw).convert("RGB")

    wide = ImageOps.fit(
        base,
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    medium = ImageOps.fit(
        base,
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS,
        centering=(0.40, 0.50) if index % 2 else (0.60, 0.50),
    )
    medium = ImageEnhance.Contrast(medium).enhance(1.03)

    crop_w = int(base.width * 0.72)
    crop_h = int(base.height * 0.72)
    center_x = int(base.width * (0.38 if index % 2 else 0.62))
    center_y = int(base.height * 0.52)
    left = max(0, min(base.width - crop_w, center_x - crop_w // 2))
    top = max(0, min(base.height - crop_h, center_y - crop_h // 2))
    detail = base.crop((left, top, left + crop_w, top + crop_h))
    detail = ImageOps.fit(
        detail,
        (WIDTH, HEIGHT),
        Image.Resampling.LANCZOS,
    )
    detail = ImageEnhance.Brightness(detail).enhance(0.97)
    detail = ImageEnhance.Contrast(detail).enhance(1.05)

    if chapter_start:
        wide = _chapter_label_frame(
            wide,
            int(scene.get("chapter_index", 1)),
            str(scene.get("chapter_title", "")),
        )

    paths = [
        target_dir / f"scene_{index:02d}_wide.jpg",
        target_dir / f"scene_{index:02d}_medium.jpg",
        target_dir / f"scene_{index:02d}_detail.jpg",
    ]
    for image, path in zip([wide, medium, detail], paths):
        image.save(path, "JPEG", quality=94, optimize=True)
    return paths


def _render_scene_editorial_clip(
    shot_frames: list[Path],
    seconds: float,
    chapter_start: bool,
    chapter_end: bool,
    target: Path,
) -> None:
    seconds = max(9.0, float(seconds))

    # Professional documentary rhythm: mostly clean cuts.
    # No blur transition, slide, zoom or fake camera shake.
    visible_parts = [
        seconds * 0.48,
        seconds * 0.31,
    ]
    visible_parts.append(
        seconds - sum(visible_parts)
    )

    command = ["ffmpeg", "-y"]
    for frame, duration in zip(
        shot_frames,
        visible_parts,
    ):
        command += [
            "-loop", "1",
            "-framerate", str(FPS),
            "-t", f"{duration:.3f}",
            "-i", str(frame),
        ]

    filters = []
    labels = []
    for index in range(3):
        label = f"v{index}"
        filters.append(
            f"[{index}:v]"
            "scale=1920:1080,setsar=1,"
            "eq=saturation=0.92:contrast=1.02:brightness=-0.008,"
            "vignette=PI/10.2,"
            f"format=yuv420p[{label}]"
        )
        labels.append(f"[{label}]")

    filters.append(
        "".join(labels)
        + "concat=n=3:v=1:a=0[cut]"
    )

    final_filters = []
    if chapter_start:
        final_filters.append(
            "fade=t=in:st=0:d=0.42"
        )
    if chapter_end:
        final_filters.append(
            f"fade=t=out:st={max(0.0, seconds - 0.46):.3f}:d=0.46"
        )

    if final_filters:
        filters.append(
            "[cut]"
            + ",".join(final_filters)
            + ",format=yuv420p[v]"
        )
    else:
        filters.append(
            "[cut]format=yuv420p[v]"
        )

    command += [
        "-filter_complex",
        ";".join(filters),
        "-map", "[v]",
        "-t", f"{seconds:.3f}",
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        str(target),
    ]
    run(command)


def _verify_final_media(
    target: Path,
    expected_seconds: float,
) -> float:
    actual = ffprobe_duration(target)
    if actual < MIN_FINAL_VIDEO_SECONDS:
        raise RuntimeError(
            f"Final video çok kısa üretildi: {actual:.2f} saniye. "
            f"Hedef yaklaşık {expected_seconds:.2f} saniyeydi."
        )
    if actual > MAX_FINAL_VIDEO_SECONDS + 15:
        raise RuntimeError(
            f"Final video beklenenden uzun: {actual:.2f} saniye."
        )
    if abs(actual - expected_seconds) > 3.0:
        raise RuntimeError(
            "Final süre güvenlik kontrolünü geçemedi: "
            f"{actual:.2f}s / {expected_seconds:.2f}s"
        )
    return actual


def render_video(
    frames: list[Path],
    scenes: list[dict[str, Any]],
    audio: Path,
    target: Path,
    visible_durations: list[float],
    transitions: list[str],
    transition_durations: list[float],
) -> float:
    clips_dir = WORK / "v7-clips"
    shots_dir = WORK / "v7-shots"
    for folder in (clips_dir, shots_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    intro_clean = WORK / "intro-clean.jpg"
    intro_brand = WORK / "intro-brand.jpg"
    intro_title = WORK / "intro-title.jpg"
    intro_exit = WORK / "intro-exit.jpg"

    second_frame = frames[1] if len(frames) > 1 else frames[0]
    package = json.loads(
        (OUTPUT / "video-paketi.json").read_text(encoding="utf-8")
    )
    make_intro_frames(
        frames[0],
        second_frame,
        intro_clean,
        intro_brand,
        intro_title,
        intro_exit,
        (OUTPUT / "baslik.txt").read_text(encoding="utf-8"),
        str(package.get("intro_hook", "")),
    )

    intro_clip = clips_dir / "000_intro.mp4"
    render_intro_sequence(
        intro_clean,
        intro_brand,
        intro_title,
        intro_exit,
        intro_clip,
    )

    scene_clips: list[Path] = []
    timeline: list[dict[str, Any]] = [{
        "type": "intro",
        "start": 0.0,
        "duration": INTRO_VISIBLE_SECONDS,
        "style": "cold-open / ident / title reveal / story handoff",
    }]
    cursor = INTRO_VISIBLE_SECONDS

    for index, (frame, scene, duration) in enumerate(
        zip(frames, scenes, visible_durations),
        start=1,
    ):
        previous_chapter = (
            int(scenes[index - 2].get("chapter_index", 1))
            if index > 1 else None
        )
        current_chapter = int(scene.get("chapter_index", 1))
        next_chapter = (
            int(scenes[index].get("chapter_index", current_chapter))
            if index < len(scenes) else None
        )
        chapter_start = previous_chapter != current_chapter
        chapter_end = next_chapter != current_chapter

        shot_frames = _scene_shot_frames(
            frame,
            scene,
            index,
            shots_dir,
            chapter_start,
        )
        scene_clip = clips_dir / f"{index:03d}_scene.mp4"
        _render_scene_editorial_clip(
            shot_frames,
            duration,
            chapter_start,
            chapter_end,
            scene_clip,
        )
        scene_clips.append(scene_clip)

        timeline.append({
            "type": "story_scene",
            "scene_id": scene.get("scene_id", index),
            "chapter_index": current_chapter,
            "chapter_title": scene.get("chapter_title", ""),
            "beat_type": scene.get("beat_type", ""),
            "continuity_bridge": scene.get("continuity_bridge", ""),
            "start": round(cursor, 3),
            "duration": round(duration, 3),
            "shots": ["establishing", "action-detail", "consequence"],
            "internal_transitions": ["cut", "cut"],
            "narration_text": scene.get("narration_text", ""),
        })
        cursor += duration

    concat_file = WORK / "v7-video-concat.txt"
    all_clips = [intro_clip, *scene_clips]
    write_concat_manifest(all_clips, concat_file)

    video_only = WORK / "v7-video-only.mp4"
    run([
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "copy",
        "-an",
        str(video_only),
    ])

    expected_duration = INTRO_VISIBLE_SECONDS + sum(visible_durations)
    audio_duration = ffprobe_duration(audio)
    video_duration = ffprobe_duration(video_only)
    if abs(audio_duration - expected_duration) > 1.2:
        raise RuntimeError(
            "Final ses süresi hedefle uyuşmuyor: "
            f"{audio_duration:.2f}s / {expected_duration:.2f}s"
        )
    if abs(video_duration - expected_duration) > 2.0:
        raise RuntimeError(
            "Final görüntü süresi hedefle uyuşmuyor: "
            f"{video_duration:.2f}s / {expected_duration:.2f}s"
        )

    run([
        "ffmpeg", "-y",
        "-i", str(video_only),
        "-i", str(audio),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-t", f"{expected_duration:.3f}",
        "-movflags", "+faststart",
        str(target),
    ])

    (OUTPUT / "edit-timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _verify_final_media(target, expected_duration)


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
    (OUTPUT / "HATA-V8-2.txt").write_text(
        f"{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )


def run_internal_preflight() -> None:
    if CHAPTER_COUNT != len(CHAPTER_BEATS):
        raise RuntimeError(
            "İç yapı hatası: CHAPTER_COUNT ile CHAPTER_BEATS eşleşmiyor."
        )

    if CHAPTER_COUNT < 1:
        raise RuntimeError("İç yapı hatası: en az bir perde gerekli.")

    # Reproduce the exact V8 crash condition deliberately. The safe clamp
    # must handle an out-of-range chapter without raising IndexError.
    probe = _local_expand_chapter(
        "Kararlar birbirini izledi.",
        "Ön kontrol konusu",
        CHAPTER_COUNT + 1,
        55,
    )
    if _word_count(probe) < 20:
        raise RuntimeError(
            "İç yapı hatası: yerel hikâye tamamlayıcı çalışmıyor."
        )

    test_scene_count = 12
    assigned = [
        min(
            CHAPTER_COUNT,
            1 + index * CHAPTER_COUNT // test_scene_count,
        )
        for index in range(test_scene_count)
    ]
    if min(assigned) < 1 or max(assigned) > CHAPTER_COUNT:
        raise RuntimeError(
            "İç yapı hatası: sahne-perde dağılımı sınır dışı."
        )

    print(
        "INTERNAL PREFLIGHT OK: "
        f"{CHAPTER_COUNT} perde, güvenli indeks ve yerel fallback doğrulandı."
    )


def main() -> None:
    run_internal_preflight()
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

    requested_target = int(
        os.getenv("TARGET_SECONDS", str(DEFAULT_TARGET_SECONDS))
    )
    target_seconds = (
        DEFAULT_TARGET_SECONDS
        if requested_target < 240
        else max(240, min(360, requested_target))
    )

    requested_scene_count = int(
        os.getenv("SCENE_COUNT", str(DEFAULT_SCENE_COUNT))
    )
    scene_count = (
        DEFAULT_SCENE_COUNT
        if requested_scene_count < 10
        else max(10, min(16, requested_scene_count))
    )
    story_seconds = target_seconds - INTRO_VISIBLE_SECONDS
    client = genai.Client(api_key=gemini_key)

    print("=" * 72)
    print("UYKU VE TARİH V8.2 — NATURAL VOICE CUT ACTIVE")
    print("Konu:", topic)
    print("NATURAL VOICE: full script → one narrator → no forced slowdown")
    print("=" * 72)

    payload, text_model = build_video_package(
        client, topic, int(story_seconds), scene_count
    )
    story_model = "disabled-for-reliability"
    if os.getenv("ENABLE_STORY_REFINEMENT", "0") == "1":
        payload, story_model = story_director_pass(
            client, topic, payload, scene_count
        )

    (OUTPUT / "video-paketi.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "hikaye-arki.json").write_text(
        json.dumps({
            "story_arc": payload.get("story_arc", {}),
            "scenes": [{
                "scene_id": scene.get("scene_id"),
                "act": scene.get("act"),
                "beat_type": scene.get("beat_type"),
                "continuity_bridge": scene.get("continuity_bridge"),
                "narration_text": scene.get("narration_text"),
            } for scene in payload.get("scenes", [])],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "senaryo.txt").write_text(payload["narration"], encoding="utf-8")
    (OUTPUT / "baslik.txt").write_text(str(payload["video_title"]), encoding="utf-8")

    print("AŞAMA 1/4: Tek anlatıcıyla tam metin seslendiriliyor.")
    narration_audio = OUTPUT / "seslendirme.wav"
    raw_audio = WORK / "narration-single-voice-raw.wav"
    normalized_audio = WORK / "narration-single-voice-normalized.wav"

    # The complete script is synthesized in one request and by one engine.
    # A model change can only happen before generation begins, never mid-video.
    tts_model = synthesize_narration(
        client,
        payload["narration"],
        raw_audio,
    )
    normalize_audio(raw_audio, normalized_audio)
    actual_story_seconds, narration_tempo = prepare_natural_narration(
        normalized_audio,
        narration_audio,
        story_seconds,
    )
    visible_durations = _allocate_exact_duration(
        actual_story_seconds,
        [
            max(
                1,
                _word_count(
                    str(scene.get("narration_text", ""))
                ),
            )
            for scene in payload["scenes"]
        ],
        minimum=12.0,
    )

    print("AŞAMA 2/4: Ses hazır. Görseller üretiliyor.")
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
        frames,
        payload["scenes"],
        OUTPUT / "storyboard-kontrol.jpg",
    )

    thumb_dir = WORK / "thumbnail-candidates"
    selected_thumb_background, thumb_info = generate_thumbnail_candidates(
        client,
        topic,
        payload,
        thumb_dir,
    )
    make_thumbnail(
        selected_thumb_background,
        str(payload["thumbnail_text"]),
        OUTPUT / "kapak.jpg",
    )

    print("AŞAMA 3/4: Ses tasarımı ve zaman çizelgesi hazırlanıyor.")
    transitions, transition_durations = scene_transition_plan(payload["scenes"])

    ambient_track = WORK / "ambient-track.wav"
    build_ambient_track(
        payload["scenes"], visible_durations, transition_durations, ambient_track
    )
    story_audio = WORK / "story-audio.wav"
    mix_narration_and_ambient(narration_audio, ambient_track, story_audio)

    intro_audio = WORK / "intro-audio.wav"
    create_intro_audio(intro_audio, INTRO_VISIBLE_SECONDS)

    final_audio_raw = WORK / "final-audio-raw.wav"
    final_audio = OUTPUT / "ses-tasarim.wav"
    concat_audio_files([intro_audio, story_audio], final_audio_raw)
    shutil.copy2(final_audio_raw, final_audio)
    expected_total_seconds = INTRO_VISIBLE_SECONDS + actual_story_seconds
    final_audio_seconds = ffprobe_duration(final_audio)
    if abs(final_audio_seconds - expected_total_seconds) > 1.2:
        raise RuntimeError(
            "Final doğal ses süresi kontrolünü geçemedi: "
            f"{final_audio_seconds:.2f}s / {expected_total_seconds:.2f}s"
        )

    print("AŞAMA 4/4: Final video render ediliyor.")
    video = OUTPUT / "uyku-tarih-v8-2-natural-voice.mp4"
    actual_duration = render_video(
        frames, payload["scenes"], final_audio, video,
        visible_durations, transitions, transition_durations,
    )
    create_scene_srt(
        payload["scenes"], visible_durations,
        INTRO_VISIBLE_SECONDS, OUTPUT / "altyazi.srt",
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
        "story_director_model": story_model,
        "tts_model": tts_model,
        "narration_tempo_factor": round(narration_tempo, 4),
        "natural_story_duration_seconds": round(actual_story_seconds, 2),
        "image_engine": "Cloudflare Workers AI",
        "image_model_default": os.getenv(
            "CLOUDFLARE_IMAGE_MODEL",
            "@cf/black-forest-labs/flux-2-klein-4b",
        ),
        "actual_duration_seconds": round(actual_duration, 2),
        "scene_count": len(frames),
        "story_director": {
            "true_intro_in_active_render_path": True,
            "three_act_story": True,
            "scene_level_tts_sync": True,
            "critic_guided_image_recovery": True,
            "finite_audio_padding": True,
            "zero_pause_direct_copy": True,
            "ffmpeg_timeout_seconds": 900,
            "fast_retry_chain": True,
            "five_minute_target": True,
            "fail_soft_story_generation": True,
            "chapter_by_chapter_generation": True,
            "deterministic_local_length_guard": True,
            "short_script_never_aborts": True,
            "verified_audio_concat_filter": True,
            "tts_before_images": True,
            "quota_safe_tts_requests": True,
            "edge_tts_fallback": True,
            "concat_manifest_newline_validation": True,
            "chapter_tts_calls": 0,
            "single_full_script_tts_call": True,
            "voice_switching_inside_video": False,
            "forced_speech_slowdown": False,
            "maximum_voice_slowdown_percent": 2,
            "maximum_voice_speedup_percent": 8,
            "video_duration_follows_voice": True,
            "edge_tts_rate": "+8%",
            "target_narration_wpm": "138-148",
            "description_ratio_cap_percent": 15,
            "cause_effect_story_structure": True,
            "continuity_editor": True,
            "chapter_index_clamp": True,
            "hardcoded_chapter_four_removed": True,
            "internal_preflight_before_api": True,
            "exact_v8_indexerror_regression_test": True,
            "internal_blur_transitions": False,
            "editorial_shots_per_scene": 3,
            "professional_intro_seconds": INTRO_VISIBLE_SECONDS,
            "hard_final_duration_guard": True,
            "usable_best_candidate_fallback": True,
            "wide_detail_editorial_cuts": True,
        },
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
        "transition_style": "clean cuts with restrained act-break fades",
    }
    (OUTPUT / "uretim-raporu.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUTPUT / "ONCE-BUNU-OKU.txt").write_text(
        (
            "V8.2 Natural Voice Cut yalnızca konu girdisiyle üretildi.\n\n"
            "Önce uyku-tarih-v8-2-natural-voice.mp4 dosyasını izle.\n"
            "kapak.jpg dosyasını mobil boyutta kontrol et.\n"
            "storyboard-kontrol.jpg yalnızca kalite kontrol dosyasıdır; "
            "üzerindeki açıklamalar final videoda bulunmaz.\n"
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("V8.2 NATURAL VOICE CUT TAMAMLANDI")
    print("Video:", video)
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        failure_file(exc)
        raise
