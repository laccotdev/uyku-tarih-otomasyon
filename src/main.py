from __future__ import annotations

import json
import os
import shutil
import sys
import traceback
from pathlib import Path

from commons_media import credits_text, download_visuals
from content import generate_payload
from render import (
    chapter_lines,
    create_fallback_source,
    create_srt,
    create_thumbnail,
    prepare_scene,
    render_video,
    synthesize_speech,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
WORK = ROOT / "work"
VOICES = ROOT / "voices"


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def reset_directories() -> None:
    for path in (OUTPUT, WORK):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    VOICES.mkdir(parents=True, exist_ok=True)


def write_failure(exc: BaseException) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    details = "".join(traceback.format_exception(exc))
    (OUTPUT / "HATA.txt").write_text(
        "Video üretimi tamamlanamadı.\n\n" + details,
        encoding="utf-8",
    )


def main() -> None:
    reset_directories()

    topic = os.getenv("VIDEO_TOPIC", "MÖ 1200'de Hattuşa'nın son gecesi").strip()
    duration_minutes = bounded_int("DURATION_MINUTES", 2, 1, 15)
    visual_count = bounded_int("VISUAL_COUNT", 8, 4, 24)
    test_mode = env_bool("TEST_MODE", True)
    voice_name = os.getenv("PIPER_VOICE", "tr_TR-dfki-medium").strip()

    print("=" * 64)
    print("UYKU VE TARİH OTOMASYONU")
    print(f"Konu: {topic}")
    print(f"Hedef süre: {duration_minutes} dakika")
    print(f"Görsel hedefi: {visual_count}")
    print(f"Test modu: {test_mode}")
    print("=" * 64)

    payload = generate_payload(topic, duration_minutes, test_mode)
    narration = payload["narration"].strip()

    (OUTPUT / "metadata.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "senaryo.txt").write_text(narration, encoding="utf-8")
    (OUTPUT / "baslik.txt").write_text(payload["title"], encoding="utf-8")

    source_dir = WORK / "sources"
    source_paths, credits = download_visuals(
        payload["search_queries"],
        visual_count,
        source_dir,
    )

    while len(source_paths) < visual_count:
        fallback = source_dir / f"fallback_{len(source_paths) + 1:03d}.jpg"
        create_fallback_source(fallback, topic, len(source_paths) + 1)
        source_paths.append(fallback)

    scene_dir = WORK / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    scenes: list[Path] = []
    for index, source in enumerate(source_paths[:visual_count], start=1):
        target = scene_dir / f"scene_{index:03d}.jpg"
        try:
            prepare_scene(source, target)
        except Exception as exc:
            print(f"Sahne hazırlanamadı, yedek oluşturuluyor: {exc}")
            create_fallback_source(target, topic, index)
        scenes.append(target)

    thumbnail_path = OUTPUT / "kapak.jpg"
    create_thumbnail(scenes[0], thumbnail_path, payload["thumbnail_text"])

    audio_path = OUTPUT / "seslendirme.wav"
    synthesize_speech(
        OUTPUT / "senaryo.txt",
        audio_path,
        VOICES,
        voice_name,
    )

    video_path = OUTPUT / "pilot-video.mp4"
    actual_duration = render_video(
        scenes,
        audio_path,
        video_path,
        WORK,
    )

    create_srt(narration, actual_duration, OUTPUT / "altyazi.srt")
    credit_block = credits_text(credits)
    (OUTPUT / "gorsel-kaynaklari.txt").write_text(credit_block, encoding="utf-8")

    chapters = chapter_lines(payload.get("chapters", []), actual_duration)
    tags = ", ".join(payload.get("tags", []))
    description_parts = [
        payload["description"].strip(),
        "",
        "BÖLÜMLER",
        *chapters,
        "",
        credit_block,
    ]
    if tags:
        description_parts.extend(["", f"Etiketler: {tags}"])

    (OUTPUT / "youtube-aciklamasi.txt").write_text(
        "\n".join(description_parts).strip() + "\n",
        encoding="utf-8",
    )

    summary = {
        "topic": topic,
        "test_mode": test_mode,
        "requested_minutes": duration_minutes,
        "actual_seconds": round(actual_duration, 2),
        "visual_count": len(scenes),
        "licensed_commons_visuals": len(credits),
        "video": video_path.name,
        "thumbnail": thumbnail_path.name,
    }
    (OUTPUT / "uretim-ozeti.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 64)
    print("ÜRETİM TAMAMLANDI")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        write_failure(exc)
        print("Üretim başarısız oldu:", exc, file=sys.stderr)
        raise
