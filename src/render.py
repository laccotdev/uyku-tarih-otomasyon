from __future__ import annotations

import math
import re
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

WIDTH = 1920
HEIGHT = 1080
FPS = 25


def run(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(x) for x in command)
    print(f"Çalıştırılıyor: {printable}")
    subprocess.run(command, cwd=cwd, check=True)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def create_fallback_source(path: Path, topic: str, index: int) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (32, 29, 25))
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        shade = 22 + int((y / HEIGHT) * 22)
        draw.line((0, y, WIDTH, y), fill=(shade + 10, shade + 7, shade + 3))
    font = _font(76)
    small = _font(30)
    wrapped = textwrap.fill(topic.upper(), width=28)
    draw.multiline_text((120, 350), wrapped, font=font, fill=(232, 222, 199), spacing=14)
    draw.text((125, 740), f"TARİH YOLCULUĞU · {index:02d}", font=small, fill=(178, 164, 138))
    image.save(path, quality=92)


def prepare_scene(source: Path, target: Path) -> None:
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")

    background = ImageOps.fit(image, (WIDTH, HEIGHT), method=Image.Resampling.LANCZOS)
    background = background.filter(ImageFilter.GaussianBlur(24))
    background = ImageEnhance.Brightness(background).enhance(0.46)

    foreground = ImageOps.contain(image, (1720, 940), method=Image.Resampling.LANCZOS)
    canvas = background.copy()
    x = (WIDTH - foreground.width) // 2
    y = (HEIGHT - foreground.height) // 2
    canvas.paste(foreground, (x, y))

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 28))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    canvas.save(target, "JPEG", quality=91, optimize=True)


def create_thumbnail(scene: Path, target: Path, text: str) -> None:
    with Image.open(scene) as raw:
        image = ImageOps.fit(raw.convert("RGB"), (1280, 720), Image.Resampling.LANCZOS)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle((0, 0, 790, 720), fill=(0, 0, 0, 155))
    image = Image.alpha_composite(image.convert("RGBA"), overlay)

    draw = ImageDraw.Draw(image)
    clean = re.sub(r"\s+", " ", text).strip().upper()
    wrapped = textwrap.fill(clean, width=14)
    font = _font(88)
    draw.multiline_text(
        (82, 185),
        wrapped,
        font=font,
        fill=(244, 234, 210),
        spacing=8,
        stroke_width=2,
        stroke_fill=(18, 16, 13),
    )
    draw.text(
        (88, 596),
        "UYKU İÇİN TARİH",
        font=_font(28),
        fill=(196, 179, 145),
    )
    image.convert("RGB").save(target, "JPEG", quality=94, optimize=True)


def synthesize_speech(
    narration_file: Path,
    output_file: Path,
    voice_dir: Path,
    voice_name: str,
) -> None:
    raw_file = output_file.with_name("narration_raw.wav")
    piper_command = [
        sys.executable,
        "-m",
        "piper",
        "-m",
        voice_name,
        "--data-dir",
        str(voice_dir),
        "-f",
        str(raw_file),
        "--sentence-silence",
        "0.55",
        "--input-file",
        str(narration_file),
    ]

    try:
        run(piper_command)
    except Exception as exc:
        print(f"Piper çalışmadı, espeak-ng yedeğine geçiliyor: {exc}")
        run(
            [
                "espeak-ng",
                "-v",
                "tr",
                "-s",
                "128",
                "-f",
                str(narration_file),
                "-w",
                str(raw_file),
            ]
        )

    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(raw_file),
            "-filter:a",
            "atempo=0.92,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar",
            "48000",
            str(output_file),
        ]
    )


def audio_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return max(1.0, float(result.stdout.strip()))


def render_video(
    scenes: list[Path],
    narration: Path,
    output_video: Path,
    work_dir: Path,
) -> float:
    duration = audio_duration(narration)
    segment_duration = duration / max(1, len(scenes))
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    for index, scene in enumerate(scenes, start=1):
        frames = max(1, math.ceil(segment_duration * FPS))
        clip = clips_dir / f"clip_{index:03d}.mp4"
        zoom_step = 0.00018 if index % 2 else 0.00012
        vf = (
            f"zoompan=z='min(zoom+{zoom_step},1.08)':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},"
            "format=yuv420p"
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-loop",
                "1",
                "-i",
                str(scene),
                "-vf",
                vf,
                "-frames:v",
                str(frames),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "24",
                "-pix_fmt",
                "yuv420p",
                str(clip),
            ]
        )
        clips.append(clip)

    concat_file = work_dir / "clips.txt"
    concat_file.write_text(
        "\n".join(f"file '{clip.resolve().as_posix()}'" for clip in clips),
        encoding="utf-8",
    )

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(narration),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
    )
    return duration


def _timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def create_srt(narration: str, duration: float, target: Path) -> None:
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", narration.strip())
        if s.strip()
    ]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= 150:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)

    if not chunks:
        chunks = [narration.strip()]

    weights = [max(1, len(chunk)) for chunk in chunks]
    total_weight = sum(weights)
    cursor = 0.0
    blocks = []

    for index, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
        chunk_duration = duration * (weight / total_weight)
        start = cursor
        end = min(duration, cursor + chunk_duration)
        blocks.append(
            f"{index}\n{_timestamp(start)} --> {_timestamp(end)}\n{chunk}\n"
        )
        cursor = end

    target.write_text("\n".join(blocks), encoding="utf-8")


def chapter_lines(chapters: list[str], duration: float) -> list[str]:
    if not chapters:
        return []
    spacing = duration / len(chapters)
    lines = []
    for index, title in enumerate(chapters):
        seconds = int(index * spacing)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            stamp = f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            stamp = f"{minutes:02d}:{secs:02d}"
        lines.append(f"{stamp} {title}")
    return lines
