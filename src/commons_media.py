from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "UykuTarihAutomation/1.0 (educational-video-generator)"
ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp"}
LICENSE_MARKERS = (
    "public domain",
    "cc0",
    "cc by ",
    "cc by-sa",
    "creative commons attribution",
)


def _plain(value: str | None) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _meta_value(metadata: dict[str, Any], key: str) -> str:
    item = metadata.get(key, {})
    if isinstance(item, dict):
        return _plain(str(item.get("value", "")))
    return _plain(str(item))


def _license_allowed(name: str) -> bool:
    lower = name.lower()
    if "noncommercial" in lower or "no derivatives" in lower:
        return False
    return any(marker in lower for marker in LICENSE_MARKERS)


def search_commons(query: str, limit: int = 12) -> list[dict[str, str]]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url|mime|extmetadata",
        "iiurlwidth": 1920,
        "format": "json",
        "formatversion": 2,
    }
    response = requests.get(
        API_URL,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])

    results: list[dict[str, str]] = []
    for page in pages:
        info_list = page.get("imageinfo") or []
        if not info_list:
            continue
        info = info_list[0]
        if info.get("mime") not in ALLOWED_MIMES:
            continue

        meta = info.get("extmetadata") or {}
        license_name = _meta_value(meta, "LicenseShortName")
        if not _license_allowed(license_name):
            continue

        file_title = str(page.get("title", "")).strip()
        image_url = str(info.get("thumburl") or info.get("url") or "").strip()
        if not file_title or not image_url:
            continue

        results.append(
            {
                "title": file_title.removeprefix("File:"),
                "file_title": file_title,
                "image_url": image_url,
                "author": _meta_value(meta, "Artist") or "Belirtilmemiş",
                "license": license_name,
                "license_url": _meta_value(meta, "LicenseUrl"),
                "credit": _meta_value(meta, "Credit"),
                "source_page": (
                    "https://commons.wikimedia.org/wiki/"
                    + quote(file_title.replace(" ", "_"))
                ),
            }
        )
    return results


def download_visuals(
    queries: list[str],
    target_count: int,
    download_dir: Path,
) -> tuple[list[Path], list[dict[str, str]]]:
    download_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    credits: list[dict[str, str]] = []
    seen_titles: set[str] = set()

    expanded_queries = list(queries) + [
        "ancient archaeological ruins",
        "ancient history museum artifact",
        "historical map public domain",
    ]

    for query in expanded_queries:
        if len(paths) >= target_count:
            break
        print(f"Wikimedia aranıyor: {query}")
        try:
            candidates = search_commons(query)
        except Exception as exc:
            print(f"Wikimedia araması başarısız: {exc}")
            continue

        for item in candidates:
            if len(paths) >= target_count:
                break
            title_key = item["file_title"].lower()
            if title_key in seen_titles:
                continue

            suffix = Path(item["image_url"].split("?", 1)[0]).suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                suffix = ".jpg"
            destination = download_dir / f"source_{len(paths) + 1:03d}{suffix}"

            try:
                response = requests.get(
                    item["image_url"],
                    headers={"User-Agent": USER_AGENT},
                    timeout=45,
                )
                response.raise_for_status()
                destination.write_bytes(response.content)
                with Image.open(destination) as check:
                    check.verify()
            except Exception as exc:
                destination.unlink(missing_ok=True)
                print(f"Görsel indirilemedi: {exc}")
                continue

            seen_titles.add(title_key)
            paths.append(destination)
            credits.append(item)
            time.sleep(0.15)

    return paths, credits


def credits_text(credits: list[dict[str, str]]) -> str:
    if not credits:
        return "Bu çalışmada dış kaynaklı görsel kullanılamadı."

    lines = ["GÖRSEL KAYNAKLARI VE LİSANSLAR", ""]
    for index, item in enumerate(credits, start=1):
        line = (
            f"{index}. {item['title']} — {item['author']} — "
            f"{item['license']} — {item['source_page']}"
        )
        if item.get("license_url"):
            line += f" — Lisans: {item['license_url']}"
        lines.append(line)
    return "\n".join(lines)
