from __future__ import annotations

import argparse
import base64
import html
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps, ImageStat
from google import genai
from google.genai import types

VERSION = "10.4.0"
ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT / "projects"
FPS = 24
WIDTH = 1920
HEIGHT = 1080
VOICE_NAME = "Charon"
TEXT_MODELS = [
    os.getenv("TEXT_MODEL", "").strip(),
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
]
TTS_MODEL = os.getenv("TTS_MODEL", "gemini-2.5-flash-preview-tts").strip()

CHARON_CHUNK_TARGET_WORDS = max(
    360,
    int(os.getenv("CHARON_CHUNK_TARGET_WORDS", "520")),
)
CHARON_ONLY = True

RESEARCH_QUESTION_WORDS = {
    "nasıl", "nasil", "nedir", "kimdir", "nerede", "ne", "neden",
    "niçin", "nicin", "gerçekleşti", "gerceklesti", "oldu",
    "olmuştur", "olmustur", "anlat", "hakkında", "hakkinda", "tarihi",
    "what", "how", "why", "when", "where", "who", "was", "were",
    "happened", "history", "explained",
}

VISUAL_REJECT_TERMS = {
    "lizard", "iguana", "reptile", "snake", "serpent", "gecko",
    "kertenkele", "sürüngen", "surungen", "animal", "wildlife",
    "bird", "fish", "insect", "butterfly", "spider", "flower",
    "plant", "fungus", "fossil", "mineral", "zoo", "aquarium",
    "sports", "football", "car", "automobile", "aircraft", "airport",
    "hotel", "restaurant", "shopping mall", "modern apartment",
}
VISUAL_GENERIC_TERMS = {
    "history", "historical", "old", "ancient", "photo", "image",
    "tarih", "tarihi", "eski", "antik", "görsel", "fotograf",
}
VISUAL_MIN_SCORE = 62.0


COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKI_API = {
    "tr": "https://tr.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}
ALLOWED_LICENSE_MARKERS = (
    "public domain",
    "pd-",
    "cc0",
    "cc by",
    "cc-by",
    "cc by-sa",
    "cc-by-sa",
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
STOPWORDS = {
    "the", "and", "of", "in", "on", "at", "for", "from", "with", "to",
    "bir", "ve", "ile", "için", "sonra", "önce", "olan", "olarak", "bu",
    "şu", "da", "de", "ile", "üzerinde", "arasındaki", "dönemi",
}


class ControlledStop(RuntimeError):
    """A safe stop that preserves checkpoints and should not waste prior work."""


class ProviderUnavailable(ControlledStop):
    pass


@dataclass
class ProjectContext:
    slug: str
    topic: str
    mode: str
    target_minutes: int
    chapter_count: int
    root: Path

    @property
    def state_file(self) -> Path:
        return self.root / "state.json"

    @property
    def deliverables(self) -> Path:
        return self.root / "deliverables"

    @property
    def chapters_dir(self) -> Path:
        return self.root / "chapters"


def log(message: str) -> None:
    print(message, flush=True)


def slugify(value: str) -> str:
    value = value.strip().lower()
    replacements = str.maketrans({
        "ı": "i", "ğ": "g", "ü": "u", "ş": "s", "ö": "o", "ç": "c",
    })
    value = value.translate(replacements)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value[:70] or "longform-video"


def run(command: list[str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    log("$ " + " ".join(command))
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def ffprobe_duration(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], timeout=60)
    return float(result.stdout.strip())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def strip_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", value).strip()


def word_count(value: str) -> int:
    return len(re.findall(r"\b\w+[\w'’-]*\b", value, flags=re.UNICODE))


def balanced_text_chunks(text: str, target_words: int = 210) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
        if item.strip()
    ]
    if not sentences:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        proposed = " ".join(current + [sentence])
        if current and word_count(proposed) > target_words:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    if len(chunks) > 1 and word_count(chunks[-1]) < 70:
        chunks[-2] += " " + chunks[-1]
        chunks.pop()
    return chunks


def split_by_word_weight(text: str, count: int) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
        if item.strip()
    ]
    if not sentences:
        return [text.strip()] * count
    total = max(1, sum(word_count(s) for s in sentences))
    target = total / count
    groups: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        w = word_count(sentence)
        if current and current_words + w > target and len(groups) < count - 1:
            groups.append(" ".join(current))
            current = [sentence]
            current_words = w
        else:
            current.append(sentence)
            current_words += w
    if current:
        groups.append(" ".join(current))
    while len(groups) < count:
        longest = max(range(len(groups)), key=lambda i: word_count(groups[i]))
        words = groups[longest].split()
        half = max(1, len(words) // 2)
        groups[longest:longest + 1] = [" ".join(words[:half]), " ".join(words[half:])]
    return groups[:count]


def model_chain(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        value = str(value or "").strip()
        if value and value not in result:
            result.append(value)
    return result


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("Model geçerli JSON döndürmedi.")
    return json.loads(text[first:last + 1])


def generate_json(
    client: genai.Client,
    prompt: str,
    *,
    max_tokens: int = 8192,
    retries: int = 3,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None
    for model in model_chain(TEXT_MODELS):
        for attempt in range(1, retries + 1):
            try:
                log(f"Gemini JSON: model={model}, deneme={attempt}/{retries}")
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=max_tokens,
                        temperature=0.35,
                    ),
                )
                payload = extract_json(response.text or "")
                return payload, model
            except Exception as exc:
                last_error = exc
                log(f"Gemini JSON başarısız: {exc}")
                time.sleep(min(25, attempt * 7))
    raise ProviderUnavailable(f"Metin modeli kullanılamadı: {last_error}")


def requests_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": os.getenv(
            "WIKIMEDIA_USER_AGENT",
            "UykuTarihV10/1.0 (automated documentary research; contact via repository)",
        )
    })
    return session


def research_tokens(value: str) -> set[str]:
    normalized = slugify(value).replace("-", " ")
    return {
        token
        for token in normalized.split()
        if (
            len(token) >= 3
            and token not in STOPWORDS
            and token not in RESEARCH_QUESTION_WORDS
        )
    }


def core_research_query(value: str) -> str:
    words = re.findall(
        r"[0-9A-Za-zÇĞİÖŞÜçğıöşü'’\-]+",
        str(value),
    )
    kept: list[str] = []
    for word in words:
        normalized = slugify(word)
        if (
            normalized
            and normalized not in RESEARCH_QUESTION_WORDS
            and normalized not in STOPWORDS
        ):
            kept.append(word)

    clean = " ".join(kept).strip()
    return clean or str(value).strip().rstrip("?!. ")


def source_relevance_score(
    source: dict[str, Any],
    query: str,
) -> int:
    query_tokens = research_tokens(query)
    title_tokens = research_tokens(
        str(source.get("title", ""))
    )
    extract_tokens = research_tokens(
        str(source.get("extract", ""))[:1800]
    )
    if not query_tokens:
        return 0

    title_overlap = len(query_tokens & title_tokens)
    extract_overlap = len(query_tokens & extract_tokens)
    normalized_query = slugify(query)
    normalized_title = slugify(
        str(source.get("title", ""))
    )

    score = title_overlap * 45 + min(30, extract_overlap * 5)
    if (
        normalized_query
        and normalized_query in normalized_title
    ):
        score += 80
    if len(str(source.get("extract", ""))) >= 350:
        score += 10
    return score


def source_is_relevant(
    source: dict[str, Any],
    query: str,
) -> bool:
    query_tokens = research_tokens(query)
    title_tokens = research_tokens(
        str(source.get("title", ""))
    )
    overlap = len(query_tokens & title_tokens)

    if len(query_tokens) >= 2 and overlap < 2:
        return False
    if len(query_tokens) == 1 and overlap < 1:
        return False
    if len(str(source.get("extract", "")).strip()) < 120:
        return False
    return source_relevance_score(source, query) >= 85


def _fetch_wiki_pages(
    session: requests.Session,
    endpoint: str,
    pageids: list[str],
    language: str,
) -> list[dict[str, Any]]:
    if not pageids:
        return []
    response = session.get(
        endpoint,
        params={
            "action": "query",
            "pageids": "|".join(pageids),
            "prop": "extracts|info",
            "explaintext": 1,
            "exchars": 5200,
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        },
        timeout=30,
    )
    response.raise_for_status()
    result: list[dict[str, Any]] = []
    for page in response.json().get(
        "query",
        {},
    ).get("pages", []):
        result.append({
            "language": language,
            "title": page.get("title", ""),
            "url": page.get("fullurl", ""),
            "extract": re.sub(
                r"\s+",
                " ",
                page.get("extract", ""),
            ).strip(),
        })
    return result


def wiki_research(
    session: requests.Session,
    query: str,
    language: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    endpoint = WIKI_API[language]
    core_query = core_research_query(query)

    search_variants = [
        f'intitle:"{core_query}"',
        f'"{core_query}"',
        core_query,
    ]
    pageids: list[str] = []
    seen_pageids: set[str] = set()

    for search_query in search_variants:
        search_response = session.get(
            endpoint,
            params={
                "action": "query",
                "list": "search",
                "srsearch": search_query,
                "srnamespace": 0,
                "srlimit": max(5, limit * 3),
                "format": "json",
                "formatversion": 2,
            },
            timeout=30,
        )
        search_response.raise_for_status()
        hits = search_response.json().get(
            "query",
            {},
        ).get("search", [])
        for item in hits:
            pageid = str(item.get("pageid", ""))
            if pageid and pageid not in seen_pageids:
                seen_pageids.add(pageid)
                pageids.append(pageid)
        if len(pageids) >= limit * 3:
            break

    candidates = _fetch_wiki_pages(
        session,
        endpoint,
        pageids,
        language,
    )
    candidates.sort(
        key=lambda source: source_relevance_score(
            source,
            core_query,
        ),
        reverse=True,
    )
    return [
        source
        for source in candidates
        if source_is_relevant(source, core_query)
    ][:limit]


def research_is_usable(
    research: dict[str, Any] | None,
    topic: str,
) -> bool:
    if not isinstance(research, dict):
        return False
    sources = research.get("sources")
    if not isinstance(sources, list):
        return False
    core = core_research_query(topic)
    return any(
        source_is_relevant(source, core)
        for source in sources
        if isinstance(source, dict)
    )


def research_fingerprint(
    research: dict[str, Any],
) -> str:
    compact = [
        {
            "title": source.get("title", ""),
            "url": source.get("url", ""),
            "extract": str(source.get("extract", ""))[:1000],
        }
        for source in research.get("sources", [])
    ]
    raw = json.dumps(
        compact,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def research_topic(
    session: requests.Session,
    topic: str,
    extra_queries: list[str],
) -> dict[str, Any]:
    core = core_research_query(topic)
    queries = [
        core,
        *[
            core_research_query(query)
            for query in extra_queries
            if str(query).strip()
        ],
    ]
    queries = list(dict.fromkeys(
        query for query in queries if query
    ))

    sources: list[dict[str, Any]] = []
    seen: set[str] = set()

    for query in queries[:8]:
        for lang in ("tr", "en"):
            try:
                for item in wiki_research(
                    session,
                    query,
                    lang,
                    limit=3,
                ):
                    key = (
                        item.get("url")
                        or f'{lang}:{item.get("title", "")}'
                    )
                    if key not in seen:
                        seen.add(key)
                        item["relevance_score"] = (
                            source_relevance_score(
                                item,
                                query,
                            )
                        )
                        item["matched_query"] = query
                        sources.append(item)
            except Exception as exc:
                log(
                    "Wikipedia araştırması atlandı "
                    f"({lang}/{query}): {exc}"
                )

    sources.sort(
        key=lambda source: int(
            source.get("relevance_score", 0)
        ),
        reverse=True,
    )

    result = {
        "topic": topic,
        "core_query": core,
        "queries": queries,
        "sources": sources[:14],
        "research_version": VERSION,
    }
    if not research_is_usable(result, topic):
        raise ControlledStop(
            "Konuya doğrudan bağlı güvenilir Wikipedia kaynağı "
            f"bulunamadı. Yanlış kaynakla video üretilmedi. Aranan konu: {core}"
        )
    return result


def story_plan_prompt(topic: str, target_minutes: int, chapter_count: int, research: dict[str, Any]) -> str:
    total_words = round(target_minutes * 112)
    per_chapter = max(850, round(total_words / chapter_count))
    source_context = [
        {"title": s["title"], "url": s["url"], "extract": s["extract"][:2400]}
        for s in research.get("sources", [])[:10]
    ]
    return f"""
Yalnızca geçerli JSON üret.

KONU: {topic}
HEDEF SÜRE: {target_minutes} dakika
BÖLÜM SAYISI: {chapter_count}
TOPLAM HEDEF: yaklaşık {total_words} Türkçe kelime

KAYNAK BAĞLAMI:
{json.dumps(source_context, ensure_ascii=False)}

ŞEMA:
{{
  "video_title": "Türkçe başlık",
  "thumbnail_text": "en fazla 6 kelime",
  "video_description": "özgün kısa açıklama",
  "story_bible": {{
    "historical_scope": "kapsam",
    "timeline": ["olay sırası"],
    "key_figures": ["kişiler"],
    "key_places": ["mekânlar"],
    "continuity_rules": ["tutarlılık kuralları"],
    "tone": "sakin fakat olay odaklı tarih belgeseli"
  }},
  "chapters": [
    {{
      "chapter_index": 1,
      "title": "bölüm adı",
      "objective": "bu bölümün olay hedefi",
      "opening_hook": "ilk merak cümlesinin amacı",
      "closing_bridge": "sonraki bölüme bağ",
      "target_words": {per_chapter},
      "research_queries": ["Türkçe sorgu", "English query"],
      "must_cover": ["somut olaylar"],
      "avoid_repetition": ["tekrarlanmaması gereken fikirler"]
    }}
  ]
}}

KURALLAR:
- Tam {chapter_count} bölüm üret.
- Kronolojik ve neden-sonuç ilişkili bir omurga kur.
- Her bölüm farklı bir dramatik göreve sahip olsun.
- Şiirsel betimleme toplamın yüzde 12'sini geçmesin.
- Olay, karar, çatışma, sonuç ve insan etkisi ana yapı olsun.
- Kaynak bağlamında bulunmayan aşırı kesin ayrıntılar uydurma.
- Aynı giriş cümlesini veya aynı özeti bölümlerde tekrar etme.
"""


def validate_plan(plan: dict[str, Any], chapter_count: int) -> dict[str, Any]:
    chapters = plan.get("chapters")
    if not isinstance(chapters, list) or len(chapters) != chapter_count:
        raise ValueError(f"Plan {chapter_count} bölüm içermiyor.")
    for index, chapter in enumerate(chapters, start=1):
        chapter["chapter_index"] = index
        chapter.setdefault("title", f"Bölüm {index}")
        chapter.setdefault("objective", "")
        chapter.setdefault("target_words", 1100)
        chapter.setdefault("research_queries", [])
        chapter.setdefault("must_cover", [])
    plan.setdefault("story_bible", {})
    return plan


def chapter_script_prompt(
    topic: str,
    plan: dict[str, Any],
    chapter: dict[str, Any],
    research: dict[str, Any],
    beat_count: int,
) -> str:
    source_context = [
        {"title": s["title"], "url": s["url"], "extract": s["extract"][:3000]}
        for s in research.get("sources", [])[:10]
    ]
    return f"""
Yalnızca geçerli JSON üret.

VİDEO KONUSU: {topic}
HİKÂYE ANAYASASI:
{json.dumps(plan.get('story_bible', {}), ensure_ascii=False)}

BÖLÜM PLANI:
{json.dumps(chapter, ensure_ascii=False)}

KAYNAK BAĞLAMI:
{json.dumps(source_context, ensure_ascii=False)}

ŞEMA:
{{
  "chapter_title": "Türkçe bölüm adı",
  "narration": "tek parça Türkçe anlatım",
  "visual_beats": [
    {{
      "beat_id": 1,
      "narration_excerpt": "anlatımdan bu görsele ait bölüm",
      "visual_contract": "karede görünmesi gereken somut olay",
      "must_show": ["ana özne", "eylem", "mekân veya nesne"],
      "query_tr": "Wikimedia Commons Türkçe arama sorgusu",
      "query_en": "Wikimedia Commons English search query",
      "asset_type": "painting|photo|architecture|artifact|timeline|map",
      "local_graphic_title": "yalnız timeline/map ise Türkçe başlık",
      "local_graphic_points": ["yalnız timeline/map ise 2-4 kısa nokta"]
    }}
  ]
}}

KURALLAR:
- narration yaklaşık {chapter.get('target_words', 1100)} kelime olsun.
- Tam {beat_count} visual_beats üret.
- Her visual beat anlatım sırasına göre ilerlesin.
- narration_excerpt alanları birleştiğinde narration sırasını izlesin.
- Görsel sorguları soyut değil, aranabilir özel isim, mekân, eser, yapı,
  tablo, gravür veya arkeolojik nesne içersin.
- Harita, zaman çizelgesi ve şema gerekiyorsa asset_type=map veya timeline kullan;
  bunlar Türkçe ve yerel olarak oluşturulacaktır.
- En fazla 2 adet map/timeline beat kullan.
- Aynı kişi veya tabloyu art arda tekrar etme.
- Görsel, anlatıcının o anda söylediği şeyi doğrudan desteklemeli.
- Şiirsel dolgu, tekrar giriş ve bölüm sonunda özet listesi kullanma.
"""


def chapter_word_targets(
    chapter: dict[str, Any],
) -> tuple[int, int]:
    requested = max(
        650,
        int(chapter.get("target_words", 1000)),
    )
    minimum = max(
        520,
        round(requested * 0.78),
    )
    return requested, minimum


def _research_context_for_prompt(
    research: dict[str, Any],
    *,
    limit: int = 10,
    extract_chars: int = 2500,
) -> list[dict[str, str]]:
    return [
        {
            "title": str(source.get("title", "")),
            "url": str(source.get("url", "")),
            "extract": str(source.get("extract", ""))[:extract_chars],
        }
        for source in research.get("sources", [])[:limit]
    ]


def _tail_sentences(
    narration: str,
    count: int = 3,
) -> str:
    sentences = [
        item.strip()
        for item in re.split(
            r"(?<=[.!?])\s+",
            re.sub(r"\s+", " ", narration).strip(),
        )
        if item.strip()
    ]
    return " ".join(sentences[-count:])


def _normalize_narration_piece(value: Any) -> str:
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def _append_narration_piece(
    narration: str,
    addition: str,
) -> str:
    narration = _normalize_narration_piece(narration)
    addition = _normalize_narration_piece(addition)
    if not addition:
        return narration
    if not narration:
        return addition

    # Remove a repeated opening sentence when the model restarts from the
    # previous tail instead of continuing it.
    existing_sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", narration)
        if item.strip()
    ]
    addition_sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", addition)
        if item.strip()
    ]
    existing_tail = {
        sentence.lower()
        for sentence in existing_sentences[-4:]
    }
    while (
        addition_sentences
        and addition_sentences[0].lower() in existing_tail
    ):
        addition_sentences.pop(0)

    clean_addition = " ".join(addition_sentences).strip()
    if not clean_addition:
        return narration
    return f"{narration} {clean_addition}".strip()


def chapter_segment_prompt(
    topic: str,
    plan: dict[str, Any],
    chapter: dict[str, Any],
    research: dict[str, Any],
    *,
    segment_index: int,
    segment_count: int,
    segment_target_words: int,
    narration_so_far: str,
) -> str:
    sources = _research_context_for_prompt(
        research,
        limit=10,
        extract_chars=2200,
    )
    previous_tail = _tail_sentences(
        narration_so_far,
        3,
    )

    if segment_index == 1:
        continuity = (
            "Bu ilk parçadır. İlk 30 kelime içinde bölümün temel sorusunu "
            "veya çatışmasını kur. Uzun bir genel giriş yapma."
        )
    else:
        continuity = (
            "Bu yeni bir bölüm başlangıcı değildir. Aşağıdaki önceki son "
            "cümlelerden doğrudan devam et; giriş, başlık ve özet tekrarı yapma.\n"
            f"ÖNCEKİ SON CÜMLELER:\n{previous_tail}"
        )

    if segment_index == segment_count:
        ending = (
            "Bu son parçadır. Bölümün olay sonucunu anlat ve planın "
            "closing_bridge alanına doğal biçimde bağlan; madde madde özet yapma."
        )
    else:
        ending = (
            "Bu parçada nihai sonuç yazma. Olayı bir sonraki parçaya taşıyacak "
            "somut bir karar, gelişme veya gerilim noktasında bırak."
        )

    return f"""
Yalnızca geçerli JSON üret:
{{
  "segment": "Türkçe anlatım parçası"
}}

VİDEO KONUSU:
{topic}

HİKÂYE ANAYASASI:
{json.dumps(plan.get('story_bible', {}), ensure_ascii=False)}

BÖLÜM PLANI:
{json.dumps(chapter, ensure_ascii=False)}

KAYNAK BAĞLAMI:
{json.dumps(sources, ensure_ascii=False)}

PARÇA:
{segment_index}/{segment_count}

HEDEF:
Yaklaşık {segment_target_words} Türkçe kelime.

KURALLAR:
- Bu bir olay anlatısıdır; betimleyici dolgu metni değildir.
- Kim ne istedi, ne yaptı, neyle karşılaştı ve bunun sonucu ne oldu sorularını
  neden-sonuç ilişkisiyle anlat.
- Özel isimleri, tarihleri, mekânları ve kararları kaynak bağlamına uygun kullan.
- Kaynaklarda desteklenmeyen kesin ayrıntı uydurma.
- Aynı bilgiyi farklı kelimelerle tekrar etme.
- Başlık, bölüm numarası, madde işareti veya kamera tarifi yazma.
- Şiirsel giriş, fragman dili ve uzun atmosfer betimlemesi kullanma.
- {continuity}
- {ending}
"""


def chapter_continuation_prompt(
    topic: str,
    plan: dict[str, Any],
    chapter: dict[str, Any],
    research: dict[str, Any],
    *,
    narration_so_far: str,
    needed_words: int,
) -> str:
    sources = _research_context_for_prompt(
        research,
        limit=10,
        extract_chars=1800,
    )
    target = max(
        180,
        min(360, needed_words + 60),
    )
    return f"""
Yalnızca geçerli JSON üret:
{{
  "continuation": "Mevcut anlatımın doğrudan devamı"
}}

KONU:
{topic}

BÖLÜM:
{json.dumps(chapter, ensure_ascii=False)}

HİKÂYE ANAYASASI:
{json.dumps(plan.get('story_bible', {}), ensure_ascii=False)}

KAYNAKLAR:
{json.dumps(sources, ensure_ascii=False)}

MEVCUT ANLATIMIN SONU:
{_tail_sentences(narration_so_far, 5)}

GÖREV:
Anlatımı yaklaşık {target} kelime daha devam ettir.

KURALLAR:
- Baştan başlama ve mevcut cümleleri tekrar etme.
- Yeni başlık veya giriş kullanma.
- Bölüm planındaki henüz anlatılmamış olay, karar ve sonuçları tamamla.
- Kaynaklarda desteklenmeyen kesin bilgi uydurma.
- Şiirsel dolgu ve genel tarih dersi yazma.
- Son cümlede bölümün closing_bridge alanına doğal biçimde yaklaş.
"""


def visual_beats_prompt(
    topic: str,
    chapter: dict[str, Any],
    narration: str,
    beat_count: int,
) -> str:
    chunks = split_by_word_weight(
        narration,
        beat_count,
    )
    numbered = [
        {
            "beat_id": index,
            "narration_excerpt": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
    return f"""
Yalnızca geçerli JSON üret:
{{
  "visual_beats": [
    {{
      "beat_id": 1,
      "narration_excerpt": "verilen parçayı aynen veya çok yakın biçimde kullan",
      "visual_contract": "karede görünmesi gereken somut olay",
      "must_show": ["ana özne", "ana eylem", "mekân veya nesne"],
      "query_tr": "Wikimedia Commons Türkçe arama sorgusu",
      "query_en": "Wikimedia Commons English search query",
      "asset_type": "painting|photo|architecture|artifact|timeline|map",
      "local_graphic_title": "yalnız timeline/map ise Türkçe başlık",
      "local_graphic_points": ["yalnız timeline/map ise 2-4 kısa nokta"]
    }}
  ]
}}

KONU:
{topic}

BÖLÜM:
{json.dumps(chapter, ensure_ascii=False)}

NİHAİ ANLATIM PARÇALARI:
{json.dumps(numbered, ensure_ascii=False)}

KURALLAR:
- Tam {beat_count} visual_beats üret.
- Her beat_id verilen anlatım parçasıyla aynı sırada ve aynı olayda kalsın.
- Görsel, anlatıcının o anda söylediği kişi, eylem, nesne ve mekânı doğrudan
  desteklemeli; genel veya alakasız tarih atmosferi önermemeli.
- query_tr ve query_en soyut değil; Wikimedia Commons'ta aranabilecek özel isim,
  yapı, şehir, tablo, gravür, eser veya arkeolojik nesne içersin.
- En fazla 2 adet map/timeline beat kullan.
- Aynı görsel sorgusunu art arda tekrar etme.
"""


def _fallback_visual_beats(
    narration: str,
    chapter: dict[str, Any],
    beat_count: int,
) -> list[dict[str, Any]]:
    chunks = split_by_word_weight(
        narration,
        beat_count,
    )
    result: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        compact = re.sub(
            r"\s+",
            " ",
            chunk,
        ).strip()
        result.append({
            "beat_id": index,
            "narration_excerpt": compact,
            "visual_contract": compact[:260],
            "must_show": [],
            "query_tr": (
                f'{chapter.get("title", "")} '
                f'{compact[:100]}'
            ).strip(),
            "query_en": (
                f'{chapter.get("title", "")} '
                "historical painting engraving"
            ).strip(),
            "asset_type": "painting",
            "local_graphic_title": "",
            "local_graphic_points": [],
        })
    return result


def build_chapter_script_resilient(
    client: genai.Client,
    topic: str,
    plan: dict[str, Any],
    chapter: dict[str, Any],
    research: dict[str, Any],
    beat_count: int,
    chapter_dir: Path,
) -> dict[str, Any]:
    target_words, minimum_words = chapter_word_targets(
        chapter,
    )
    draft_file = chapter_dir / "script-draft.json"
    segments_file = chapter_dir / "narration-segments.json"

    draft = read_json(
        draft_file,
        {},
    )
    narration = _normalize_narration_piece(
        draft.get("narration", "")
    )
    text_models: list[str] = list(
        draft.get("text_models", [])
    )
    segments = read_json(
        segments_file,
        [],
    )
    if not isinstance(segments, list):
        segments = []

    segment_count = 4
    segment_target = max(
        210,
        round(target_words / segment_count),
    )

    log(
        "RESILIENT SCRIPT: "
        f"hedef={target_words}, minimum={minimum_words}, "
        f"mevcut={word_count(narration)} kelime"
    )

    for segment_index in range(
        len(segments) + 1,
        segment_count + 1,
    ):
        payload, model = generate_json(
            client,
            chapter_segment_prompt(
                topic,
                plan,
                chapter,
                research,
                segment_index=segment_index,
                segment_count=segment_count,
                segment_target_words=segment_target,
                narration_so_far=narration,
            ),
            max_tokens=5200,
        )
        piece = _normalize_narration_piece(
            payload.get("segment")
            or payload.get("continuation")
            or payload.get("narration")
        )
        if word_count(piece) < 90:
            log(
                "Kısa segment alındı; aynı parça bir kez daha üretilecek: "
                f"{word_count(piece)} kelime"
            )
            retry_payload, retry_model = generate_json(
                client,
                chapter_segment_prompt(
                    topic,
                    plan,
                    chapter,
                    research,
                    segment_index=segment_index,
                    segment_count=segment_count,
                    segment_target_words=segment_target + 80,
                    narration_so_far=narration,
                ),
                max_tokens=6200,
            )
            retry_piece = _normalize_narration_piece(
                retry_payload.get("segment")
                or retry_payload.get("continuation")
                or retry_payload.get("narration")
            )
            if word_count(retry_piece) > word_count(piece):
                piece = retry_piece
                model = retry_model

        if word_count(piece) < 50:
            raise ControlledStop(
                "Gemini bölüm parçasını yeterli uzunlukta üretemedi. "
                f"Parça {segment_index}: {word_count(piece)} kelime. "
                "Mevcut segment checkpointleri korundu."
            )

        narration = _append_narration_piece(
            narration,
            piece,
        )
        segments.append({
            "segment_index": segment_index,
            "model": model,
            "word_count": word_count(piece),
            "text": piece,
        })
        if model not in text_models:
            text_models.append(model)

        write_json(
            segments_file,
            segments,
        )
        write_json(
            draft_file,
            {
                "chapter_title": chapter.get("title", ""),
                "narration": narration,
                "text_models": text_models,
                "target_words": target_words,
                "minimum_words": minimum_words,
                "status": "segments_in_progress",
            },
        )
        log(
            f"Segment {segment_index}/{segment_count} hazır: "
            f"toplam {word_count(narration)} kelime"
        )

    # The model may still undershoot its requested word count. Continue in
    # small source-grounded passes rather than discarding the entire chapter.
    continuation_pass = 0
    while (
        word_count(narration) < minimum_words
        and continuation_pass < 4
    ):
        continuation_pass += 1
        needed = minimum_words - word_count(narration)
        payload, model = generate_json(
            client,
            chapter_continuation_prompt(
                topic,
                plan,
                chapter,
                research,
                narration_so_far=narration,
                needed_words=needed,
            ),
            max_tokens=5200,
        )
        piece = _normalize_narration_piece(
            payload.get("continuation")
            or payload.get("segment")
            or payload.get("narration")
        )
        before = word_count(narration)
        narration = _append_narration_piece(
            narration,
            piece,
        )
        gained = word_count(narration) - before

        if model not in text_models:
            text_models.append(model)

        write_json(
            draft_file,
            {
                "chapter_title": chapter.get("title", ""),
                "narration": narration,
                "text_models": text_models,
                "target_words": target_words,
                "minimum_words": minimum_words,
                "continuation_pass": continuation_pass,
                "status": "length_recovery",
            },
        )
        log(
            f"Uzunluk tamamlama {continuation_pass}/4: "
            f"+{gained}, toplam={word_count(narration)} kelime"
        )
        if gained < 45:
            break

    if word_count(narration) < minimum_words:
        raise ControlledStop(
            "Bölüm anlatımı otomatik tamamlama sonrasında hâlâ kısa kaldı: "
            f"{word_count(narration)} / minimum {minimum_words} kelime. "
            "Taslak ve tamamlanan segmentler checkpointte korundu; aynı "
            "project_slug ile tekrar çalıştırıldığında kaldığı yerden devam eder."
        )

    # Visual beats are generated only after the final narration is ready.
    beats: list[dict[str, Any]]
    beat_model = "local-fallback"
    try:
        beat_payload, beat_model = generate_json(
            client,
            visual_beats_prompt(
                topic,
                chapter,
                narration,
                beat_count,
            ),
            max_tokens=9000,
        )
        candidate = beat_payload.get("visual_beats")
        if not isinstance(candidate, list) or len(candidate) != beat_count:
            raise ValueError(
                f"Görsel plan {len(candidate) if isinstance(candidate, list) else 0}"
                f"/{beat_count} beat döndürdü."
            )
        beats = candidate
    except Exception as exc:
        log(
            "Görsel beat modeli kullanılamadı; nihai anlatımdan güvenli "
            f"yerel plan oluşturuldu: {exc}"
        )
        beats = _fallback_visual_beats(
            narration,
            chapter,
            beat_count,
        )

    result = {
        "chapter_title": chapter.get("title", ""),
        "narration": narration,
        "visual_beats": beats,
        "text_model": ", ".join(text_models) or "unknown",
        "visual_plan_model": beat_model,
        "target_words": target_words,
        "minimum_words": minimum_words,
        "actual_words": word_count(narration),
        "generation_mode": "resilient-segmented",
    }
    result = validate_chapter_script(
        result,
        chapter,
        beat_count,
    )
    write_json(
        draft_file,
        {
            **result,
            "status": "complete",
        },
    )
    return result


def validate_chapter_script(payload: dict[str, Any], chapter: dict[str, Any], beat_count: int) -> dict[str, Any]:
    narration = re.sub(r"\s+", " ", str(payload.get("narration", ""))).strip()
    _, minimum_words = chapter_word_targets(chapter)
    if word_count(narration) < minimum_words:
        raise ControlledStop(
            "Bölüm metni minimum uzunluğa ulaşmadı: "
            f"{word_count(narration)} / {minimum_words} kelime. "
            "Taslak checkpointi korunuyor."
        )
    beats = payload.get("visual_beats")
    if not isinstance(beats, list):
        beats = []
    if len(beats) != beat_count:
        chunks = split_by_word_weight(narration, beat_count)
        fallback: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            fallback.append({
                "beat_id": index,
                "narration_excerpt": chunk,
                "visual_contract": chunk[:220],
                "must_show": [],
                "query_tr": f'{chapter.get("title", "")} {chunk[:80]}',
                "query_en": f'{chapter.get("title", "")} history',
                "asset_type": "painting",
                "local_graphic_title": "",
                "local_graphic_points": [],
            })
        beats = fallback
    for index, beat in enumerate(beats, start=1):
        beat["beat_id"] = index
        beat.setdefault("narration_excerpt", "")
        beat.setdefault("visual_contract", beat["narration_excerpt"])
        beat.setdefault("must_show", [])
        beat.setdefault("query_tr", chapter.get("title", ""))
        beat.setdefault("query_en", chapter.get("title", ""))
        beat.setdefault("asset_type", "painting")
        beat.setdefault("local_graphic_title", "")
        beat.setdefault("local_graphic_points", [])
    payload["narration"] = narration
    payload["visual_beats"] = beats
    payload.setdefault("chapter_title", chapter.get("title", ""))
    return payload


def license_allowed(value: str) -> bool:
    normalized = strip_html(value).lower()
    return any(marker in normalized for marker in ALLOWED_LICENSE_MARKERS)


def candidate_tokens(value: str) -> set[str]:
    tokens = {
        token for token in re.findall(r"[a-zA-ZÀ-ž0-9]+", value.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }
    return tokens


def commons_candidates(session: requests.Session, query: str, limit: int = 12) -> list[dict[str, Any]]:
    response = session.get(
        COMMONS_API,
        params={
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "iiurlwidth": 1920,
            "format": "json",
            "formatversion": 2,
        },
        timeout=40,
    )
    response.raise_for_status()
    pages = response.json().get("query", {}).get("pages", [])
    result: list[dict[str, Any]] = []
    for page in pages:
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        mime = str(info.get("mime", ""))
        url = info.get("thumburl") or info.get("url")
        if not mime.startswith("image/") or not url:
            continue
        suffix = Path(url.split("?", 1)[0]).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        metadata = info.get("extmetadata", {})
        license_name = strip_html(metadata.get("LicenseShortName", {}).get("value", ""))
        usage_terms = strip_html(metadata.get("UsageTerms", {}).get("value", ""))
        if not license_allowed(license_name + " " + usage_terms):
            continue
        result.append({
            "pageid": page.get("pageid"),
            "title": page.get("title", ""),
            "url": url,
            "description_url": info.get("descriptionurl", ""),
            "width": info.get("thumbwidth") or info.get("width") or 0,
            "height": info.get("thumbheight") or info.get("height") or 0,
            "mime": mime,
            "license": license_name or usage_terms,
            "license_url": strip_html(metadata.get("LicenseUrl", {}).get("value", "")),
            "artist": strip_html(metadata.get("Artist", {}).get("value", "")),
            "credit": strip_html(metadata.get("Credit", {}).get("value", "")),
            "description": strip_html(metadata.get("ImageDescription", {}).get("value", "")),
            "date": strip_html(metadata.get("DateTimeOriginal", {}).get("value", "")),
        })
    return result


def _visual_text(candidate: dict[str, Any]) -> str:
    return " ".join([
        str(candidate.get("title", "")),
        str(candidate.get("description", "")),
        str(candidate.get("credit", "")),
        str(candidate.get("artist", "")),
    ])


def _meaningful_visual_tokens(value: str) -> set[str]:
    return {
        token
        for token in candidate_tokens(value)
        if token not in VISUAL_GENERIC_TERMS
    }


def candidate_is_forbidden(
    candidate: dict[str, Any],
    beat: dict[str, Any],
) -> bool:
    candidate_text = slugify(_visual_text(candidate)).replace("-", " ")
    beat_text = slugify(" ".join([
        str(beat.get("narration_excerpt", "")),
        str(beat.get("visual_contract", "")),
        " ".join(str(item) for item in beat.get("must_show", [])),
        " ".join(str(item) for item in beat.get("sync_keywords", [])),
    ])).replace("-", " ")

    for term in VISUAL_REJECT_TERMS:
        normalized = slugify(term).replace("-", " ")
        if normalized in candidate_text and normalized not in beat_text:
            return True

    title = str(candidate.get("title", "")).lower()
    if any(
        term in title
        for term in (
            "logo", "coat of arms", "flag", "stamp", "currency",
            "banknote", "poster", "book cover", "album cover",
        )
    ):
        return True
    return False


def score_candidate(
    candidate: dict[str, Any],
    query: str,
    beat: dict[str, Any],
    used_ids: set[int],
    topic: str = "",
) -> float:
    if candidate.get("pageid") in used_ids:
        return -999.0
    if candidate_is_forbidden(candidate, beat):
        return -999.0

    haystack = _visual_text(candidate)
    candidate_set = _meaningful_visual_tokens(haystack)
    title_set = _meaningful_visual_tokens(
        str(candidate.get("title", ""))
    )
    topic_set = _meaningful_visual_tokens(
        core_research_query(topic)
    )
    beat_set = _meaningful_visual_tokens(" ".join([
        query,
        str(beat.get("visual_contract", "")),
        " ".join(str(item) for item in beat.get("must_show", [])),
        " ".join(str(item) for item in beat.get("sync_keywords", [])),
    ]))

    beat_overlap = len(candidate_set & beat_set)
    title_beat_overlap = len(title_set & beat_set)
    topic_overlap = len(candidate_set & topic_set)
    title_topic_overlap = len(title_set & topic_set)

    # A high-resolution but unrelated image must never win.
    if beat_overlap < 1:
        return -999.0
    if topic_set and topic_overlap < 1 and beat_overlap < 2:
        return -999.0

    score = (
        beat_overlap * 18.0
        + title_beat_overlap * 15.0
        + topic_overlap * 11.0
        + title_topic_overlap * 12.0
    )

    normalized_query = slugify(query)
    normalized_title = slugify(
        str(candidate.get("title", ""))
    )
    normalized_description = slugify(
        str(candidate.get("description", ""))
    )
    if normalized_query and normalized_query in normalized_title:
        score += 45
    elif normalized_query and normalized_query in normalized_description:
        score += 25

    width = int(candidate.get("width") or 0)
    height = int(candidate.get("height") or 0)
    if width >= 1400 and height >= 800:
        score += 8
    if width >= height:
        score += 4

    expected_type = str(
        beat.get("asset_type", "painting")
    ).lower()
    type_text = (_visual_text(candidate)).lower()
    type_markers = {
        "painting": ("painting", "miniature", "engraving", "illustration", "tableau"),
        "photo": ("photo", "photograph"),
        "architecture": ("wall", "fortress", "castle", "mosque", "church", "gate", "tower"),
        "artifact": ("museum", "artifact", "object", "cannon", "chain", "coin"),
    }
    if any(
        marker in type_text
        for marker in type_markers.get(expected_type, ())
    ):
        score += 10

    license_name = str(candidate.get("license", "")).lower()
    if "public domain" in license_name or "cc0" in license_name:
        score += 5
    return score


def _commons_query_variants(
    topic: str,
    beat: dict[str, Any],
) -> list[str]:
    topic_core = core_research_query(topic)
    contract = re.sub(
        r"\s+",
        " ",
        str(beat.get("visual_contract", "")),
    ).strip()
    query_en = re.sub(
        r"\s+",
        " ",
        str(beat.get("query_en", "")),
    ).strip()
    query_tr = re.sub(
        r"\s+",
        " ",
        str(beat.get("query_tr", "")),
    ).strip()

    variants = [
        query_en,
        query_tr,
        f"{topic_core} {query_en}".strip(),
        f"{topic_core} {query_tr}".strip(),
        f"{topic_core} {contract}".strip(),
        contract,
    ]
    return list(dict.fromkeys(
        value for value in variants if len(value) >= 4
    ))


def select_commons_asset(
    session: requests.Session,
    topic: str,
    beat: dict[str, Any],
    used_ids: set[int],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any], str] | None = None

    for query in _commons_query_variants(topic, beat):
        try:
            candidates = commons_candidates(
                session,
                query,
                limit=18,
            )
        except Exception as exc:
            log(f"Commons araması başarısız ({query}): {exc}")
            continue

        for candidate in candidates:
            score = score_candidate(
                candidate,
                query,
                beat,
                used_ids,
                topic,
            )
            if best is None or score > best[0]:
                best = (score, candidate, query)

        if best and best[0] >= 92:
            break

    if not best or best[0] < VISUAL_MIN_SCORE:
        return None

    selected = dict(best[1])
    selected["match_score"] = round(best[0], 2)
    selected["matched_query"] = best[2]
    selected["match_tokens"] = sorted(
        _meaningful_visual_tokens(
            _visual_text(selected)
        )
        & _meaningful_visual_tokens(" ".join([
            best[2],
            str(beat.get("visual_contract", "")),
            " ".join(str(item) for item in beat.get("must_show", [])),
        ]))
    )
    return selected


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def process_frame(source: Path, target: Path) -> None:
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    image = ImageOps.fit(image, (WIDTH, HEIGHT), Image.Resampling.LANCZOS, centering=(0.5, 0.48))
    image = ImageOps.autocontrast(image, cutoff=0.4)
    mean = ImageStat.Stat(ImageOps.grayscale(image)).mean[0]
    image = ImageEnhance.Brightness(image).enhance(max(0.92, min(1.22, 92 / max(1, mean))))
    image = ImageEnhance.Color(image).enhance(1.02)
    image = ImageEnhance.Contrast(image).enhance(1.08)
    image = ImageEnhance.Sharpness(image).enhance(1.12)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=85, threshold=3))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "JPEG", quality=94, optimize=True)


def make_local_graphic(beat: dict[str, Any], chapter_title: str, target: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (14, 20, 30))
    draw = ImageDraw.Draw(image)
    for i in range(10):
        inset = 40 + i * 28
        draw.rounded_rectangle(
            (inset, inset, WIDTH - inset, HEIGHT - inset),
            radius=48,
            outline=(120, 105, 82),
            width=2,
        )
    draw.text((120, 100), chapter_title.upper(), font=font(26, True), fill=(205, 181, 136))
    title = beat.get("local_graphic_title") or beat.get("visual_contract") or "Tarihsel Bağlam"
    title = re.sub(r"\s+", " ", str(title)).strip()
    lines: list[str] = []
    current = ""
    for word in title.upper().split():
        candidate = (current + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font(54, True))[2] < 1500:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for idx, line in enumerate(lines[:3]):
        draw.text((120, 210 + idx * 72), line, font=font(54, True), fill=(244, 238, 224))
    points = beat.get("local_graphic_points") or beat.get("must_show") or []
    y = 520
    for idx, point in enumerate(points[:4], start=1):
        draw.ellipse((130, y + 7, 158, y + 35), fill=(215, 176, 104))
        draw.text((185, y), re.sub(r"\s+", " ", str(point))[:85], font=font(30), fill=(224, 218, 205))
        y += 90
    draw.text((120, 980), "UYKU VE TARİH · KAYNAKLI ANLATIM", font=font(20, True), fill=(150, 139, 119))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "JPEG", quality=95, optimize=True)


def _wrap_text_lines(
    draw: ImageDraw.ImageDraw,
    value: str,
    used_font: ImageFont.FreeTypeFont,
    max_width: int,
    max_lines: int,
) -> list[str]:
    words = re.sub(r"\s+", " ", str(value)).strip().split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), candidate, font=used_font)
        if bbox[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
            if len(lines) >= max_lines:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    if len(lines) == max_lines and words:
        lines[-1] = lines[-1][:75].rstrip() + "…"
    return lines


def make_chapter_storyboard(
    ctx: ProjectContext,
    chapter_dir: Path,
    manifest: list[dict[str, Any]],
    chapter_title: str,
) -> Path:
    valid_items = [
        item
        for item in manifest
        if item.get("frame") and not item.get("missing")
    ]
    if not valid_items:
        raise ControlledStop(
            "Storyboard oluşturulamadı: kullanılabilir görsel yok."
        )

    columns = 3
    rows = math.ceil(len(valid_items) / columns)
    cell_w = 640
    image_h = 360
    caption_h = 150
    header_h = 110
    sheet = Image.new(
        "RGB",
        (cell_w * columns, header_h + rows * (image_h + caption_h)),
        (14, 18, 25),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (32, 24),
        f"UYKU VE TARİH · STORYBOARD · {chapter_title}",
        font=font(34, True),
        fill=(243, 236, 222),
    )
    draw.text(
        (32, 70),
        f"{len(valid_items)} gerçek görsel · V10.4 semantik kontrol",
        font=font(22),
        fill=(177, 165, 145),
    )

    for index, item in enumerate(valid_items):
        col = index % columns
        row = index // columns
        x = col * cell_w
        y = header_h + row * (image_h + caption_h)

        frame_path = ctx.root / str(item["frame"])
        with Image.open(frame_path) as raw:
            frame = ImageOps.fit(
                ImageOps.exif_transpose(raw).convert("RGB"),
                (cell_w, image_h),
                method=Image.Resampling.LANCZOS,
            )
        sheet.paste(frame, (x, y))

        draw.rectangle(
            (x, y + image_h, x + cell_w, y + image_h + caption_h),
            fill=(22, 25, 31),
        )
        beat_id = int(item.get("beat_id", index + 1))
        score = (
            item.get("asset", {}).get("match_score")
            if isinstance(item.get("asset"), dict)
            else None
        )
        label = f"SAHNE {beat_id:02d}"
        if score is not None:
            label += f" · EŞLEŞME {score}"
        draw.text(
            (x + 18, y + image_h + 12),
            label,
            font=font(20, True),
            fill=(213, 176, 105),
        )
        contract = str(
            item.get("visual_contract")
            or item.get("narration_excerpt")
            or ""
        )
        lines = _wrap_text_lines(
            draw,
            contract,
            font(21),
            cell_w - 36,
            3,
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (x + 18, y + image_h + 43 + line_index * 27),
                line,
                font=font(21),
                fill=(235, 230, 220),
            )

    target = chapter_dir / "storyboard.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, "JPEG", quality=94, optimize=True)

    ctx.deliverables.mkdir(parents=True, exist_ok=True)
    deliverable = ctx.deliverables / (
        f"storyboard-{chapter_dir.name}.jpg"
    )
    shutil.copyfile(target, deliverable)
    log(f"STORYBOARD READY: {deliverable}")
    return deliverable


def build_storyboard_index(
    ctx: ProjectContext,
    plan: dict[str, Any],
) -> Path | None:
    storyboards = sorted(
        ctx.deliverables.glob("storyboard-chapter-*.jpg")
    )
    if not storyboards:
        return None

    thumb_w, thumb_h = 640, 420
    rows = math.ceil(len(storyboards) / 2)
    canvas = Image.new(
        "RGB",
        (thumb_w * 2, 100 + rows * thumb_h),
        (12, 16, 23),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (28, 25),
        "UYKU VE TARİH · STORYBOARD İNDEKSİ",
        font=font(34, True),
        fill=(243, 236, 222),
    )

    for index, path in enumerate(storyboards):
        with Image.open(path) as raw:
            thumb = ImageOps.fit(
                raw.convert("RGB"),
                (thumb_w, thumb_h - 50),
                method=Image.Resampling.LANCZOS,
            )
        x = (index % 2) * thumb_w
        y = 100 + (index // 2) * thumb_h
        canvas.paste(thumb, (x, y))
        title = (
            plan.get("chapters", [{}])[index].get("title", "")
            if index < len(plan.get("chapters", []))
            else path.stem
        )
        draw.rectangle(
            (x, y + thumb_h - 50, x + thumb_w, y + thumb_h),
            fill=(22, 25, 31),
        )
        draw.text(
            (x + 14, y + thumb_h - 38),
            f"BÖLÜM {index + 1}: {str(title)[:55]}",
            font=font(20, True),
            fill=(220, 207, 184),
        )

    target = ctx.deliverables / "storyboard-index.jpg"
    canvas.save(target, "JPEG", quality=93, optimize=True)
    return target


def download_asset(session: requests.Session, asset: dict[str, Any], target: Path) -> None:
    response = session.get(asset["url"], timeout=60)
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    if target.stat().st_size < 10_000:
        raise ValueError("Commons görseli çok küçük veya boş.")


def collect_chapter_assets(
    session: requests.Session,
    ctx: ProjectContext,
    chapter_dir: Path,
    chapter_script: dict[str, Any],
    used_ids: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assets_dir = chapter_dir / "assets"
    frames_dir = chapter_dir / "frames"
    assets_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    credits: list[dict[str, Any]] = []
    local_graphics = 0
    missing = 0

    for beat in chapter_script["visual_beats"]:
        beat_id = int(beat["beat_id"])
        frame = frames_dir / f"beat_{beat_id:02d}.jpg"
        if frame.exists() and frame.stat().st_size > 20_000:
            manifest.append({**beat, "frame": str(frame.relative_to(ctx.root)), "checkpoint": True})
            continue

        asset_type = str(beat.get("asset_type", "painting")).lower()
        if asset_type in {"map", "timeline", "diagram"} and local_graphics < 2:
            make_local_graphic(beat, chapter_script.get("chapter_title", ""), frame)
            local_graphics += 1
            manifest.append({**beat, "frame": str(frame.relative_to(ctx.root)), "source_type": "local_graphic"})
            continue

        asset = select_commons_asset(session, ctx.topic, beat, used_ids)
        if asset is None:
            missing += 1
            manifest.append({
                **beat,
                "missing": True,
                "error": (
                    "Anlatımla yeterince eşleşen Wikimedia görseli bulunamadı. "
                    "Yerel bilgi kartı kullanılmadı."
                ),
            })
            continue

        raw = assets_dir / f"beat_{beat_id:02d}{Path(asset['url'].split('?', 1)[0]).suffix.lower()}"
        try:
            download_asset(session, asset, raw)
            process_frame(raw, frame)
        except Exception as exc:
            log(f"Commons görseli indirilemedi, beat={beat_id}: {exc}")
            missing += 1
            manifest.append({**beat, "missing": True, "error": str(exc)})
            continue

        used_ids.add(int(asset["pageid"]))
        record = {
            **beat,
            "frame": str(frame.relative_to(ctx.root)),
            "source_type": "wikimedia_commons",
            "asset": asset,
        }
        manifest.append(record)
        credits.append({
            "beat_id": beat_id,
            "title": asset.get("title", ""),
            "artist": asset.get("artist", ""),
            "license": asset.get("license", ""),
            "license_url": asset.get("license_url", ""),
            "source_page": asset.get("description_url", ""),
            "credit": asset.get("credit", ""),
        })

    missing_count = sum(1 for item in manifest if item.get("missing"))
    if missing_count > 0:
        report = chapter_dir / "ASSET_GAPS.txt"
        lines = [
            "Bütün sahnelerde gerçek ve anlatımla eşleşen görsel zorunludur. TTS ve render başlatılmadı.",
            f"Eksik beat: {missing_count}/{len(manifest)}",
            "",
        ]
        for item in manifest:
            if item.get("missing"):
                lines.append(f"Beat {item['beat_id']}: {item.get('visual_contract', '')}")
        report.write_text("\n".join(lines), encoding="utf-8")
        raise ControlledStop(report.read_text(encoding="utf-8"))

    write_json(chapter_dir / "assets-manifest.json", manifest)
    write_json(chapter_dir / "credits.json", credits)
    return manifest, credits


def write_wav(path: Path, pcm: bytes, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(pcm)


def extract_audio_pcm(response: Any) -> bytes:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline else None
            if isinstance(data, str):
                data = base64.b64decode(data)
            if isinstance(data, bytearray):
                data = bytes(data)
            if isinstance(data, bytes) and data:
                return data
    raise ValueError("TTS yanıtında ses parçası yok.")


def tts_prompt(text: str) -> str:
    return f"""
# VOICE
Use the Charon voice as a mature Turkish historical documentary narrator.

# PERFORMANCE
Natural conversational documentary pace. Warm, credible and controlled.
Do not sound sleepy, dragged out, theatrical, like a trailer or newsreader.
Use short punctuation-based pauses. Standard Turkey Turkish pronunciation.
Read the transcript exactly without adding or removing words.
No music and no sound effects.

# TRANSCRIPT
{text}
""".strip()


def is_gemini_tts_quota_error(
    exc: Exception,
) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "resource_exhausted",
            "quota exceeded",
            "generate_content_free_tier_requests",
            "generate requests per day",
            "429",
        )
    )


def synthesize_charon_chunk(client: genai.Client, text: str, target: Path) -> None:
    last_error: Exception | None = None
    for attempt, delay in enumerate((0, 25, 70), start=1):
        if delay:
            time.sleep(delay)
        try:
            log(f"Charon TTS parçası: deneme={attempt}/3")
            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=tts_prompt(text),
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
            pcm = extract_audio_pcm(response)
            if len(pcm) < 24_000:
                raise ValueError("TTS verisi çok kısa.")
            write_wav(target, pcm)
            if ffprobe_duration(target) < max(8.0, word_count(text) / 220 * 60 * 0.65):
                raise ValueError("TTS sesi kesilmiş görünüyor.")
            return
        except Exception as exc:
            last_error = exc
            target.unlink(missing_ok=True)
            log(f"Charon TTS başarısız: {exc}")
            if is_gemini_tts_quota_error(exc):
                break
    raise ProviderUnavailable(
        f"Charon TTS kotası veya servisi hazır değil: {last_error}"
    )


def concat_audio(files: list[Path], target: Path) -> None:
    command = ["ffmpeg", "-y"]
    filters = []
    labels = []
    for index, path in enumerate(files):
        command += ["-i", str(path)]
        label = f"a{index}"
        filters.append(
            f"[{index}:a]aresample=48000,aformat=sample_fmts=s16:channel_layouts=mono,asetpts=N/SR/TB[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append("".join(labels) + f"concat=n={len(files)}:v=0:a=1[joined]")
    filters.append("[joined]loudnorm=I=-17:TP=-2:LRA=7[outa]")
    command += [
        "-filter_complex", ";".join(filters), "-map", "[outa]",
        "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ]
    run(command, timeout=900)


def _charon_chunk_state(
    chapter_dir: Path,
    *,
    total_chunks: int,
    completed_chunks: list[int],
    next_chunk: int,
    status: str,
    message: str = "",
) -> None:
    write_json(
        chapter_dir / "tts-charon-checkpoint.json",
        {
            "provider": "gemini_charon",
            "voice": VOICE_NAME,
            "model": TTS_MODEL,
            "total_chunks": total_chunks,
            "completed_chunks": completed_chunks,
            "next_chunk": next_chunk,
            "status": status,
            "message": message,
            "version": VERSION,
        },
    )


def synthesize_charon_chapter(
    client: genai.Client,
    chapter_dir: Path,
    narration: str,
) -> Path:
    """
    Charon-only narration with persistent chunk checkpoints.

    Completed Charon chunks are never deleted on quota exhaustion. The next
    workflow run resumes from the first missing chunk, guaranteeing that no
    second voice can enter the chapter.
    """
    final = chapter_dir / "narration.wav"
    provider_file = chapter_dir / "tts-provider.json"
    chunks_dir = chapter_dir / "tts-chunks-charon"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Remove any legacy non-Charon output from previous experimental versions.
    legacy_provider = read_json(
        provider_file,
        {},
    ).get("provider")
    if legacy_provider and legacy_provider != "gemini_charon":
        log(
            "CHARON ONLY GUARD: eski farklı-ses çıktısı siliniyor."
        )
        final.unlink(missing_ok=True)
        provider_file.unlink(missing_ok=True)
    shutil.rmtree(
        chapter_dir / "tts-chunks-piper",
        ignore_errors=True,
    )

    chunks = balanced_text_chunks(
        narration,
        CHARON_CHUNK_TARGET_WORDS,
    )
    files: list[Path] = []
    completed: list[int] = []

    for index, chunk in enumerate(chunks, start=1):
        path = chunks_dir / f"chunk_{index:02d}.wav"

        if (
            path.exists()
            and ffprobe_duration(path) >= 8
        ):
            log(
                f"Charon checkpoint kullanıldı: "
                f"{index}/{len(chunks)}"
            )
            files.append(path)
            completed.append(index)
            continue

        path.unlink(missing_ok=True)
        _charon_chunk_state(
            chapter_dir,
            total_chunks=len(chunks),
            completed_chunks=completed,
            next_chunk=index,
            status="generating",
        )
        log(
            f"Charon TTS parçası: {index}/{len(chunks)} "
            f"(hedef yaklaşık {CHARON_CHUNK_TARGET_WORDS} kelime)"
        )

        try:
            synthesize_charon_chunk(
                client,
                chunk,
                path,
            )
        except ProviderUnavailable as exc:
            path.unlink(missing_ok=True)
            quota = is_gemini_tts_quota_error(exc)
            status = (
                "quota_paused"
                if quota
                else "provider_paused"
            )
            _charon_chunk_state(
                chapter_dir,
                total_chunks=len(chunks),
                completed_chunks=completed,
                next_chunk=index,
                status=status,
                message=str(exc),
            )

            if quota:
                raise ControlledStop(
                    "Charon ücretsiz günlük TTS kotası doldu. "
                    f"Tamamlanan Charon parçaları korundu: "
                    f"{len(completed)}/{len(chunks)}. "
                    f"Sıradaki parça: {index}. "
                    "Kota yenilendikten sonra aynı project_slug ve aynı "
                    "bölümle yeniden çalıştırın; yalnızca eksik Charon "
                    "parçasından devam edilecek. Başka ses kullanılmayacak."
                ) from exc

            raise ControlledStop(
                "Charon TTS servisi geçici olarak kullanılamadı. "
                f"Tamamlanan Charon parçaları korundu: "
                f"{len(completed)}/{len(chunks)}. "
                f"Sıradaki parça: {index}. "
                "Aynı project_slug ile yeniden çalıştırıldığında kaldığı "
                "yerden devam edilir. Başka ses kullanılmayacak. "
                f"Teknik hata: {exc}"
            ) from exc

        files.append(path)
        completed.append(index)
        _charon_chunk_state(
            chapter_dir,
            total_chunks=len(chunks),
            completed_chunks=completed,
            next_chunk=index + 1,
            status="chunk_ready",
        )

    concat_audio(files, final)
    actual = ffprobe_duration(final)
    minimum = max(
        60.0,
        word_count(narration) / 225 * 60 * 0.65,
    )
    if actual < minimum:
        final.unlink(missing_ok=True)
        _charon_chunk_state(
            chapter_dir,
            total_chunks=len(chunks),
            completed_chunks=completed,
            next_chunk=1,
            status="concat_invalid",
            message=(
                f"Birleşik ses kısa: {actual:.2f}s / "
                f"minimum {minimum:.2f}s"
            ),
        )
        raise ControlledStop(
            "Birleşik Charon sesi beklenenden kısa göründü. "
            "Tek tek Charon parçaları korunuyor; final ses sonraki "
            "çalışmada yeniden birleştirilecek."
        )

    write_json(
        provider_file,
        {
            "provider": "gemini_charon",
            "voice": VOICE_NAME,
            "model": TTS_MODEL,
            "chunk_count": len(files),
            "charon_only": True,
            "audio_seconds": round(actual, 2),
        },
    )
    _charon_chunk_state(
        chapter_dir,
        total_chunks=len(chunks),
        completed_chunks=completed,
        next_chunk=len(chunks) + 1,
        status="complete",
    )
    return final


def synthesize_chapter(
    client: genai.Client,
    chapter_dir: Path,
    narration: str,
) -> tuple[Path, str]:
    final = chapter_dir / "narration.wav"
    provider_file = chapter_dir / "tts-provider.json"
    provider_info = read_json(
        provider_file,
        {},
    )

    if (
        final.exists()
        and ffprobe_duration(final) > 120
        and provider_info.get("provider") == "gemini_charon"
        and provider_info.get("voice") == VOICE_NAME
    ):
        log("Charon final ses checkpointi kullanıldı.")
        return final, "gemini_charon"

    log(
        "TTS PROVIDER LOCK: yalnızca Gemini Charon kullanılacak."
    )
    return (
        synthesize_charon_chapter(
            client,
            chapter_dir,
            narration,
        ),
        "gemini_charon",
    )


def detect_silence_points(audio: Path) -> list[float]:
    result = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(audio),
            "-af", "silencedetect=n=-38dB:d=0.18", "-f", "null", "-",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    starts = [float(x) for x in re.findall(r"silence_start: ([0-9.]+)", result.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([0-9.]+)", result.stderr)]
    points = []
    for start, end in zip(starts, ends):
        points.append((start + end) / 2)
    return points


def allocate_durations(beats: list[dict[str, Any]], total: float, audio: Path) -> list[float]:
    weights = [max(1, word_count(str(beat.get("narration_excerpt", "")))) for beat in beats]
    total_weight = sum(weights)
    durations = [total * w / total_weight for w in weights]
    boundaries = []
    cursor = 0.0
    for duration in durations[:-1]:
        cursor += duration
        boundaries.append(cursor)
    silence_points = detect_silence_points(audio)
    aligned = []
    previous = 0.0
    for boundary in boundaries:
        nearby = [p for p in silence_points if abs(p - boundary) <= 2.5 and p > previous + 5]
        chosen = min(nearby, key=lambda p: abs(p - boundary)) if nearby else boundary
        aligned.append(chosen)
        previous = chosen
    points = [0.0, *aligned, total]
    result = [max(2.0, points[i + 1] - points[i]) for i in range(len(points) - 1)]
    difference = total - sum(result)
    result[-1] += difference
    return result


def audio_stream_info(path: Path) -> dict[str, Any]:
    result = run([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,channels,sample_rate,duration",
        "-of",
        "json",
        str(path),
    ], timeout=60)
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        return {}
    return streams[0]


def audio_volume_info(path: Path) -> dict[str, float]:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-vn",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=600,
    )
    combined = process.stdout + "\n" + process.stderr

    def parse(label: str, fallback: float) -> float:
        match = re.search(
            rf"{re.escape(label)}:\s*(-?[0-9.]+)\s*dB",
            combined,
        )
        return float(match.group(1)) if match else fallback

    return {
        "mean_volume_db": parse("mean_volume", -99.0),
        "max_volume_db": parse("max_volume", -99.0),
    }


def verify_final_audio(path: Path) -> dict[str, Any]:
    stream = audio_stream_info(path)
    if not stream:
        raise ControlledStop(
            "Final videoda ses kanalı bulunamadı."
        )

    volume = audio_volume_info(path)
    duration = float(stream.get("duration") or ffprobe_duration(path))
    if duration < 30:
        raise ControlledStop(
            f"Final ses kanalı çok kısa: {duration:.2f}s"
        )
    if volume["mean_volume_db"] < -45:
        raise ControlledStop(
            "Final ses kanalı teknik olarak mevcut fakat sessiz veya "
            f"duyulamayacak kadar düşük: {volume['mean_volume_db']:.1f} dB"
        )
    if volume["max_volume_db"] < -30:
        raise ControlledStop(
            "Final ses tepe seviyesi sessizliğe çok yakın: "
            f"{volume['max_volume_db']:.1f} dB"
        )

    report = {
        "codec": stream.get("codec_name"),
        "channels": int(stream.get("channels") or 0),
        "sample_rate": int(stream.get("sample_rate") or 0),
        "duration": round(duration, 3),
        **volume,
        "verified": True,
    }
    return report


def make_audio_preview(
    final_video: Path,
    target: Path,
    seconds: int = 30,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run([
        "ffmpeg",
        "-y",
        "-i",
        str(final_video),
        "-vn",
        "-t",
        str(seconds),
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        str(target),
    ], timeout=180)


def render_still_clip(frame: Path, seconds: float, target: Path) -> None:
    run([
        "ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS),
        "-t", f"{seconds:.3f}", "-i", str(frame),
        "-vf", f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},setsar=1,format=yuv420p",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
        "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
        "-t", f"{seconds:.3f}", str(target),
    ], timeout=600)


def write_concat_manifest(files: list[Path], target: Path) -> None:
    lines = []
    for path in files:
        safe = path.resolve().as_posix().replace("'", r"'\''")
        lines.append(f"file '{safe}'")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_chapter(ctx: ProjectContext, chapter_dir: Path, script: dict[str, Any], manifest: list[dict[str, Any]], audio: Path) -> Path:
    target = chapter_dir / "chapter.mp4"
    if target.exists() and ffprobe_duration(target) > 120:
        return target
    duration = ffprobe_duration(audio)
    durations = allocate_durations(script["visual_beats"], duration, audio)
    clips_dir = chapter_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    timeline: list[dict[str, Any]] = []
    cursor = 0.0
    for item, seconds in zip(manifest, durations):
        if item.get("missing"):
            raise ControlledStop("Eksik görsel varken render başlatılamaz.")
        frame = ctx.root / item["frame"]
        clip = clips_dir / f"beat_{int(item['beat_id']):02d}.mp4"
        if not clip.exists() or abs(ffprobe_duration(clip) - seconds) > 0.5:
            render_still_clip(frame, seconds, clip)
        clips.append(clip)
        timeline.append({
            "beat_id": item["beat_id"],
            "start": round(cursor, 3),
            "end": round(cursor + seconds, 3),
            "frame": item["frame"],
            "narration_excerpt": item.get("narration_excerpt", ""),
        })
        cursor += seconds
    write_json(chapter_dir / "timeline.json", timeline)
    manifest_file = chapter_dir / "clips.txt"
    write_concat_manifest(clips, manifest_file)
    visual = chapter_dir / "visual-only.mp4"
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest_file),
        "-c", "copy", str(visual),
    ], timeout=1200)
    run([
        "ffmpeg", "-y", "-i", str(visual), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-profile:a", "aac_low",
        "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", "-movflags", "+faststart", str(target),
    ], timeout=1200)
    return target


def combine_chapters(ctx: ProjectContext, chapter_files: list[Path]) -> Path:
    ctx.deliverables.mkdir(parents=True, exist_ok=True)
    target = ctx.deliverables / f"{ctx.slug}-v10-longform.mp4"
    manifest = ctx.root / "chapters-final.txt"
    write_concat_manifest(chapter_files, manifest)
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-profile:a", "aac_low",
        "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(target),
    ], timeout=3600)
    return target


def build_credits(ctx: ProjectContext, plan: dict[str, Any], all_credits: list[dict[str, Any]], research: dict[str, Any]) -> None:
    ctx.deliverables.mkdir(parents=True, exist_ok=True)
    lines = [
        plan.get("video_description", ""),
        "",
        "GÖRSEL KAYNAKLARI VE LİSANSLAR",
        "",
    ]
    for index, item in enumerate(all_credits, start=1):
        lines.append(
            f"{index}. {item.get('title', '')} — {item.get('artist') or 'Yükleyen/üretici belirtilmemiş'} "
            f"— {item.get('license', '')} — {item.get('source_page', '')}"
        )
    lines += ["", "ARAŞTIRMA KAYNAKLARI", ""]
    for source in research.get("sources", []):
        lines.append(f"- {source.get('title', '')}: {source.get('url', '')}")
    (ctx.deliverables / "youtube-description.txt").write_text("\n".join(lines), encoding="utf-8")
    write_json(ctx.deliverables / "credits.json", all_credits)


def state(ctx: ProjectContext) -> dict[str, Any]:
    return read_json(ctx.state_file, {
        "version": VERSION,
        "topic": ctx.topic,
        "mode": ctx.mode,
        "target_minutes": ctx.target_minutes,
        "chapter_count": ctx.chapter_count,
        "chapters": {},
        "status": "created",
    })


def save_state(ctx: ProjectContext, data: dict[str, Any]) -> None:
    data["version"] = VERSION
    data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(ctx.state_file, data)


def reset_generated_chapters_for_new_plan(
    ctx: ProjectContext,
    st: dict[str, Any],
    reason: str,
) -> None:
    if ctx.chapters_dir.exists():
        shutil.rmtree(
            ctx.chapters_dir,
            ignore_errors=True,
        )
    ctx.chapters_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    for filename in (
        "used-commons-pageids.json",
        "credits.json",
    ):
        (ctx.root / filename).unlink(
            missing_ok=True,
        )
    st["chapters"] = {}
    st["status"] = "plan_regenerated"
    write_json(
        ctx.root / "V10_2_YENIDEN_PLANLAMA.json",
        {
            "reason": reason,
            "version": VERSION,
            "message": (
                "Eski konu dışı araştırmaya bağlı üretilen bölüm "
                "checkpointleri temizlendi."
            ),
        },
    )


def create_context(args: argparse.Namespace) -> ProjectContext:
    chapter_count = 1 if args.mode == "pilot" else args.chapters
    target_minutes = args.pilot_minutes if args.mode == "pilot" else args.target_minutes
    slug = slugify(args.project_slug or args.topic)
    root = PROJECTS / slug
    root.mkdir(parents=True, exist_ok=True)
    (root / "chapters").mkdir(parents=True, exist_ok=True)
    (root / "deliverables").mkdir(parents=True, exist_ok=True)
    return ProjectContext(slug, args.topic, args.mode, target_minutes, chapter_count, root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Uyku ve Tarih V10 Longform Core")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--project-slug", default="")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--target-minutes", type=int, default=60)
    parser.add_argument("--pilot-minutes", type=int, default=10)
    parser.add_argument("--chapters", type=int, default=6)
    parser.add_argument("--beats-per-chapter", type=int, default=12)
    parser.add_argument("--chapter", type=int, default=0, help="0=tümü, diğer değer yalnızca o bölüm")
    parser.add_argument("--stage", choices=["all", "plan", "scripts", "assets", "tts", "render"], default="all")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY bulunamadı.")

    ctx = create_context(args)
    st = state(ctx)
    save_state(ctx, st)
    client = genai.Client(api_key=api_key)
    session = requests_session()

    log("=" * 72)
    log(f"UYKU VE TARİH V10.4 SEMANTIC STORYBOARD CORE {VERSION}")
    log(f"Proje: {ctx.slug} | mod={ctx.mode} | hedef={ctx.target_minutes} dk")
    log("Checkpoint/resume aktif. Tamamlanan dosyalar yeniden üretilmez.")
    log("Resilient script: bölüm metni segmentlere ayrılır ve her segment kaydedilir.")
    log("TTS kilidi: yalnızca Charon; farklı ses ve fallback kapalı.")
    log("Görsel kilidi: konu dışı Commons sonuçları ve hayvan/modern alakasız kareler reddedilir.")
    log("Storyboard ve final ses doğrulaması zorunlu.")
    log(f"Charon parça hedefi: yaklaşık {CHARON_CHUNK_TARGET_WORDS} kelime.")
    log(f"Araştırma çekirdek sorgusu: {core_research_query(ctx.topic)}")
    log("=" * 72)

    try:
        research_file = ctx.root / "research.json"
        research = read_json(research_file)
        research_rebuilt = False

        if not research_is_usable(
            research,
            ctx.topic,
        ):
            if research is not None:
                log(
                    "RESEARCH QUALITY GUARD: eski kaynaklar konu dışı. "
                    "Tek kelimelik yanlış eşleşmeler reddediliyor."
                )
            research = research_topic(
                session,
                ctx.topic,
                [],
            )
            write_json(
                research_file,
                research,
            )
            research_rebuilt = True

        fingerprint = research_fingerprint(
            research,
        )
        plan_file = ctx.root / "story-plan.json"
        plan = read_json(plan_file)
        plan_invalid = (
            not isinstance(plan, dict)
            or plan.get("research_fingerprint") != fingerprint
            or plan.get("generator_version") != VERSION
        )

        if plan_invalid:
            payload, model = generate_json(
                client,
                story_plan_prompt(
                    ctx.topic,
                    ctx.target_minutes,
                    ctx.chapter_count,
                    research,
                ),
                max_tokens=9000,
            )
            plan = validate_plan(
                payload,
                ctx.chapter_count,
            )
            plan["text_model"] = model
            plan["research_fingerprint"] = fingerprint
            plan["generator_version"] = VERSION
            write_json(
                plan_file,
                plan,
            )
            reset_generated_chapters_for_new_plan(
                ctx,
                st,
                (
                    "Araştırma kaynakları düzeltildi."
                    if research_rebuilt
                    else "Plan V10.2 araştırma parmak iziyle yenilendi."
                ),
            )
            save_state(ctx, st)
        if args.stage == "plan":
            st["status"] = "plan_ready"
            save_state(ctx, st)
            return

        used_ids = set(read_json(ctx.root / "used-commons-pageids.json", []))
        all_credits: list[dict[str, Any]] = []
        chapter_files: list[Path] = []

        for chapter in plan["chapters"]:
            index = int(chapter["chapter_index"])
            if args.chapter and index != args.chapter:
                continue
            chapter_dir = ctx.chapters_dir / f"chapter-{index:02d}"
            chapter_dir.mkdir(parents=True, exist_ok=True)
            chapter_state = st["chapters"].setdefault(str(index), {})
            log(f"--- BÖLÜM {index}/{ctx.chapter_count}: {chapter.get('title', '')} ---")

            chapter_research_file = chapter_dir / "research.json"
            chapter_research = read_json(chapter_research_file)
            if not research_is_usable(
                chapter_research,
                ctx.topic,
            ):
                chapter_research = research_topic(
                    session,
                    ctx.topic,
                    list(chapter.get("research_queries", [])),
                )
                write_json(
                    chapter_research_file,
                    chapter_research,
                )

            script_file = chapter_dir / "script.json"
            script = read_json(script_file)
            if script is None:
                script = build_chapter_script_resilient(
                    client,
                    ctx.topic,
                    plan,
                    chapter,
                    chapter_research,
                    args.beats_per_chapter,
                    chapter_dir,
                )
                write_json(script_file, script)
            chapter_state["script"] = "ready"
            save_state(ctx, st)
            if args.stage == "scripts":
                continue

            manifest_file = chapter_dir / "assets-manifest.json"
            manifest = read_json(manifest_file)
            credits = read_json(chapter_dir / "credits.json", [])
            if manifest is None:
                manifest, credits = collect_chapter_assets(
                    session,
                    ctx,
                    chapter_dir,
                    script,
                    used_ids,
                )
                write_json(ctx.root / "used-commons-pageids.json", sorted(used_ids))
            all_credits.extend(credits)
            storyboard_file = make_chapter_storyboard(
                ctx,
                chapter_dir,
                manifest,
                str(script.get("chapter_title", chapter.get("title", ""))),
            )
            chapter_state["assets"] = "ready"
            chapter_state["storyboard"] = str(
                storyboard_file.relative_to(ctx.root)
            )
            save_state(ctx, st)
            if args.stage == "assets":
                continue

            audio, used_tts_provider = synthesize_chapter(
                client,
                chapter_dir,
                script["narration"],
            )
            chapter_state["tts"] = "ready"
            chapter_state["tts_provider"] = used_tts_provider
            chapter_state["audio_seconds"] = round(ffprobe_duration(audio), 2)
            save_state(ctx, st)
            if args.stage == "tts":
                continue

            chapter_video = render_chapter(ctx, chapter_dir, script, manifest, audio)
            chapter_files.append(chapter_video)
            chapter_state["render"] = "ready"
            chapter_state["video_seconds"] = round(ffprobe_duration(chapter_video), 2)
            save_state(ctx, st)

        if args.stage in {"scripts", "assets", "tts"}:
            st["status"] = f"{args.stage}_ready"
            save_state(ctx, st)
            return

        if len(chapter_files) != ctx.chapter_count:
            chapter_files = [
                ctx.chapters_dir / f"chapter-{index:02d}" / "chapter.mp4"
                for index in range(1, ctx.chapter_count + 1)
            ]
        if not all(path.exists() for path in chapter_files):
            raise ControlledStop("Bütün bölüm videoları hazır değil; final birleştirme yapılmadı.")

        final = combine_chapters(ctx, chapter_files)
        build_credits(ctx, plan, all_credits, research)
        build_storyboard_index(ctx, plan)
        audio_report = verify_final_audio(final)
        write_json(
            ctx.deliverables / "audio-quality-report.json",
            audio_report,
        )
        make_audio_preview(
            final,
            ctx.deliverables / "charon-audio-preview-30s.mp3",
        )
        final_seconds = ffprobe_duration(final)
        st["status"] = "complete"
        st["audio_verified"] = True
        st["audio_report"] = audio_report
        st["final_video"] = str(final.relative_to(ctx.root))
        st["final_seconds"] = round(final_seconds, 2)
        save_state(ctx, st)
        (ctx.deliverables / "DEVAM_ETME_RAPORU.txt").unlink(
            missing_ok=True
        )
        log("=" * 72)
        log(f"V10 LONGFORM COMPLETE: {final}")
        log(f"Süre: {final_seconds / 60:.2f} dakika")
        log("=" * 72)

    except ControlledStop as exc:
        st["status"] = "paused"
        st["last_message"] = str(exc)
        save_state(ctx, st)
        report = ctx.deliverables / "DEVAM_ETME_RAPORU.txt"
        report.write_text(
            "Üretim kontrollü olarak durduruldu. Tamamlanan checkpointler korunuyor.\n\n"
            + str(exc)
            + "\n\nAynı project_slug ile workflow'u yeniden çalıştırın.\n",
            encoding="utf-8",
        )
        log("CONTROLLED PAUSE: " + str(exc))
        return


if __name__ == "__main__":
    main()
