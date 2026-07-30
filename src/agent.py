from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
import requests
import yaml
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont, ImageOps

VERSION = "11.1.5"


class ControlledPause(RuntimeError):
    pass


class QualityGateError(RuntimeError):
    pass


class ProviderUnavailable(RuntimeError):
    pass


def log(message: str) -> None:
    print(time.strftime("[%H:%M:%S]"), message, flush=True)


def slugify(value: str) -> str:
    table = str.maketrans({
        "ç": "c", "Ç": "c", "ğ": "g", "Ğ": "g",
        "ı": "i", "İ": "i", "ö": "o", "Ö": "o",
        "ş": "s", "Ş": "s", "ü": "u", "Ü": "u",
    })
    value = value.translate(table).lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-")[:64] or "proje"


def project_id(topic: str, minutes: int) -> str:
    digest = hashlib.sha256(f"{topic.strip()}|{minutes}".encode()).hexdigest()[:8]
    return f"{slugify(topic)}-{minutes}m-{digest}"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run(command: list[str], timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    log("RUN: " + " ".join(command))
    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Komut başarısız ({result.returncode}): {' '.join(command)}\n"
            f"STDOUT:\n{result.stdout[-5000:]}\nSTDERR:\n{result.stderr[-5000:]}"
        )
    return result


def ffprobe_duration(path: Path) -> float:
    if not path.exists():
        return 0.0
    result = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], 60)
    return float(result.stdout.strip() or 0)


def word_count(value: str) -> int:
    return len(re.findall(r"\b[\wÇĞİÖŞÜçğıöşü'’\-]+\b", value))


def quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(token in text for token in (
        "429", "resource_exhausted", "quota", "daily free allocation",
        "free_tier_requests", "requests per day", "account limited",
    ))


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class Gemini:
    def __init__(self, config: dict[str, Any]):
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ProviderUnavailable("GEMINI_API_KEY eksik.")
        self.client = genai.Client(api_key=key)
        self.models = list(config["text_models"])
        self.tts_model = str(config["tts_model"])
        self.voice = str(config["voice"])
        self.retries = int(config.get("json_retries", 3))

    @staticmethod
    def parse_json(raw: str) -> dict[str, Any]:
        """Parse one usable JSON object even with trailing model output."""
        clean = str(raw).replace("\ufeff", "").strip()
        clean = re.sub(
            r"^```(?:json)?\s*",
            "",
            clean,
            flags=re.IGNORECASE,
        )
        clean = re.sub(r"\s*```$", "", clean)
        if not clean:
            raise ValueError("Boş JSON yanıtı.")

        try:
            value = json.loads(clean)
            if not isinstance(value, dict):
                raise ValueError("JSON kökü nesne olmalı.")
            return value
        except json.JSONDecodeError as strict_error:
            decoder = json.JSONDecoder()
            recovered: list[tuple[int, int, dict[str, Any]]] = []
            for start, char in enumerate(clean):
                if char != "{":
                    continue
                try:
                    value, consumed = decoder.raw_decode(clean[start:])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    recovered.append((start, consumed, value))

            if recovered:
                first_start = min(item[0] for item in recovered)
                choices = [
                    item for item in recovered if item[0] == first_start
                ]
                _, consumed, value = max(
                    choices,
                    key=lambda item: item[1],
                )
                trailing = clean[first_start + consumed:].strip()
                log(
                    "JSON RECOVERY: geçerli ilk nesne kurtarıldı; "
                    f"sonraki_fazla_karakter={len(trailing)}"
                )
                return value

            repaired = re.sub(r",\s*([}\]])", r"\1", clean)
            if repaired != clean:
                value = json.loads(repaired)
                if isinstance(value, dict):
                    log("JSON RECOVERY: sondaki virgüller temizlendi.")
                    return value
            raise strict_error

    def json(self, system: str, prompt: str, temperature: float = 0.25) -> tuple[dict[str, Any], str]:
        last_error: Exception | None = None
        for model in self.models:
            for attempt in range(1, self.retries + 1):
                try:
                    log(f"Gemini JSON: {model}, deneme={attempt}/{self.retries}")
                    strict_suffix = (
                        ""
                        if attempt == 1
                        else "\n\nÇIKTI KURALI: Yalnızca tek bir geçerli JSON "
                        "nesnesi döndür. İkinci JSON nesnesi, açıklama, "
                        "markdown ve kod bloğu ekleme."
                    )
                    response = self.client.models.generate_content(
                        model=model,
                        contents=f"{system}\n\n{prompt}{strict_suffix}",
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=temperature,
                        ),
                    )
                    raw = getattr(response, "text", None)
                    if not raw:
                        raise ValueError("Boş Gemini yanıtı.")
                    return self.parse_json(raw), model
                except Exception as exc:
                    last_error = exc
                    if quota_error(exc):
                        break
                    time.sleep(attempt * 3)
        if last_error and quota_error(last_error):
            raise ControlledPause(f"Gemini metin kotası doldu: {last_error}") from last_error
        raise ProviderUnavailable(f"Gemini JSON başarısız: {last_error}")

    def tts_pcm(self, text: str, instruction: str) -> bytes:
        try:
            response = self.client.models.generate_content(
                model=self.tts_model,
                contents=f"{instruction}\n\nOKUNACAK METİN:\n{text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.voice
                            )
                        )
                    ),
                ),
            )
            for candidate in getattr(response, "candidates", None) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", None) or []:
                    inline = getattr(part, "inline_data", None)
                    data = getattr(inline, "data", None)
                    if data:
                        return data
            raise ValueError("Charon ses verisi bulunamadı.")
        except Exception as exc:
            if quota_error(exc):
                raise ControlledPause(f"Charon ücretsiz kotası doldu: {exc}") from exc
            raise ProviderUnavailable(f"Charon başarısız: {exc}") from exc


WIKI = {
    "tr": "https://tr.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}


def wiki_request(
    session: requests.Session,
    endpoint: str,
    params: dict[str, Any],
    *,
    attempts: int = 5,
) -> requests.Response:
    """Wikimedia-friendly request with Retry-After and exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                endpoint,
                params={**params, "maxlag": 5},
                timeout=45,
            )
            if response.status_code == 429:
                retry_header = response.headers.get("Retry-After", "").strip()
                try:
                    wait = float(retry_header)
                except ValueError:
                    wait = min(35.0, 3.5 * (2 ** (attempt - 1)))
                log(
                    "Wikipedia 429; yeniden denenecek: "
                    f"deneme={attempt}/{attempts}, bekleme={wait:.1f}s"
                )
                time.sleep(wait)
                continue
            if response.status_code in {500, 502, 503, 504}:
                wait = min(30.0, 2.5 * (2 ** (attempt - 1)))
                log(
                    "Wikipedia geçici sunucu hatası; yeniden denenecek: "
                    f"HTTP={response.status_code}, bekleme={wait:.1f}s"
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= attempts:
                break
            wait = min(30.0, 2.5 * (2 ** (attempt - 1)))
            log(
                "Wikipedia bağlantısı yeniden denenecek: "
                f"deneme={attempt}/{attempts}, bekleme={wait:.1f}s, hata={exc}"
            )
            time.sleep(wait)
    raise ProviderUnavailable(
        f"Wikipedia isteği {attempts} denemede tamamlanamadı: {last_error}"
    )


def wiki_search(
    session: requests.Session,
    language: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Search and fetch extracts in one MediaWiki request."""
    endpoint = WIKI[language]
    response = wiki_request(
        session,
        endpoint,
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 0,
            "gsrlimit": max(2, min(limit, 5)),
            "prop": "extracts|info",
            "explaintext": 1,
            "exchars": 6000,
            "inprop": "url",
            "format": "json",
            "formatversion": 2,
        },
    )
    pages = response.json().get("query", {}).get("pages", [])
    return [{
        "language": language,
        "title": page.get("title", ""),
        "url": page.get("fullurl", ""),
        "extract": " ".join(page.get("extract", "").split()),
    } for page in pages]


def research_terms(value: str) -> set[str]:
    ignored = {
        "ve", "ile", "bir", "bu", "şu", "icin", "için", "nasil", "nasıl",
        "nedir", "kimdir", "nerede", "neden", "tarihi", "history", "the",
        "and", "of", "in", "how", "what", "why",
    }
    return {
        token
        for token in slugify(value).replace("-", " ").split()
        if len(token) >= 3 and token not in ignored
    }


def lexical_source_score(
    source: dict[str, Any],
    reference: str,
) -> int:
    reference_terms = research_terms(reference)
    title_terms = research_terms(str(source.get("title", "")))
    extract_terms = research_terms(str(source.get("extract", ""))[:1500])
    title_overlap = len(reference_terms & title_terms)
    extract_overlap = len(reference_terms & extract_terms)
    return (
        title_overlap * 50
        + min(35, extract_overlap * 5)
        + (10 if len(str(source.get("extract", ""))) >= 600 else 0)
    )


def build_research(
    topic: str,
    config: dict[str, Any],
    gemini: Gemini,
) -> dict[str, Any]:
    resolver, resolver_model = gemini.json(
        "Sen evrensel bir tarih araştırma editörüsün.",
        f'''KONU: {topic}
Geçerli JSON üret:
{{
  "canonical_tr": "doğru kısa Türkçe ad",
  "canonical_en": "uluslararası veya İngilizce ad",
  "queries_tr": ["en fazla 4 somut arama sorgusu"],
  "queries_en": ["at most 4 concrete search queries"],
  "scope": "konunun sınırı ve yanlış benzer eşleşmeler"
}}
Kurallar:
- Soru eklerini ve gündelik ifadeleri kanonik addan çıkar.
- Özel isimleri, savaşları, eserleri, kişileri ve olayları doğru çöz.
- Konuya özel sabit kod kullanma.
- Aynı anlamı taşıyan gereksiz sorgular üretme.''',
        0.15,
    )
    queries = {
        "tr": list(dict.fromkeys([
            str(resolver.get("canonical_tr", topic)),
            *[str(item) for item in resolver.get("queries_tr", [])],
        ]))[:4],
        "en": list(dict.fromkeys([
            str(resolver.get("canonical_en", topic)),
            *[str(item) for item in resolver.get("queries_en", [])],
        ]))[:4],
    }

    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "UykuTarihV11.1.1/1.0 "
            "(GitHub Actions educational documentary research)"
        ),
        "Accept": "application/json",
    })

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for language in config["languages"]:
        for query_index, query in enumerate(queries[language], start=1):
            try:
                results = wiki_search(
                    session,
                    language,
                    query,
                    min(int(config["results_per_query"]), 5),
                )
            except Exception as exc:
                log(
                    f"Wikipedia sorgusu kullanılamadı "
                    f"({language}/{query}): {exc}"
                )
                continue
            for item in results:
                key = item.get("url") or f"{language}:{item['title']}"
                if key not in seen and len(item.get("extract", "")) >= 250:
                    seen.add(key)
                    item["matched_query"] = query
                    candidates.append(item)
            if query_index < len(queries[language]):
                time.sleep(1.4)

    if not candidates:
        raise QualityGateError(
            "Araştırma adayı bulunamadı. Wikipedia geçici olarak "
            "kullanılamadıysa checkpoint korunarak sonraki çalışmada yeniden denenir."
        )

    reference = " | ".join((
        topic,
        str(resolver.get("canonical_tr", "")),
        str(resolver.get("canonical_en", "")),
        str(resolver.get("scope", "")),
    ))
    for item in candidates:
        item["lexical_score"] = lexical_source_score(item, reference)
    candidates.sort(
        key=lambda item: int(item.get("lexical_score", 0)),
        reverse=True,
    )

    compact = [{
        "id": index,
        "title": item["title"],
        "language": item["language"],
        "matched_query": item.get("matched_query", ""),
        "extract": item["extract"][:900],
    } for index, item in enumerate(candidates[:30], 1)]

    selection, selection_model = gemini.json(
        "Sen kaynak alaka denetçisisin. Yalnız verilen adayları değerlendir.",
        f'''KONU: {topic}
KANONİK KONU: {resolver}
ADAYLAR: {compact}

Geçerli JSON:
{{
  "selected_ids": [1, 2, 3, 4],
  "research_summary": "yalnız seçilen kaynaklara dayalı özet",
  "rejected_examples": ["yanlış eşleşmelerin kısa nedenleri"]
}}

Kurallar:
- 4 ile 12 arasında geçerli aday kimliği seç.
- Benzer isimli farklı kişi veya olayları reddet.
- Başlıkta tek bir ortak kelime bulunması yeterli değildir.''',
        0.05,
    )

    valid_ids = set(range(1, len(compact) + 1))
    wanted: list[int] = []
    for value in selection.get("selected_ids", []):
        try:
            candidate_id = int(value)
        except (TypeError, ValueError):
            continue
        if candidate_id in valid_ids and candidate_id not in wanted:
            wanted.append(candidate_id)

    minimum = int(config["minimum_sources"])
    if len(wanted) < minimum:
        log(
            "Gemini kaynak listesi eksik döndü; kanonik konuya göre "
            "deterministik kaynak tamamlama uygulanıyor."
        )
        for candidate_id in range(1, len(compact) + 1):
            if candidate_id not in wanted:
                wanted.append(candidate_id)
            if len(wanted) >= minimum:
                break

    selected = [
        candidates[candidate_id - 1]
        for candidate_id in wanted[:12]
    ]
    if len(selected) < minimum:
        raise QualityGateError(
            f"Alakalı kaynak yetersiz: {len(selected)}/{minimum}"
        )

    return {
        "topic": topic,
        "resolver": resolver,
        "resolver_model": resolver_model,
        "selection_model": selection_model,
        "research_summary": selection.get("research_summary", ""),
        "rejected_examples": selection.get("rejected_examples", []),
        "sources": selected,
        "research_version": VERSION,
    }




SYSTEM = '''Sen kıdemli tarih belgeseli yazarı ve editörüsün. Yalnız verilen
araştırma çerçevesine bağlı kal. Kaynakta olmayan kesin ayrıntı uydurma.
Olayları neden-sonuç ilişkisiyle anlat. Gereksiz şiirsel betimleme, tekrar,
madde listesi ve genel tarih dersi kullanma. Her paragraf somut biçimde
görselleştirilebilir kişi, eylem, nesne veya mekân taşısın.'''


def source_context(research: dict[str, Any], limit: int = 18000) -> str:
    text = []
    for i, source in enumerate(research["sources"], 1):
        text.append(f"[{i}] {source['title']} — {source['url']}\n{source['extract'][:2600]}")
    return "\n\n".join(text)[:limit]


def duration_profile(minutes: int, config: dict[str, Any]) -> dict[str, Any]:
    profiles = config.get("project", {}).get("duration_profiles", {})
    selected = profiles.get(str(minutes))
    if not isinstance(selected, dict):
        if minutes <= 5:
            selected = {"chapters": 1, "scenes_per_chapter": 10, "script_parts": 2}
        elif minutes <= 10:
            selected = {"chapters": 2, "scenes_per_chapter": 10, "script_parts": 2}
        elif minutes <= 30:
            selected = {"chapters": 4, "scenes_per_chapter": 15, "script_parts": 3}
        else:
            selected = {"chapters": 6, "scenes_per_chapter": 18, "script_parts": 4}

    wpm = int(config["project"]["narration_words_per_minute"])
    chapters = max(1, int(selected["chapters"]))
    scenes = max(4, int(selected["scenes_per_chapter"]))
    parts = max(1, int(selected["script_parts"]))
    total_words = max(300, round(minutes * wpm))
    chapter_words = max(220, round(total_words / chapters))
    part_words = max(110, round(chapter_words / parts))
    minimum_part_words = max(85, round(part_words * 0.58))
    maximum_part_words = max(minimum_part_words + 30, round(part_words * 1.28))
    soft_minimum_chapter_words = max(
        180,
        round(chapter_words * 0.88),
    )
    hard_minimum_chapter_words = max(
        150,
        round(chapter_words * 0.60),
    )
    return {
        "minutes": int(minutes),
        "chapters": chapters,
        "scenes_per_chapter": scenes,
        "script_parts": parts,
        "target_total_words": total_words,
        "target_chapter_words": chapter_words,
        "target_part_words": part_words,
        "minimum_part_words": minimum_part_words,
        "maximum_part_words": maximum_part_words,
        "soft_minimum_chapter_words": soft_minimum_chapter_words,
        "hard_minimum_chapter_words": hard_minimum_chapter_words,
        "profile_version": VERSION,
    }


def profile_matches(value: Any, expected: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    keys = (
        "minutes", "chapters", "scenes_per_chapter", "script_parts",
        "target_total_words", "target_chapter_words",
    )
    return all(value.get(key) == expected.get(key) for key in keys)


def reset_stale_story_structure(root: Path, state: dict[str, Any], reason: str) -> None:
    (root / "story_bible.json").unlink(missing_ok=True)
    shutil.rmtree(root / "chapters", ignore_errors=True)
    state["chapters"] = {}
    state["structure_reset"] = {
        "version": VERSION,
        "reason": reason,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    log(f"ADAPTIVE PROFILE RESET: {reason}")


def trim_to_sentence_word_limit(value: str, maximum_words: int) -> str:
    value = " ".join(str(value).split())
    if word_count(value) <= maximum_words:
        return value
    sentences = split_sentences(value)
    kept: list[str] = []
    for sentence in sentences:
        candidate = " ".join([*kept, sentence])
        if kept and word_count(candidate) > maximum_words:
            break
        kept.append(sentence)
        if word_count(" ".join(kept)) >= maximum_words:
            break
    trimmed = " ".join(kept).strip()
    return trimmed or " ".join(value.split()[:maximum_words]).strip()


def generate_script_part_resilient(
    topic: str,
    bible: dict[str, Any],
    research: dict[str, Any],
    chapter: dict[str, Any],
    previous_summary: str,
    existing_parts: list[str],
    part_index: int,
    part_count: int,
    target_words: int,
    minimum_words: int,
    maximum_words: int,
    gemini: Gemini,
) -> tuple[str, list[str]]:
    accumulated = ""
    models: list[str] = []
    tail = " ".join(" ".join(existing_parts).split()[-150:])

    for attempt in range(1, 4):
        if attempt == 1:
            task = (
                f"Bu bölümün {part_index}/{part_count}. parçasını "
                f"{minimum_words}-{maximum_words} Türkçe kelime arasında yaz."
            )
            continuation_context = ""
        else:
            task = (
                "Aşağıdaki kısa taslağı tekrar etmeden doğrudan devam ettir ve "
                f"toplam uzunluğu en az {minimum_words}, tercihen {target_words} "
                "kelimeye tamamla. Yalnız eklenecek devam metnini yaz."
            )
            continuation_context = f"KISA TASLAK:\n{accumulated}\n"

        payload, model = gemini.json(SYSTEM, f'''KONU: {topic}
HİKÂYE ANAYASASI: {bible}
BÖLÜM: {chapter}
ÖNCEKİ BÖLÜM ÖZETİ: {previous_summary}
ÖNCEKİ PARÇALARIN SONU: {tail}
{continuation_context}
KAYNAKLAR:\n{source_context(research, 13000)}

GÖREV: {task}
Geçerli JSON: {{"text":"yalnız anlatıcının okuyacağı akıcı metin"}}
Kurallar:
- Yeni giriş yapma; önceki akıştan devam et.
- Somut kişi, karar, olay, nesne ve mekân anlat.
- Kamera komutu, başlık, madde işareti ve kaynak numarası yazma.
- Aynı cümleyi veya olayı tekrar etme.
- Nihai parça değilse bölümü erkenden bitirme.''', 0.26)
        piece = " ".join(str(payload.get("text", "")).split())
        models.append(model)
        if piece and not (accumulated and piece in accumulated):
            accumulated = " ".join(item for item in (accumulated, piece) if item).strip()
        current_words = word_count(accumulated)
        log(
            f"Senaryo parçası kalite kontrolü: bölüm={chapter['index']} "
            f"parça={part_index}/{part_count} deneme={attempt}/3 "
            f"kelime={current_words} minimum={minimum_words}"
        )
        if current_words >= minimum_words:
            return trim_to_sentence_word_limit(accumulated, maximum_words), models

    raise QualityGateError(
        f"Bölüm {chapter['index']} parça {part_index} üç denemede kısa kaldı: "
        f"{word_count(accumulated)}/{minimum_words} kelime."
    )



def complete_chapter_narration(
    root: Path,
    topic: str,
    bible: dict[str, Any],
    research: dict[str, Any],
    chapter: dict[str, Any],
    previous_summary: str,
    narration: str,
    target_words: int,
    soft_minimum_words: int,
    hard_minimum_words: int,
    profile: dict[str, Any],
    gemini: Gemini,
) -> tuple[str, list[str], bool]:
    directory = chapter_dir(root, int(chapter["index"]))
    completion_dir = directory / "script_completion"
    completion_dir.mkdir(parents=True, exist_ok=True)

    completed_texts: list[str] = []
    models: list[str] = []

    for index in range(1, 7):
        saved = read_json(completion_dir / f"{index:02d}.json")
        if not isinstance(saved, dict):
            break
        saved_text = str(saved.get("text", "")).strip()
        if not saved_text:
            break
        completed_texts.append(saved_text)
        models.extend(
            str(model)
            for model in saved.get("models", [])
            if str(model).strip()
        )

    combined = " ".join(
        item for item in [narration, *completed_texts] if str(item).strip()
    ).strip()

    for attempt in range(len(completed_texts) + 1, 7):
        current_words = word_count(combined)
        if current_words >= soft_minimum_words:
            break

        missing = max(1, soft_minimum_words - current_words)
        requested = max(90, min(220, missing + 35))
        tail = " ".join(combined.split()[-190:])
        closing_rule = (
            "Bu ek metinle bölümü doğal bir sonuca bağla."
            if missing <= 170
            else "Bölümü henüz sonlandırmadan akışı ilerlet."
        )

        prompt = f'''KONU: {topic}
HİKÂYE ANAYASASI: {bible}
BÖLÜM: {chapter}
ÖNCEKİ BÖLÜM ÖZETİ: {previous_summary}
MEVCUT BÖLÜMÜN SONU:
{tail}

KAYNAKLAR:
{source_context(research, 12000)}

Mevcut bölüm {current_words} kelime. Kaliteli anlatı için en az
{soft_minimum_words} kelime gerekiyor. Yalnızca mevcut metnin devamını,
yaklaşık {requested} Türkçe kelime olarak yaz.

Geçerli JSON:
{{"text":"yalnız eklenecek devam metni"}}

Kurallar:
- Baştan başlama ve önceki cümleleri tekrar etme.
- Yeni bir başlık veya giriş yazma.
- Somut olay, karar, kişi, mekân ve neden-sonuç ilişkisiyle devam et.
- Kaynaklarda bulunmayan kesin ayrıntı uydurma.
- {closing_rule}
- Kamera komutu, kaynak numarası ve madde işareti yazma.'''

        payload, model = gemini.json(SYSTEM, prompt, 0.22)
        piece = " ".join(str(payload.get("text", "")).split()).strip()
        piece_words = word_count(piece)

        if piece_words < 35:
            log(
                f"Bölüm tamamlama kısa/boş: bölüm={chapter['index']} "
                f"deneme={attempt}/6 kelime={piece_words}"
            )
            continue

        write_json(
            completion_dir / f"{attempt:02d}.json",
            {
                "index": attempt,
                "text": piece,
                "words": piece_words,
                "models": [model],
                "generation_profile": profile,
            },
        )
        completed_texts.append(piece)
        models.append(model)
        combined = " ".join([combined, piece]).strip()
        log(
            f"Bölüm otomatik tamamlandı: bölüm={chapter['index']} "
            f"ek={attempt}/6 toplam_kelime={word_count(combined)} "
            f"yumuşak_hedef={soft_minimum_words}"
        )

    final_words = word_count(combined)
    below_soft = final_words < soft_minimum_words

    if final_words < hard_minimum_words:
        raise QualityGateError(
            "Bölüm metni otomatik tamamlama sonrasında da "
            f"kullanılamayacak kadar kısa: "
            f"{final_words}/{hard_minimum_words} kelime."
        )

    if below_soft:
        log(
            "FAIL-SOFT SCRIPT GUARD: bölüm yumuşak hedefin altında kaldı "
            f"({final_words}/{soft_minimum_words}) fakat güvenli alt sınırı "
            "geçtiği için üretim durdurulmadı."
        )

    maximum_words = max(
        soft_minimum_words,
        round(target_words * 1.12),
    )
    combined = trim_to_sentence_word_limit(combined, maximum_words)
    return combined, models, below_soft



def create_story_bible(topic: str, minutes: int, count: int, research: dict[str, Any], gemini: Gemini) -> dict[str, Any]:
    payload, model = gemini.json(SYSTEM, f'''KONU: {topic}
HEDEF: {minutes} dakika, {count} bölüm
ARAŞTIRMA: {research.get('research_summary', '')}
KAYNAKLAR:\n{source_context(research)}
Geçerli JSON:
{{
 "title":"video başlığı",
 "central_question":"ana soru",
 "timeline":["olay sırası"],
 "continuity_rules":["tutarlılık kuralları"],
 "visual_identity":"tek sinematik dünya",
 "chapters":[{{"index":1,"title":"ad","objective":"görev","must_cover":["olay"],"opening_bridge":"bağ","closing_bridge":"bağ"}}]
}}
Tam {count} bölüm üret; tekrar yapma.''', 0.18)
    if len(payload.get("chapters", [])) != count:
        raise QualityGateError("Hikâye anayasası bölüm sayısı hatalı.")
    payload["model"] = model
    return payload


def chapter_dir(root: Path, index: int) -> Path:
    path = root / "chapters" / f"{index:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_script_checkpoint(
    root: Path,
    topic: str,
    bible: dict[str, Any],
    research: dict[str, Any],
    chapter: dict[str, Any],
    previous_summary: str,
    target_words: int,
    part_count: int,
    profile: dict[str, Any],
    gemini: Gemini,
) -> bool:
    directory = chapter_dir(root, int(chapter["index"]))
    final = directory / "script.json"
    existing_final = read_json(final)
    if (
        isinstance(existing_final, dict)
        and profile_matches(existing_final.get("generation_profile"), profile)
        and word_count(existing_final.get("narration", "")) >= int(target_words * 0.72)
    ):
        return True
    if final.exists():
        final.unlink(missing_ok=True)

    parts_dir = directory / "script_parts"
    parts_dir.mkdir(exist_ok=True)
    parts: list[str] = []
    for index in range(1, part_count + 1):
        saved = read_json(parts_dir / f"{index:02d}.json")
        if (
            isinstance(saved, dict)
            and profile_matches(saved.get("generation_profile"), profile)
            and str(saved.get("text", "")).strip()
        ):
            parts.append(str(saved["text"]))
        else:
            for stale_index in range(index, 9):
                (parts_dir / f"{stale_index:02d}.json").unlink(missing_ok=True)
            break

    if len(parts) < part_count:
        part_index = len(parts) + 1
        generated, models = generate_script_part_resilient(
            topic,
            bible,
            research,
            chapter,
            previous_summary,
            parts,
            part_index,
            part_count,
            int(profile["target_part_words"]),
            int(profile["minimum_part_words"]),
            int(profile["maximum_part_words"]),
            gemini,
        )
        write_json(parts_dir / f"{part_index:02d}.json", {
            "part": part_index,
            "part_count": part_count,
            "text": generated,
            "words": word_count(generated),
            "models": models,
            "generation_profile": profile,
        })
        log(
            f"Senaryo checkpoint: bölüm={chapter['index']} "
            f"parça={part_index}/{part_count} kelime={word_count(generated)}"
        )
        return False

    narration = " ".join(parts)
    soft_minimum = int(
        profile.get(
            "soft_minimum_chapter_words",
            max(180, round(target_words * 0.88)),
        )
    )
    hard_minimum = int(
        profile.get(
            "hard_minimum_chapter_words",
            max(150, round(target_words * 0.60)),
        )
    )

    completion_models: list[str] = []
    below_soft = False
    if word_count(narration) < soft_minimum:
        narration, completion_models, below_soft = complete_chapter_narration(
            root,
            topic,
            bible,
            research,
            chapter,
            previous_summary,
            narration,
            target_words,
            soft_minimum,
            hard_minimum,
            profile,
            gemini,
        )

    summary, model = gemini.json(SYSTEM, f'''Aşağıdaki bölümün 90-150 kelimelik
sonraki bölüm tutarlılık özetini JSON ver: {{"summary":"..."}}\n{narration}''', 0.05)
    write_json(final, {
        "chapter_index": chapter["index"],
        "chapter_title": chapter["title"],
        "narration": narration,
        "word_count": word_count(narration),
        "target_words": target_words,
        "soft_minimum_words": soft_minimum,
        "hard_minimum_words": hard_minimum,
        "length_warning": below_soft,
        "completion_models": completion_models,
        "summary": summary["summary"],
        "summary_model": model,
        "generation_profile": profile,
    })
    log(
        f"Bölüm senaryosu hazır: bölüm={chapter['index']} "
        f"kelime={word_count(narration)} hedef={target_words} "
        f"minimum={hard_minimum}"
    )
    return True


def split_sentences(text: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", text) if item.strip()]


def scene_chunks(text: str, count: int) -> list[str]:
    sentences = split_sentences(text)
    total = max(1, sum(word_count(item) for item in sentences))
    target = total / count
    chunks, current, current_words = [], [], 0
    for sentence in sentences:
        words = word_count(sentence)
        if current and current_words + words > target * 1.22 and len(chunks) < count - 1:
            chunks.append(" ".join(current)); current = []; current_words = 0
        current.append(sentence); current_words += words
    if current:
        chunks.append(" ".join(current))
    while len(chunks) > count:
        chunks[-2] += " " + chunks[-1]; chunks.pop()
    while len(chunks) < count:
        longest = max(range(len(chunks)), key=lambda i: word_count(chunks[i]))
        parts = split_sentences(chunks[longest])
        if len(parts) < 2:
            break
        midpoint = len(parts) // 2
        chunks[longest:longest + 1] = [" ".join(parts[:midpoint]), " ".join(parts[midpoint:])]
    return chunks


def fallback_scene(
    topic: str,
    chapter: dict[str, Any],
    narration: str,
    scene_id: int,
    *,
    reason: str = "",
) -> dict[str, Any]:
    compact = " ".join(str(narration).split())
    contract = compact[:420]
    return {
        "scene_id": scene_id,
        "visual_contract_tr": contract,
        "prompt_en": (
            "Premium photorealistic historical documentary reconstruction. "
            f"Main topic: {topic}. Chapter: {chapter.get('title', '')}. "
            "Depict the exact concrete historical event described here: "
            f"{compact[:900]}. Show the central historical person or group, "
            "their action, and the period location together in one coherent "
            "widescreen frame. Historically plausible clothing, architecture "
            "and objects; no generic scenery."
        ),
        "must_show": [
            str(topic),
            str(chapter.get("title", "")),
            contract[:180],
        ],
        "must_not_show": [
            "visible text",
            "letters",
            "numbers",
            "subtitle",
            "logo",
            "watermark",
            "modern objects",
            "generic unrelated scenery",
        ],
        "importance": "key" if scene_id in {1, 2} else "normal",
        "planning_fallback": True,
        "fallback_reason": reason[:500],
    }


def normalize_scene(
    raw_scene: Any,
    topic: str,
    chapter: dict[str, Any],
    narration: str,
    scene_id: int,
    model: str,
) -> dict[str, Any]:
    if not isinstance(raw_scene, dict):
        scene = fallback_scene(
            topic,
            chapter,
            narration,
            scene_id,
            reason="Gemini sahnesi JSON nesnesi değildi.",
        )
    else:
        scene = dict(raw_scene)
        contract = " ".join(
            str(scene.get("visual_contract_tr", "")).split()
        )
        prompt_en = " ".join(
            str(scene.get("prompt_en", "")).split()
        )
        if len(contract) < 18 or len(prompt_en) < 35:
            fallback = fallback_scene(
                topic,
                chapter,
                narration,
                scene_id,
                reason="Gemini sahnesinde zorunlu alanlar eksikti.",
            )
            for key, value in fallback.items():
                if not scene.get(key):
                    scene[key] = value

    scene["scene_id"] = scene_id
    scene["narration"] = narration
    scene["model"] = model
    scene["must_show"] = [
        str(item)
        for item in scene.get("must_show", [])
        if str(item).strip()
    ][:8]
    scene["must_not_show"] = [
        str(item)
        for item in scene.get("must_not_show", [])
        if str(item).strip()
    ][:10]
    if not scene["must_not_show"]:
        scene["must_not_show"] = [
            "visible text",
            "subtitle",
            "logo",
            "modern objects",
        ]
    if scene.get("importance") not in {"normal", "key"}:
        scene["importance"] = "normal"
    return scene


def create_scene_plan(
    root: Path,
    topic: str,
    bible: dict[str, Any],
    chapter: dict[str, Any],
    script: dict[str, Any],
    count: int,
    gemini: Gemini,
) -> list[dict[str, Any]]:
    target = chapter_dir(root, int(chapter["index"])) / "scenes.json"
    existing = read_json(target)
    if isinstance(existing, list) and len(existing) == count:
        return existing

    chunks = scene_chunks(script["narration"], count)
    if len(chunks) != count:
        raise QualityGateError(
            f"Metin {count} sahneye ayrılamadı: {len(chunks)}"
        )

    batch_size = max(3, math.ceil(count / 2))
    scenes: list[dict[str, Any]] = []
    for batch_start in range(0, count, batch_size):
        batch_chunks = chunks[batch_start:batch_start + batch_size]
        batch_rows = [
            {
                "scene_id": batch_start + offset + 1,
                "narration": narration,
            }
            for offset, narration in enumerate(batch_chunks)
        ]
        expected_ids = [row["scene_id"] for row in batch_rows]
        prompt = f'''KONU: {topic}
GÖRSEL KİMLİK: {bible.get('visual_identity', '')}
BÖLÜM: {chapter}
BU PARTİDEKİ ANLATIM PARÇALARI: {batch_rows}

Yalnızca tek geçerli JSON nesnesi üret:
{{"scenes":[{{"scene_id":1,"visual_contract_tr":"kişi+eylem+mekân","prompt_en":"professional English image prompt","must_show":["somut öğe"],"must_not_show":["dönem dışı öğe"],"importance":"normal"}}]}}

Kurallar:
- Yalnız şu scene_id değerlerini üret: {expected_ids}
- Tam {len(batch_chunks)} sahne üret.
- Anlatıcı ne diyorsa aynı kişi, eylem ve mekân karede doğrudan görünsün.
- Genel manzara, rastgele kalabalık ve sembolik nesneyle kaçma.
- Görsel içinde yazı, sayı, tabela, altyazı, logo, yazıt ve sahte alfabe olmasın.
- prompt_en tek bir sinematik tarihsel anı açıkça tarif etsin.
- İkinci JSON nesnesi veya açıklama ekleme.'''

        model = "deterministic-scene-fallback"
        raw_scenes: list[Any] = []
        failure_reason = ""
        try:
            payload, model = gemini.json(
                "Sen profesyonel tarih belgeseli görsel yönetmenisin.",
                prompt,
                0.12,
            )
            value = payload.get("scenes", [])
            if isinstance(value, list):
                raw_scenes = value
            else:
                failure_reason = (
                    "Gemini scenes alanını liste döndürmedi."
                )
        except ControlledPause as exc:
            failure_reason = f"Gemini kota duraklaması: {exc}"
            log(
                "SCENE PLAN FAIL-SOFT: Gemini kotası nedeniyle "
                "anlatım-temelli sahne sözleşmesi kullanılacak."
            )
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            log(
                "SCENE PLAN FAIL-SOFT: bozuk JSON nedeniyle video "
                f"üretimi durdurulmadı: {failure_reason}"
            )

        by_id: dict[int, Any] = {}
        for raw_scene in raw_scenes:
            if not isinstance(raw_scene, dict):
                continue
            try:
                raw_id = int(raw_scene.get("scene_id"))
            except (TypeError, ValueError):
                continue
            if raw_id in expected_ids and raw_id not in by_id:
                by_id[raw_id] = raw_scene

        for offset, narration in enumerate(batch_chunks):
            scene_id = batch_start + offset + 1
            raw_scene = by_id.get(scene_id)
            if raw_scene is None and offset < len(raw_scenes):
                raw_scene = raw_scenes[offset]
            if raw_scene is None:
                raw_scene = fallback_scene(
                    topic,
                    chapter,
                    narration,
                    scene_id,
                    reason=(
                        failure_reason
                        or "Gemini eksik sahne döndürdü."
                    ),
                )
            scenes.append(
                normalize_scene(
                    raw_scene,
                    topic,
                    chapter,
                    narration,
                    scene_id,
                    model,
                )
            )

        write_json(
            target.with_name("scenes.partial.json"),
            scenes,
        )
        log(
            f"Sahne planı checkpoint: {len(scenes)}/{count} "
            f"(parti={batch_start // batch_size + 1})"
        )

    if len(scenes) != count:
        raise QualityGateError(
            f"Görsel plan iç hata: {len(scenes)}/{count}"
        )
    write_json(target, scenes)
    target.with_name("scenes.partial.json").unlink(missing_ok=True)
    fallback_count = sum(
        1
        for scene in scenes
        if scene.get("planning_fallback")
    )
    log(
        f"Sahne planı hazır: {len(scenes)} sahne, "
        f"fail_soft={fallback_count}"
    )
    return scenes


def technical_quality(path: Path) -> dict[str, float]:
    image = cv2.imread(str(path))
    if image is None:
        return {"sharpness": 0.0, "brightness": 0.0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
        "brightness": round(float(gray.mean()), 2),
    }


def ocr_report(path: Path, confidence: float) -> dict[str, Any]:
    image = ImageOps.autocontrast(Image.open(path).convert("L"))
    data = pytesseract.image_to_data(
        image, lang="eng+tur", config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )
    tokens, total = [], 0
    for raw_text, raw_conf in zip(data.get("text", []), data.get("conf", [])):
        clean = "".join(char for char in str(raw_text) if char.isalnum())
        try:
            score = float(raw_conf)
        except (TypeError, ValueError):
            score = -1
        if score >= confidence and len(clean) >= 3:
            tokens.append({"text": clean, "confidence": round(score, 1)})
            total += len(clean)
    return {"tokens": tokens[:15], "total_chars": total}


@lru_cache(maxsize=1)
def clip_components():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name)
    processor = CLIPProcessor.from_pretrained(name)
    model.eval()
    return torch, model, processor


def compact_clip_text(value: str, max_words: int = 52) -> str:
    """
    Keep the exact subject/action/place while removing verbose prompt language.

    CLIP ViT-B/32 supports at most 77 text positions. Word compaction improves
    relevance; tokenizer truncation below is still the definitive guard.
    """
    clean = " ".join(str(value).split())
    words = clean.split()
    if len(words) <= max_words:
        return clean

    # Preserve both the scene identity at the beginning and concrete mandatory
    # elements often placed near the end.
    head_count = max(1, round(max_words * 0.72))
    tail_count = max_words - head_count
    return " ".join(words[:head_count] + words[-tail_count:])


def clip_max_length(model: Any, processor: Any) -> int:
    values: list[int] = [77]

    text_config = getattr(getattr(model, "config", None), "text_config", None)
    model_limit = getattr(text_config, "max_position_embeddings", None)
    if isinstance(model_limit, int) and 2 <= model_limit <= 4096:
        values.append(model_limit)

    tokenizer_limit = getattr(
        getattr(processor, "tokenizer", None),
        "model_max_length",
        None,
    )
    if isinstance(tokenizer_limit, int) and 2 <= tokenizer_limit <= 4096:
        values.append(tokenizer_limit)

    return min(values)


def _clip_inputs(
    processor: Any,
    image: Image.Image,
    text: str,
    max_length: int,
) -> dict[str, Any]:
    """
    Tokenize text and image separately so truncation is guaranteed before the
    CLIP model receives input_ids.
    """
    text_inputs = processor.tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    image_inputs = processor.image_processor(
        images=[image],
        return_tensors="pt",
    )
    inputs = {
        **text_inputs,
        **image_inputs,
    }

    input_ids = inputs.get("input_ids")
    if input_ids is not None and int(input_ids.shape[-1]) > max_length:
        inputs["input_ids"] = input_ids[..., :max_length]
        attention = inputs.get("attention_mask")
        if attention is not None:
            inputs["attention_mask"] = attention[..., :max_length]

    return inputs


def clip_score(path: Path, text: str) -> float:
    """
    Safe CLIP image/text score.

    The previous implementation passed 107 tokens into a model with a 77-token
    position limit. This implementation always truncates at the model's actual
    limit and retries once with a shorter semantic contract if a processor
    behaves unexpectedly.
    """
    torch, model, processor = clip_components()
    maximum = clip_max_length(model, processor)
    image = Image.open(path).convert("RGB")

    attempts = (
        compact_clip_text(text, 52),
        compact_clip_text(text, 28),
    )
    last_error: Exception | None = None

    for attempt_index, semantic_text in enumerate(attempts, start=1):
        try:
            inputs = _clip_inputs(
                processor,
                image,
                semantic_text,
                maximum,
            )
            token_count = int(inputs["input_ids"].shape[-1])
            log(
                f"CLIP SAFE SCORE: deneme={attempt_index}/2 "
                f"token={token_count}/{maximum}"
            )
            with torch.no_grad():
                output = model(**inputs)
                image_norm = output.image_embeds.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                text_norm = output.text_embeds.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
                image_vector = output.image_embeds / image_norm
                text_vector = output.text_embeds / text_norm
                return round(
                    float((image_vector @ text_vector.T).item()),
                    4,
                )
        except Exception as exc:
            last_error = exc
            log(
                f"CLIP güvenli deneme başarısız: "
                f"{attempt_index}/2 — {type(exc).__name__}: {exc}"
            )

    raise QualityGateError(
        "CLIP kalite kontrolü iki güvenli denemede tamamlanamadı: "
        f"{last_error}"
    )


class CloudflareImages:
    def __init__(self, config: dict[str, Any]):
        account = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        token = os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        if not account or not token:
            raise ProviderUnavailable("Cloudflare account ID veya API token eksik.")
        self.config = config
        self.endpoint = f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{config['model']}"
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def generate(self, prompt: str, seed: int, target: Path) -> None:
        response = requests.post(
            self.endpoint,
            headers=self.headers,
            json={"prompt": prompt[:2048], "steps": int(self.config["steps"]), "seed": int(seed)},
            timeout=180,
        )
        if response.status_code >= 400:
            error = RuntimeError(f"Cloudflare HTTP {response.status_code}: {response.text[:1000]}")
            if quota_error(error):
                raise ControlledPause(str(error)) from error
            raise ProviderUnavailable(str(error)) from error
        payload = response.json()
        result = payload.get("result", payload)
        encoded = result.get("image") if isinstance(result, dict) else None
        if not encoded:
            raise ProviderUnavailable("Cloudflare görsel döndürmedi.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(encoded))


def deterministic_seed(pid: str, chapter: int, scene: int, attempt: int) -> int:
    raw = f"{pid}|{chapter}|{scene}|{attempt}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def image_prompt(topic: str, chapter_title: str, scene: dict[str, Any], config: dict[str, Any], repair: str = "") -> str:
    return " ".join((
        str(config["style_bible"]),
        f"Main topic: {topic}.",
        f"Chapter: {chapter_title}.",
        f"Exact narrated event: {scene['narration']}",
        f"Visual contract: {scene['visual_contract_tr']}.",
        f"Scene direction: {scene['prompt_en']}.",
        "Mandatory: " + ", ".join(str(x) for x in scene.get("must_show", [])) + ".",
        "Never show: " + ", ".join(str(x) for x in scene.get("must_not_show", [])) + ".",
        str(config["negative_prompt"]), repair,
        "Absolutely no visible writing anywhere in the pixels.",
    ))[:2048]


def make_storyboard(root: Path, chapter: int, scenes: list[dict[str, Any]], records: list[dict[str, Any]]) -> Path:
    cols, cell_w, image_h, caption_h = 3, 640, 360, 115
    rows = math.ceil(len(records) / cols)
    canvas = Image.new("RGB", (cols * cell_w, rows * (image_h + caption_h)), (14, 17, 23))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    for i, record in enumerate(records):
        x, y = (i % cols) * cell_w, (i // cols) * (image_h + caption_h)
        image = ImageOps.fit(Image.open(root / record["selected"]).convert("RGB"), (cell_w, image_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x, y))
        draw.rectangle((x, y + image_h, x + cell_w, y + image_h + caption_h), fill=(23, 26, 33))
        caption = f"S{i+1:02d} · CLIP {record['clip_score']:.3f} · {scenes[i]['visual_contract_tr']}"
        words, lines, current = caption.split(), [], ""
        for word in words:
            candidate = (current + " " + word).strip()
            if draw.textlength(candidate, font=font) < cell_w - 20:
                current = candidate
            else:
                lines.append(current); current = word
        if current:
            lines.append(current)
        for line_i, line in enumerate(lines[:3]):
            draw.text((x + 10, y + image_h + 8 + line_i * 28), line, font=font, fill=(236, 231, 219))
    target = root / "deliverables" / f"storyboard-chapter-{chapter:02d}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, "JPEG", quality=92, optimize=True)
    return target


def generate_visual_batch(root: Path, pid: str, topic: str, chapter: int, title: str, scenes: list[dict[str, Any]], config: dict[str, Any], budget: int) -> tuple[int, bool]:
    directory = chapter_dir(root, chapter)
    manifest_file = directory / "visual_manifest.json"
    records = {int(item["scene_id"]): item for item in read_json(manifest_file, [])}
    provider = CloudflareImages(config)
    used = 0
    for scene in scenes:
        scene_id = int(scene["scene_id"])
        selected = directory / "visuals" / f"scene-{scene_id:03d}.jpg"
        old = records.get(scene_id)
        if old and selected.exists() and old.get("accepted"):
            continue
        if used >= budget:
            return used, False
        best, repair = None, ""
        for attempt in range(1, int(config["attempts_per_scene"]) + 1):
            if used >= budget:
                break
            candidate = selected.with_name(f"scene-{scene_id:03d}-candidate-{attempt}.jpg")
            prompt = image_prompt(topic, title, scene, config, repair)
            provider.generate(prompt, deterministic_seed(pid, chapter, scene_id, attempt), candidate)
            used += 1
            with Image.open(candidate) as raw:
                frame = ImageOps.fit(raw.convert("RGB"), (int(config["width"]), int(config["height"])), Image.Resampling.LANCZOS)
                frame.save(candidate, "JPEG", quality=93, optimize=True)
            quality = technical_quality(candidate)
            ocr = ocr_report(candidate, float(config["ocr_confidence"]))
            semantic_text = " | ".join((
                str(scene.get("visual_contract_tr", "")),
                str(scene.get("prompt_en", "")),
                "mandatory: " + ", ".join(
                    str(item)
                    for item in scene.get("must_show", [])[:6]
                ),
            ))
            semantic = clip_score(
                candidate,
                semantic_text,
            )
            record = {
                "scene_id": scene_id,
                "candidate": str(candidate.relative_to(root)),
                "clip_score": semantic,
                "technical": quality,
                "ocr": ocr,
                "prompt": prompt,
                "accepted": False,
            }
            if best is None or semantic > best["clip_score"]:
                best = record
            good = (
                quality["sharpness"] >= float(config["minimum_sharpness"])
                and float(config["minimum_brightness"]) <= quality["brightness"] <= float(config["maximum_brightness"])
                and ocr["total_chars"] <= int(config["maximum_ocr_chars"])
                and semantic >= float(config["target_clip_score"])
            )
            if good:
                break
            repair = "Previous candidate was rejected. Rebuild exact person, action and place more clearly, with clean exposure and no writing."
        if not best:
            return used, False
        hard_ok = (
            best["technical"]["sharpness"] >= float(config["minimum_sharpness"])
            and float(config["minimum_brightness"]) <= best["technical"]["brightness"] <= float(config["maximum_brightness"])
            and best["ocr"]["total_chars"] <= int(config["maximum_ocr_chars"])
            and best["clip_score"] >= float(config["minimum_clip_score"])
        )
        if not hard_ok:
            raise QualityGateError(f"Sahne {scene_id} kalite eşiğini geçemedi: CLIP={best['clip_score']}")
        source = root / best["candidate"]
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(source.read_bytes())
        for candidate in selected.parent.glob(f"scene-{scene_id:03d}-candidate-*.jpg"):
            candidate.unlink(missing_ok=True)
        best.update({"selected": str(selected.relative_to(root)), "accepted": True})
        records[scene_id] = best
        write_json(manifest_file, sorted(records.values(), key=lambda x: x["scene_id"]))
        log(f"Görsel seçildi: bölüm={chapter} sahne={scene_id} CLIP={best['clip_score']}")
    complete = len([x for x in records.values() if x.get("accepted")]) == len(scenes)
    if complete:
        make_storyboard(root, chapter, scenes, sorted(records.values(), key=lambda x: x["scene_id"]))
    return used, complete


def audio_chunks(text: str, target_words: int) -> list[str]:
    sentences = split_sentences(text)
    output, current, current_words = [], [], 0
    for sentence in sentences:
        words = word_count(sentence)
        if current and current_words + words > target_words:
            output.append(" ".join(current)); current = []; current_words = 0
        current.append(sentence); current_words += words
    if current:
        output.append(" ".join(current))
    return output


def write_pcm(path: Path, pcm: bytes, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(sample_rate); wav.writeframes(pcm)


def normalize_voice(raw: Path, target: Path, words: int, config: dict[str, Any]) -> dict[str, float]:
    seconds = ffprobe_duration(raw)
    actual = words / max(seconds, 1) * 60
    if actual < float(config["minimum_wpm"]) * 0.82 or actual > float(config["maximum_wpm"]) * 1.18:
        raise QualityGateError(f"Charon tempo kabul edilemez: {actual:.1f} WPM")
    tempo = max(0.90, min(1.10, float(config["target_wpm"]) / max(actual, 1)))
    run([
        "ffmpeg", "-y", "-i", str(raw), "-af",
        f"atempo={tempo:.4f},highpass=f=65,lowpass=f=11000,loudnorm=I=-17:TP=-2:LRA=7",
        "-ar", str(config["output_sample_rate"]), "-ac", "1", "-c:a", "flac", str(target),
    ], 300)
    return {"raw_seconds": round(seconds, 2), "final_seconds": round(ffprobe_duration(target), 2), "raw_wpm": round(actual, 2), "tempo": round(tempo, 4)}


def generate_tts_batch(root: Path, chapter: int, narration: str, config: dict[str, Any], gemini: Gemini, budget: int) -> tuple[int, bool]:
    directory = chapter_dir(root, chapter) / "tts"
    directory.mkdir(parents=True, exist_ok=True)
    manifest_file = directory / "manifest.json"
    manifest = read_json(manifest_file, {"voice": "Charon", "chunks": []})
    records = {int(item["index"]): item for item in manifest.get("chunks", [])}
    parts = audio_chunks(narration, int(config["chunk_words"]))
    used = 0
    for i, text in enumerate(parts, 1):
        target = directory / f"chunk-{i:02d}.flac"
        if i in records and target.exists() and records[i].get("voice") == "Charon":
            continue
        if used >= budget:
            return used, False
        raw = directory / f"chunk-{i:02d}-raw.wav"
        log(f"Charon: bölüm={chapter} parça={i}/{len(parts)}")
        write_pcm(raw, gemini.tts_pcm(text, str(config["instruction"])))
        info = normalize_voice(raw, target, word_count(text), config)
        raw.unlink(missing_ok=True)
        records[i] = {"index": i, "file": str(target.relative_to(root)), "voice": "Charon", "words": word_count(text), **info}
        write_json(manifest_file, {"voice": "Charon", "chunks": sorted(records.values(), key=lambda x: x["index"])})
        used += 1
    complete = len(records) == len(parts)
    if complete:
        concat = directory / "concat.txt"
        concat.write_text("\n".join(f"file '{(directory / f'chunk-{i:02d}.flac').as_posix()}'" for i in range(1, len(parts)+1)), encoding="utf-8")
        final = chapter_dir(root, chapter) / "narration.flac"
        run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c:a", "flac", str(final)], 900)
    return used, complete


def scene_durations(scenes: list[dict[str, Any]], audio_seconds: float) -> list[float]:
    weights = [max(1, word_count(scene["narration"])) for scene in scenes]
    total = sum(weights)
    values = [audio_seconds * weight / total for weight in weights]
    values[-1] += audio_seconds - sum(values)
    return values


def render_chapter(root: Path, chapter: int, scenes: list[dict[str, Any]], config: dict[str, Any]) -> Path:
    directory = chapter_dir(root, chapter)
    target = directory / "chapter.mp4"
    if target.exists() and ffprobe_duration(target) > 120:
        return target
    audio = directory / "narration.flac"
    manifest = read_json(directory / "visual_manifest.json", [])
    selected = {int(item["scene_id"]): root / item["selected"] for item in manifest if item.get("accepted")}
    if len(selected) != len(scenes):
        raise RuntimeError("Görseller tamamlanmadan render başlatılamaz.")
    values = scene_durations(scenes, ffprobe_duration(audio))
    render_dir = directory / "render"
    render_dir.mkdir(exist_ok=True)
    clips = []
    fps, zoom = int(config["fps"]), float(config["subtle_zoom"])
    for scene, seconds in zip(scenes, values):
        sid = int(scene["scene_id"])
        clip = render_dir / f"scene-{sid:03d}.mp4"
        frames = max(1, round(seconds * fps))
        zoom_step = max(0.000001, (zoom - 1.0) / frames)
        run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(selected[sid]),
            "-vf",
            f"scale={config['width']}:{config['height']}:force_original_aspect_ratio=increase,"
            f"crop={config['width']}:{config['height']},"
            f"zoompan=z='min(zoom+{zoom_step:.8f},{zoom})':"
            "x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={config['width']}x{config['height']}:fps={fps},format=yuv420p",
            "-t", f"{seconds:.3f}", "-an", "-c:v", "libx264",
            "-preset", str(config["preset"]), "-crf", str(config["crf"]),
            "-r", str(fps), str(clip),
        ], 1500)
        clips.append(clip)
    concat = render_dir / "concat.txt"
    concat.write_text("\n".join(f"file '{clip.as_posix()}'" for clip in clips), encoding="utf-8")
    silent = render_dir / "silent.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-c:v", "copy", str(silent)], 3600)
    run([
        "ffmpeg", "-y", "-i", str(silent), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
        "-c:a", "aac", "-profile:a", "aac_low", "-b:a", str(config["audio_bitrate"]),
        "-ar", "48000", "-ac", "2", "-shortest", "-movflags", "+faststart", str(target),
    ], 3600)
    for item in clips + [silent, concat]:
        item.unlink(missing_ok=True)
    return target


def assemble_final(root: Path, count: int, config: dict[str, Any]) -> Path:
    deliverables = root / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    target = deliverables / "final-video.mp4"
    files = [chapter_dir(root, i) / "chapter.mp4" for i in range(1, count + 1)]
    if not all(path.exists() for path in files):
        raise RuntimeError("Bölüm videoları eksik.")
    concat = deliverables / "chapters.txt"
    concat.write_text("\n".join(f"file '{path.as_posix()}'" for path in files), encoding="utf-8")
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
        "-c:a", "aac", "-profile:a", "aac_low", "-b:a", str(config["audio_bitrate"]),
        "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(target),
    ], 7200)
    return target


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def clean_project(root: Path, maximum_gb: float) -> dict[str, float]:
    for pattern in ("*-raw.wav", "*-candidate-*.jpg"):
        for item in root.rglob(pattern):
            item.unlink(missing_ok=True)
    for render_dir in root.rglob("render"):
        if render_dir.is_dir():
            for item in render_dir.iterdir():
                if item.is_file():
                    item.unlink(missing_ok=True)
    size = directory_size(root) / (1024 ** 3)
    if size > maximum_gb:
        for directory in sorted((root / "chapters").glob("[0-9][0-9]")):
            if (directory / "chapter.mp4").exists() and (directory / "narration.flac").exists():
                shutil.rmtree(directory / "tts", ignore_errors=True)
        size = directory_size(root) / (1024 ** 3)
    if size > maximum_gb:
        raise RuntimeError(f"Cache sınırı aşıldı: {size:.2f} GB > {maximum_gb:.2f} GB")
    log(f"Cache proje boyutu: {size:.2f} GB")
    return {"size_gb": round(size, 3), "maximum_gb": maximum_gb}


def state_template(pid: str, topic: str, minutes: int) -> dict[str, Any]:
    return {
        "version": VERSION,
        "project_id": pid,
        "topic": topic,
        "minutes": minutes,
        "status": "created",
        "chapters": {},
    }


def run_project(root: Path, pid: str, topic: str, minutes: int, config: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    max_minutes = int(config["project"]["max_wall_minutes"])
    state_file = root / "state.json"
    state = read_json(state_file, state_template(pid, topic, minutes))
    state.update({"version": VERSION, "project_id": pid, "topic": topic, "minutes": minutes, "status": "running"})
    write_json(state_file, state)
    gemini = Gemini(config["gemini"])
    profile = duration_profile(minutes, config)
    chapters = int(profile["chapters"])
    scenes_per_chapter = int(profile["scenes_per_chapter"])
    script_parts = int(profile["script_parts"])
    state["generation_profile"] = profile
    write_json(state_file, state)
    log(
        "ADAPTIVE DURATION PROFILE: "
        f"minutes={minutes}, chapters={chapters}, "
        f"scenes/chapter={scenes_per_chapter}, script_parts={script_parts}, "
        f"target_words={profile['target_total_words']}, "
        f"chapter_soft_min={profile['soft_minimum_chapter_words']}, "
        f"chapter_hard_min={profile['hard_minimum_chapter_words']}"
    )
    try:
        research_file = root / "research.json"
        research = read_json(research_file)
        if not research:
            research = build_research(topic, config["research"], gemini)
            write_json(research_file, research); state["research"] = "ready"; write_json(state_file, state)
        bible_file = root / "story_bible.json"
        bible = read_json(bible_file)
        if bible and not profile_matches(bible.get("generation_profile"), profile):
            reset_stale_story_structure(
                root,
                state,
                "Önceki sabit 6-bölüm yapısı istenen video süresiyle uyumsuzdu.",
            )
            write_json(state_file, state)
            bible = None
        if not bible:
            bible = create_story_bible(topic, minutes, chapters, research, gemini)
            bible["generation_profile"] = profile
            write_json(bible_file, bible)
            state["story_bible"] = "ready"
            write_json(state_file, state)
        target_total = int(profile["target_total_words"])
        target_chapter = int(profile["target_chapter_words"])
        image_budget_remaining = int(config["images"]["maximum_images_per_run"])
        tts_budget_remaining = int(config["tts"]["maximum_chunks_per_run"])
        while time.monotonic() - started < max_minutes * 60:
            progress = False
            previous_summary = ""
            for chapter in bible["chapters"]:
                index = int(chapter["index"])
                cstate = state.setdefault("chapters", {}).setdefault(str(index), {})
                script_file = chapter_dir(root, index) / "script.json"
                script = read_json(script_file)
                if not script:
                    finished = build_script_checkpoint(root, topic, bible, research, chapter, previous_summary, target_chapter, script_parts, profile, gemini)
                    progress = True
                    if finished:
                        script = read_json(script_file); cstate["script"] = "ready"; write_json(state_file, state)
                    break
                previous_summary = script.get("summary", previous_summary)
                cstate["script"] = "ready"
                scenes = read_json(chapter_dir(root, index) / "scenes.json")
                if not scenes:
                    create_scene_plan(root, topic, bible, chapter, script, scenes_per_chapter, gemini)
                    cstate["scenes"] = "ready"; write_json(state_file, state); progress = True; break
                cstate["scenes"] = "ready"
            if progress:
                continue
            image_budget = image_budget_remaining
            for chapter in bible["chapters"]:
                if image_budget <= 0:
                    break
                index = int(chapter["index"]); cstate = state["chapters"][str(index)]
                if cstate.get("visuals") == "ready":
                    continue
                scenes = read_json(chapter_dir(root, index) / "scenes.json")
                used, complete = generate_visual_batch(root, pid, topic, index, str(chapter["title"]), scenes, config["images"], image_budget)
                image_budget -= used
                image_budget_remaining -= used
                progress = progress or used > 0
                if complete:
                    cstate["visuals"] = "ready"; write_json(state_file, state)
                if image_budget <= 0 or not complete:
                    break
            if progress:
                continue
            tts_budget = tts_budget_remaining
            for chapter in bible["chapters"]:
                if tts_budget <= 0:
                    break
                index = int(chapter["index"]); cstate = state["chapters"][str(index)]
                if cstate.get("visuals") != "ready":
                    break
                if cstate.get("tts") == "ready":
                    continue
                script = read_json(chapter_dir(root, index) / "script.json")
                used, complete = generate_tts_batch(root, index, script["narration"], config["tts"], gemini, tts_budget)
                tts_budget -= used
                tts_budget_remaining -= used
                progress = progress or used > 0
                if complete:
                    cstate["tts"] = "ready"; write_json(state_file, state)
                if tts_budget <= 0 or not complete:
                    break
            if progress:
                continue
            rendered = False
            for chapter in bible["chapters"]:
                index = int(chapter["index"]); cstate = state["chapters"][str(index)]
                if cstate.get("tts") != "ready":
                    break
                if cstate.get("render") == "ready":
                    continue
                scenes = read_json(chapter_dir(root, index) / "scenes.json")
                video = render_chapter(root, index, scenes, config["render"])
                cstate["render"] = "ready"; cstate["video_seconds"] = round(ffprobe_duration(video), 2)
                write_json(state_file, state); rendered = progress = True; break
            if rendered:
                continue
            all_rendered = all(state.get("chapters", {}).get(str(i), {}).get("render") == "ready" for i in range(1, chapters + 1))
            if all_rendered:
                final = root / "deliverables" / "final-video.mp4"
                if not final.exists():
                    assemble_final(root, chapters, config["render"])
                    state["final"] = "ready"; state["final_seconds"] = round(ffprobe_duration(final), 2)
                    write_json(state_file, state); progress = True; continue
                state["status"] = "complete"; write_json(state_file, state); break
            if not progress:
                state["status"] = "waiting_for_next_run"; write_json(state_file, state); break
        if state.get("status") == "running":
            state["status"] = "waiting_for_next_run"
            write_json(state_file, state)
    except ControlledPause as exc:
        state["status"] = "paused_quota"
        state["pause_reason"] = str(exc)
        write_json(state_file, state)
        log(f"KONTROLLÜ DURAKLAMA: {exc}")
    except Exception as exc:
        state["status"] = "failed"
        state["error_type"] = type(exc).__name__
        state["error_message"] = str(exc)
        state["failed_at"] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        )
        write_json(state_file, state)
        report = root / "deliverables" / "ERROR_REPORT.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            "UYKU VE TARİH V11.1.1 HATA RAPORU\n\n"
            f"Tür: {type(exc).__name__}\n"
            f"Mesaj: {exc}\n"
            f"Proje: {pid}\n"
            f"Konu: {topic}\n",
            encoding="utf-8",
        )
        log(
            "FATAL ERROR SAVED: "
            f"{type(exc).__name__}: {exc}"
        )
        try:
            state["storage"] = clean_project(
                root,
                float(config["project"]["maximum_cache_gb"]),
            )
            write_json(state_file, state)
        finally:
            raise
    state["storage"] = clean_project(
        root,
        float(config["project"]["maximum_cache_gb"]),
    )
    write_json(state_file, state)
    return state


def control_set(path: Path, topic: str, minutes: int, pid: str = "") -> str:
    pid = pid or project_id(topic, minutes)
    write_json(path, {
        "active": True, "project_id": pid, "topic": topic, "minutes": minutes,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return pid


def main() -> None:
    parser = argparse.ArgumentParser("Uyku ve Tarih V11.1 Free Cloud Agent")
    parser.add_argument("--config", default="config/defaults.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("project-id"); p.add_argument("--topic", required=True); p.add_argument("--minutes", type=int, default=60)
    s = sub.add_parser("step"); s.add_argument("--topic", required=True); s.add_argument("--minutes", type=int, default=60); s.add_argument("--project-id", default="")
    c = sub.add_parser("control-set"); c.add_argument("--topic", required=True); c.add_argument("--minutes", type=int, default=60); c.add_argument("--project-id", default=""); c.add_argument("--file", default="control/active_project.json")
    d = sub.add_parser("control-complete"); d.add_argument("--file", default="control/active_project.json")
    args = parser.parse_args()
    if args.command == "project-id":
        print(project_id(args.topic, args.minutes)); return
    if args.command == "control-set":
        print(control_set(Path(args.file), args.topic, args.minutes, args.project_id)); return
    if args.command == "control-complete":
        data = read_json(Path(args.file), {}); data["active"] = False; write_json(Path(args.file), data); print(data); return
    config = load_config(Path(args.config))
    pid = args.project_id or project_id(args.topic, args.minutes)
    root = Path(config["project"]["data_root"]).resolve() / pid
    root.mkdir(parents=True, exist_ok=True)
    state = run_project(root, pid, args.topic, args.minutes, config)
    print(f"PROJECT_STATUS={state.get('status', 'unknown')}")


if __name__ == "__main__":
    main()
