#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF preprocessing utility for section-aware biomedical paper extraction.

Design goals:
1. Use bookmarks when they are available and trustworthy.
2. Fall back to full-text chunking when bookmarks are missing or sparse.
3. Emit a single flat JSON schema that downstream loaders can consume.
"""

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PyPDF2
import pdfplumber


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_KEYWORDS = ("reference", "references", "bibliography")
SECTION_HEADING_ALIASES = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "related works": "Related Work",
    "materials and methods": "Materials and Methods",
    "methods and materials": "Methods and Materials",
    "methods": "Methods",
    "methodology": "Methodology",
    "patients and methods": "Patients and Methods",
    "case presentation": "Case Presentation",
    "results": "Results",
    "results and discussion": "Results and Discussion",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusions",
    "limitations": "Limitations",
}
TAIL_SECTION_ALIASES = {
    "references": "References",
    "reference": "References",
    "bibliography": "References",
    "acknowledgments": "Acknowledgments",
    "acknowledgements": "Acknowledgments",
    "funding": "Funding",
    "author contributions": "Author Contributions",
    "authors contributions": "Author Contributions",
    "authors contribution": "Author Contributions",
    "credit authorship contribution statement": "CRediT Authorship Contribution Statement",
    "declaration of competing interest": "Declaration of Competing Interest",
    "declaration of competing interests": "Declaration of Competing Interest",
    "conflict of interest": "Conflict of Interest",
    "conflicts of interest": "Conflict of Interest",
    "ethics statement": "Ethics Statement",
    "ethics approval": "Ethics Approval",
    "data availability": "Data Availability",
    "data availability statement": "Data Availability Statement",
    "disclosures": "Disclosures",
    "supplementary data": "Supplementary Data",
    "supplementary information": "Supplementary Information",
    "supplementary materials": "Supplementary Materials",
    "supporting information": "Supporting Information",
    "appendix": "Appendix",
    "orcid": "ORCID",
    "orcid ids": "ORCID iDs",
    "additional information": "Additional Information",
    "affiliations": "Affiliations",
}
NOISE_LINE_PATTERNS = (
    r"^biorxiv preprint doi\b",
    r"^\(which was not certified by peer review\)",
    r"^available under a[a-z\- ]+license\.?$",
    r"^open access\b",
    r"^running title\b",
    r"^contents lists available",
    r"^journal homepage\b",
    r"^available online\b",
    r"^received\b",
    r"^accepted\b",
    r"^published\b",
    r"^review began\b",
    r"^review ended\b",
    r"^how to cite this article\b",
    r"^corresponding author\b",
    r"^\*?correspondence\b",
    r"^affiliations?\b",
    r"^authors?' contributions?\b",
    r"^contributed equally to this work\b",
    r"^doi\s*:",
    r"^https?://doi\.org/",
    r"^page \d+",
    r"^\d+\s+of\s+\d+$",
    r"^(article|research article|case report|graphical abstract)$",
    r"^(a r t i c l e i n f o|article info|highlights)$",
)
MOJIBAKE_REPLACEMENTS = {
    "鈥": "-",
    "铿": "",
    "卤": "",
    "æ¼": "©",
    "éˆ¥": "-",
    "éŠ†": ".",
    "åº": " ",
    "(cid:0)": " ",
}


@dataclass
class BookmarkInfo:
    title: str
    page_num: int
    level: int


@dataclass
class Section:
    title: str
    content: str
    page_start: int
    page_end: int
    section_id: str
    level: int = 1
    bookmark_title: str = ""


@dataclass
class ProcessingStats:
    total_files: int = 0
    processed_successfully: int = 0
    skipped_no_bookmarks: int = 0
    skipped_too_many_bookmarks: int = 0
    failed: int = 0
    skipped_reasons: List[str] = field(default_factory=list)


class BookmarkBasedSplitter:
    """Split PDF papers into section-aware JSON chunks."""

    def __init__(self, max_chunk_size: int = 5000, max_bookmarks: int = 20, min_bookmarks: int = 3):
        self.max_chunk_size = max_chunk_size
        self.max_bookmarks = max_bookmarks
        self.min_bookmarks = min_bookmarks

    def extract_bookmarks(self, pdf_path: str) -> List[BookmarkInfo]:
        """Extract top-level bookmarks from a PDF."""
        bookmarks: List[BookmarkInfo] = []

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                outline = getattr(pdf_reader, "outline", None)
                if not outline:
                    logger.warning("PDF has no bookmark structure: %s", pdf_path)
                    return []

                def process_items(items: List[Any], level: int = 1) -> None:
                    for item in items:
                        if isinstance(item, list):
                            if level == 1:
                                process_items(item, level + 1)
                            continue

                        if level > 1:
                            continue

                        title = getattr(item, "title", str(item)).strip()
                        if not title:
                            continue

                        try:
                            page_num = pdf_reader.get_destination_page_number(item)
                        except Exception:
                            page_num = None

                        if page_num is None:
                            logger.warning("Cannot determine page number for bookmark '%s'", title)
                            continue

                        bookmarks.append(BookmarkInfo(title=title, page_num=page_num, level=level))

                process_items(outline)

        except Exception as exc:
            logger.error("Failed to extract bookmarks from %s: %s", pdf_path, exc)

        return bookmarks

    def _has_valid_title_case(self, title: str) -> bool:
        """Allow normal English headings and also keep CJK headings."""
        title = title.strip()
        if not title:
            return False

        first_char = title[0]
        if "\u4e00" <= first_char <= "\u9fff":
            return True

        match = re.match(r"^[\d\.\s]*([a-zA-Z])", title)
        if match:
            return match.group(1).isupper()

        return True

    def filter_bookmarks_by_title(self, bookmarks: List[BookmarkInfo]) -> List[BookmarkInfo]:
        filtered: List[BookmarkInfo] = []
        for bookmark in bookmarks:
            title = bookmark.title.strip()
            if re.match(r"^abstract\b", title, re.IGNORECASE):
                filtered.append(bookmark)
                continue
            if self._has_valid_title_case(title):
                filtered.append(bookmark)
        return filtered

    def remove_references_and_after(self, bookmarks: List[BookmarkInfo]) -> List[BookmarkInfo]:
        for index, bookmark in enumerate(bookmarks):
            title_lower = bookmark.title.lower().strip()
            if any(keyword in title_lower for keyword in REFERENCE_KEYWORDS):
                return bookmarks[:index]
        return bookmarks

    def detect_column_layout(self, page: pdfplumber.page.Page) -> bool:
        chars = page.chars
        if not chars:
            return False

        x_centers = [
            (char["x0"] + char["x1"]) / 2
            for char in chars
            if char.get("text", "").strip()
        ]
        if not x_centers:
            return False

        mid_x = page.width / 2
        gutter_half_width = page.width * 0.06
        left_chars = sum(1 for x in x_centers if x < mid_x - gutter_half_width)
        right_chars = sum(1 for x in x_centers if x > mid_x + gutter_half_width)
        center_chars = sum(1 for x in x_centers if abs(x - mid_x) <= gutter_half_width)
        total_chars = len(x_centers)
        if total_chars == 0:
            return False
        center_ratio_limit = 0.12
        if getattr(page, "page_number", 1) > 1:
            center_ratio_limit = 0.16

        if center_chars / total_chars > center_ratio_limit:
            return False
        return total_chars > 0 and left_chars / total_chars > 0.3 and right_chars / total_chars > 0.3

    def extract_page_text_smart(self, page: pdfplumber.page.Page) -> str:
        return self._extract_two_column_text(page) if self.detect_column_layout(page) else (page.extract_text() or "")

    def _extract_two_column_text(self, page: pdfplumber.page.Page) -> str:
        chars = page.chars
        if not chars:
            return ""

        mid_x = page.width / 2
        left_chars = [char for char in chars if char["x0"] < mid_x]
        right_chars = [char for char in chars if char["x0"] >= mid_x]

        left_text = self._chars_to_text(left_chars)
        right_text = self._chars_to_text(right_chars)

        if left_text and right_text:
            return left_text + "\n\n" + right_text
        return left_text or right_text

    def _chars_to_text(self, chars: List[Dict[str, Any]]) -> str:
        if not chars:
            return ""

        sorted_chars = sorted(chars, key=lambda item: (-item["y0"], item["x0"]))
        lines: List[str] = []
        current_line: List[Dict[str, Any]] = []
        current_y: Optional[float] = None

        for char in sorted_chars:
            y_value = round(char["y0"], 1)
            if current_y is None or abs(y_value - current_y) <= 3:
                current_line.append(char)
                current_y = y_value if current_y is None else current_y
                continue

            lines.append(self._build_spaced_line_text(current_line))
            current_line = [char]
            current_y = y_value

        if current_line:
            lines.append(self._build_spaced_line_text(current_line))

        return "\n".join(lines)

    def _build_spaced_line_text(self, chars: List[Dict[str, Any]]) -> str:
        chars = sorted(chars, key=lambda item: item["x0"])
        result: List[str] = []
        previous_char: Optional[Dict[str, Any]] = None

        for char in chars:
            if previous_char is not None:
                gap = char["x0"] - previous_char["x1"]
                previous_width = max(previous_char["x1"] - previous_char["x0"], 6)
                current_width = max(char["x1"] - char["x0"], 6)
                average_width = (previous_width + current_width) / 2
                if gap > average_width * 0.3:
                    result.append("  " if gap > average_width * 1.5 else " ")

            result.append(char["text"])
            previous_char = char

        return "".join(result)

    def split_by_bookmarks(self, pdf_path: str) -> Tuple[Optional[List[Section]], str]:
        """Split PDF by bookmarks when enough good bookmarks exist."""
        bookmarks = self.extract_bookmarks(pdf_path)
        if not bookmarks:
            return None, "No bookmark structure found"

        bookmarks = self.filter_bookmarks_by_title(bookmarks)
        if len(bookmarks) < self.min_bookmarks:
            return None, f"Too few bookmarks: {len(bookmarks)} < {self.min_bookmarks}"
        if len(bookmarks) > self.max_bookmarks:
            return None, f"Too many bookmarks: {len(bookmarks)} > {self.max_bookmarks}"

        bookmarks = self.remove_references_and_after(bookmarks)
        bookmarks.sort(key=lambda item: item.page_num)

        sections: List[Section] = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                for index, bookmark in enumerate(bookmarks):
                    start_page = bookmark.page_num
                    end_page = total_pages - 1
                    next_bookmark = None

                    for later in bookmarks[index + 1:]:
                        if later.level <= bookmark.level:
                            next_bookmark = later
                            end_page = bookmark.page_num if later.page_num == bookmark.page_num else later.page_num - 1
                            break

                    content = self._extract_section_content(pdf, start_page, end_page, bookmark, next_bookmark)
                    if not content.strip():
                        continue

                    sections.append(
                        Section(
                            title=self._clean_title(bookmark.title),
                            content=content,
                            page_start=start_page + 1,
                            page_end=end_page + 1,
                            section_id=f"section_{index + 1}",
                            level=bookmark.level,
                            bookmark_title=bookmark.title,
                        )
                    )

        except Exception as exc:
            logger.error("Failed to split PDF %s by bookmarks: %s", pdf_path, exc)
            return None, f"Processing error: {exc}"

        return (sections, "success") if sections else (None, "No valid sections")

    def split_by_char_count(self, pdf_path: str) -> Tuple[Optional[List[Section]], str]:
        """Fallback path when bookmark-based splitting is unavailable."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_texts: List[Tuple[int, str]] = []
                for page_index, page in enumerate(pdf.pages, start=1):
                    page_texts.append((page_index, self.extract_page_text_smart(page)))
        except Exception as exc:
            logger.error("Failed to extract PDF text from %s: %s", pdf_path, exc)
            return None, str(exc)

        repeated_boundary_lines = self._identify_repeated_boundary_lines([text for _, text in page_texts])
        cleaned_pages = [
            (page_num, self.clean_text_block(text, repeated_boundary_lines))
            for page_num, text in page_texts
        ]
        sections = self._split_by_detected_headings(cleaned_pages)
        if sections:
            return sections, "detected section headings"

        full_text = "\n\n".join(text for _, text in cleaned_pages if text.strip())
        full_text, _ = self._truncate_tail_matter(full_text)
        if not full_text.strip():
            return None, "No text extracted from PDF"

        sections: List[Section] = []
        chunks = self._build_chunks(full_text, "chunk")
        for index, chunk in enumerate(chunks, start=1):
            sections.append(
                Section(
                    title=f"Chunk {index}",
                    content=chunk["content"],
                    page_start=1,
                    page_end=page_texts[-1][0] if page_texts else 1,
                    section_id=f"chunk_{index}",
                )
            )

        return sections, "success"

    def _normalize_text_artifacts(self, text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        for bad, good in MOJIBAKE_REPLACEMENTS.items():
            text = text.replace(bad, good)
        text = re.sub(r"(\w)-\n(?=\w)", r"\1", text)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def _normalize_line_for_matching(self, line: str) -> str:
        normalized = self._normalize_text_artifacts(line)
        normalized = re.sub(r"\s+", " ", normalized).strip().lower()
        normalized = re.sub(r"\d+", "#", normalized)
        normalized = re.sub(r"[^a-z0-9# ]", "", normalized)
        return normalized

    def _normalize_heading_candidate(self, line: str) -> str:
        normalized = self._normalize_text_artifacts(line)
        normalized = re.sub(r"\s+", " ", normalized).strip(" :-\t")
        normalized = re.sub(r"^(?:section\s+)?(?:\d+(?:\.\d+){0,3}|[ivxlcdm]+)[\.\)\-: ]+\s*", "", normalized, flags=re.IGNORECASE)
        return normalized.lower().strip()

    def _identify_repeated_boundary_lines(self, text_blocks: List[str]) -> set[str]:
        counts: Dict[str, int] = {}
        for text in text_blocks:
            lines = [line.strip() for line in self._normalize_text_artifacts(text).splitlines() if line.strip()]
            boundary_lines = lines[:3] + lines[-3:]
            for line in boundary_lines:
                normalized = self._normalize_line_for_matching(line)
                if len(normalized) < 6 or len(normalized) > 100:
                    continue
                counts[normalized] = counts.get(normalized, 0) + 1

        return {line for line, count in counts.items() if count >= 2}

    def _is_noise_line(self, line: str, repeated_boundary_lines: Optional[set[str]] = None) -> bool:
        stripped = line.strip()
        if not stripped:
            return True

        normalized = self._normalize_line_for_matching(stripped)
        if repeated_boundary_lines and normalized in repeated_boundary_lines:
            return True

        if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in NOISE_LINE_PATTERNS):
            return True

        if re.match(r"^https?://", stripped, flags=re.IGNORECASE):
            return True
        if "creativecommons" in stripped.lower():
            return True
        if "all rights reserved" in stripped.lower():
            return True
        if "@" in stripped and len(stripped) < 120:
            return True
        if re.match(r"^\d{4} .* doi ", stripped, flags=re.IGNORECASE):
            return True
        if re.match(r"^[A-Z][a-z]+ et al\.$", stripped):
            return True

        return False

    def _parse_heading_line(self, line: str) -> Tuple[Optional[str], bool, str]:
        stripped = line.strip()
        normalized = self._normalize_heading_candidate(stripped)
        if not normalized:
            return None, False, ""

        for alias, canonical in SECTION_HEADING_ALIASES.items():
            if normalized == alias:
                return canonical, False, ""
            if normalized.startswith(f"{alias}:"):
                return canonical, False, stripped.split(":", 1)[1].strip()
            numbered_match = re.match(
                rf"^(?:section\s+)?(?:\d+(?:\.\d+){{0,3}}|[ivxlcdm]+)[\.\)\-: ]+{re.escape(alias)}\b(.*)$",
                stripped,
                flags=re.IGNORECASE,
            )
            if numbered_match:
                return canonical, False, numbered_match.group(1).strip()

        for alias, canonical in TAIL_SECTION_ALIASES.items():
            if normalized == alias:
                return canonical, True, ""
            if normalized.startswith(f"{alias}:"):
                return canonical, True, stripped.split(":", 1)[1].strip()
            numbered_match = re.match(
                rf"^(?:section\s+)?(?:\d+(?:\.\d+){{0,3}}|[ivxlcdm]+)[\.\)\-: ]+{re.escape(alias)}\b(.*)$",
                stripped,
                flags=re.IGNORECASE,
            )
            if numbered_match:
                return canonical, True, numbered_match.group(1).strip()

        return None, False, ""

    def _merge_wrapped_lines(self, lines: List[str]) -> List[str]:
        merged: List[str] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            if not merged:
                merged.append(line)
                continue

            previous = merged[-1]
            if previous.endswith("-") and line and line[0].islower():
                merged[-1] = previous[:-1] + line
                continue

            if (
                previous
                and previous[-1] not in ".!?;:"
                and line
                and (line[0].islower() or line[0].isdigit() or line[0] in "([" or line[:2].islower())
            ):
                merged[-1] = f"{previous} {line}"
                continue

            merged.append(line)

        return merged

    def _drop_leading_front_matter(self, lines: List[str]) -> List[str]:
        probe_limit = min(len(lines), 80)
        for index in range(probe_limit):
            heading, is_tail, remainder = self._parse_heading_line(lines[index])
            if heading and not is_tail and heading in {
                "Abstract",
                "Introduction",
                "Background",
                "Methods",
                "Methodology",
                "Materials and Methods",
                "Results",
                "Results and Discussion",
                "Case Presentation",
            }:
                new_lines = [heading]
                if remainder:
                    new_lines.append(remainder)
                new_lines.extend(lines[index + 1 :])
                return new_lines
        return lines

    def _looks_like_body_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False

        heading, is_tail, _ = self._parse_heading_line(stripped)
        if heading and not is_tail:
            return True

        if re.match(r"^\d+(?:\.\d+){0,3}\s+[A-Z]", stripped):
            return True

        words = stripped.split()
        return len(words) >= 8 and bool(re.search(r"[A-Za-z]{3,}", stripped))

    def _is_caption_start_line(self, line: str) -> bool:
        stripped = line.strip()
        if re.match(r"^(fig(?:ure)?\.?|table|scheme)\s*\d+", stripped, flags=re.IGNORECASE):
            return True
        if re.match(r"^\([A-Z]\)\s", stripped):
            return True
        return False

    def _drop_caption_blocks(self, lines: List[str]) -> List[str]:
        cleaned: List[str] = []
        skipping_caption = False

        for line in lines:
            if self._is_caption_start_line(line):
                skipping_caption = True
                continue

            if skipping_caption:
                if self._looks_like_body_line(line):
                    skipping_caption = False
                else:
                    continue

            cleaned.append(line)

        return cleaned

    def clean_text_block(self, text: str, repeated_boundary_lines: Optional[set[str]] = None) -> str:
        normalized = self._normalize_text_artifacts(text)
        lines = [line.strip() for line in normalized.splitlines()]
        cleaned_lines: List[str] = []

        for line in lines:
            if self._is_noise_line(line, repeated_boundary_lines):
                continue
            cleaned_lines.append(line)

        cleaned_lines = self._drop_leading_front_matter(cleaned_lines)
        cleaned_lines = self._drop_caption_blocks(cleaned_lines)
        cleaned_lines = self._merge_wrapped_lines(cleaned_lines)
        cleaned_text = "\n".join(cleaned_lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        return cleaned_text.strip()

    def _truncate_tail_matter(self, text: str) -> Tuple[str, bool]:
        result_lines: List[str] = []
        for line in text.splitlines():
            heading, is_tail, _ = self._parse_heading_line(line)
            if heading and is_tail:
                return "\n".join(result_lines).strip(), True
            result_lines.append(line)
        return "\n".join(result_lines).strip(), False

    def _finalize_section_content(self, lines: List[str]) -> str:
        cleaned = self._merge_wrapped_lines(lines)
        text = "\n".join(cleaned)
        text, _ = self._truncate_tail_matter(text)
        return text.strip()

    def _split_by_detected_headings(self, page_texts: List[Tuple[int, str]]) -> Optional[List[Section]]:
        sections: List[Section] = []
        current_title: Optional[str] = None
        current_lines: List[str] = []
        current_page_start: Optional[int] = None

        def flush_section(end_page: int) -> None:
            nonlocal current_title, current_lines, current_page_start
            if not current_title:
                current_lines = []
                current_page_start = None
                return

            content = self._finalize_section_content(current_lines)
            if content:
                section_id = f"section_{len(sections) + 1}"
                sections.append(
                    Section(
                        title=current_title,
                        content=content,
                        page_start=current_page_start or 1,
                        page_end=end_page,
                        section_id=section_id,
                    )
                )

            current_lines = []
            current_page_start = None
            current_title = None

        for page_num, page_text in page_texts:
            if not page_text.strip():
                continue

            for line_index, line in enumerate(page_text.splitlines()):
                stripped = line.strip()
                if not stripped:
                    continue

                heading, is_tail, remainder = self._parse_heading_line(stripped)
                if heading is None and line_index < 3:
                    for alias, canonical in SECTION_HEADING_ALIASES.items():
                        loose_match = re.match(rf"^{re.escape(alias)}\b(.*)$", stripped, flags=re.IGNORECASE)
                        if loose_match and loose_match.group(1).strip():
                            heading = canonical
                            is_tail = False
                            remainder = loose_match.group(1).strip()
                            break
                if heading:
                    if is_tail:
                        if current_title is not None:
                            flush_section(page_num)
                            return sections or None
                        continue

                    if current_title is not None:
                        flush_section(page_num)

                    current_title = heading
                    current_page_start = page_num
                    current_lines = [remainder] if remainder else []
                    continue

                if current_title is not None:
                    current_lines.append(stripped)

        if current_title is not None:
            flush_section(page_texts[-1][0] if page_texts else 1)

        return sections or None

    def _extract_section_content(
        self,
        pdf: pdfplumber.pdf.PDF,
        start_page: int,
        end_page: int,
        bookmark: BookmarkInfo,
        next_bookmark: Optional[BookmarkInfo] = None,
    ) -> str:
        raw_page_texts: List[str] = []

        for page_num in range(start_page, min(end_page + 1, len(pdf.pages))):
            page = pdf.pages[page_num]
            page_text = self.extract_page_text_smart(page)
            if not page_text:
                continue

            if page_num == start_page:
                page_text = self._remove_title_from_content(page_text, bookmark.title)

            if next_bookmark and next_bookmark.page_num == page_num:
                page_text = self._extract_content_until_next_title(page_text, next_bookmark.title)

            page_text = self._remove_references_content(page_text)
            if page_text.strip():
                raw_page_texts.append(page_text)

        repeated_boundary_lines = self._identify_repeated_boundary_lines(raw_page_texts)
        cleaned_pages = [
            self.clean_text_block(page_text, repeated_boundary_lines)
            for page_text in raw_page_texts
        ]
        return "\n\n".join(page_text for page_text in cleaned_pages if page_text.strip())

    def _extract_content_until_next_title(self, text: str, next_title: str) -> str:
        clean_next_title = self._clean_title(next_title).lower()
        result_lines: List[str] = []

        for line in text.splitlines():
            line_lower = line.lower().strip()
            if clean_next_title and clean_next_title in line_lower:
                break
            if any(keyword in line_lower and len(line_lower) < 50 for keyword in REFERENCE_KEYWORDS):
                break
            result_lines.append(line)

        return "\n".join(result_lines)

    def _remove_references_content(self, text: str) -> str:
        text, _ = self._truncate_tail_matter(text)
        result_lines: List[str] = []
        lines = text.splitlines()

        for index, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            is_reference_title = any(
                line_lower == keyword
                or (line_lower.startswith(keyword) and len(line_stripped) < 30)
                or (keyword in line_lower and len(line_stripped) < 20)
                for keyword in REFERENCE_KEYWORDS
            )

            if not is_reference_title and line_stripped and re.match(r"^\[\d+\]", line_stripped):
                citation_count = 0
                for probe in range(index, min(index + 5, len(lines))):
                    if re.match(r"^\[\d+\]", lines[probe].strip()):
                        citation_count += 1
                is_reference_title = citation_count >= 2

            if is_reference_title:
                break

            result_lines.append(line)

        return "\n".join(result_lines)

    def _remove_title_from_content(self, text: str, title: str) -> str:
        lines = text.splitlines()
        title_clean = self._clean_title(title).lower()

        for index, line in enumerate(lines):
            line_clean = self._clean_title(line).lower()
            if title_clean and line_clean and (title_clean in line_clean or line_clean in title_clean):
                if len(title_clean) > 5 and len(line_clean) > 5:
                    return "\n".join(lines[index + 1:])

        return text

    def _clean_title(self, title: str) -> str:
        title = re.sub(r"^[\d\.]+\s*", "", title or "")
        title = " ".join(title.split())
        return title.strip()

    def _clean_title_for_id(self, title: str) -> str:
        title = self._clean_title(title).lower()
        title = re.sub(r"[^a-z0-9]", "", title)
        return title or "section"

    def _split_text_units(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
        units: List[str] = []

        for paragraph in paragraphs or [text]:
            if len(paragraph) <= self.max_chunk_size:
                units.append(paragraph)
                continue

            sentence_like_parts = [
                part.strip()
                for part in re.split(r"(?<=[\.\!\?。！？])\s+", paragraph)
                if part.strip()
            ]
            units.extend(sentence_like_parts or [paragraph])

        normalized_units: List[str] = []
        for unit in units:
            if len(unit) <= self.max_chunk_size:
                normalized_units.append(unit)
                continue

            start = 0
            while start < len(unit):
                end = min(start + self.max_chunk_size, len(unit))
                if end < len(unit):
                    window = unit[start:end]
                    split_candidates = [
                        window.rfind("\n\n"),
                        window.rfind("\n"),
                        window.rfind(". "),
                        window.rfind("。"),
                        window.rfind(" "),
                    ]
                    best_split = max(split_candidates)
                    if best_split > int(self.max_chunk_size * 0.6):
                        end = start + best_split + 1

                piece = unit[start:end].strip()
                if piece:
                    normalized_units.append(piece)
                start = end

        return normalized_units

    def _build_chunks(self, text: str, chunk_prefix: str) -> List[Dict[str, Any]]:
        text = (text or "").strip()
        if not text:
            return []

        units = self._split_text_units(text) or [text]
        chunks: List[Dict[str, Any]] = []
        current_chunk = ""
        chunk_start = 0
        chunk_num = 1

        def flush_current_chunk() -> None:
            nonlocal current_chunk, chunk_start, chunk_num
            content = current_chunk.strip()
            if not content:
                return

            chunk_id = chunk_prefix if chunk_num == 1 else f"{chunk_prefix}{chunk_num}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "start_char": chunk_start,
                    "end_char": chunk_start + len(content),
                    "word_count": len(content.split()),
                }
            )
            chunk_start += len(content)
            chunk_num += 1
            current_chunk = ""

        for unit in units:
            separator = "\n\n" if current_chunk else ""
            candidate = f"{current_chunk}{separator}{unit}" if current_chunk else unit

            if len(candidate) <= self.max_chunk_size:
                current_chunk = candidate
                continue

            flush_current_chunk()
            current_chunk = unit

            if len(current_chunk) > self.max_chunk_size:
                flush_current_chunk()

        flush_current_chunk()
        return chunks

    def split_sections_into_chunks(self, sections: List[Section]) -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []
        for section in sections:
            all_chunks.append(
                {
                    "section_id": section.section_id,
                    "title": section.title,
                    "bookmark_title": section.bookmark_title,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "level": section.level,
                    "content_length": len(section.content),
                    "chunks": self._build_chunks(section.content, self._clean_title_for_id(section.title)),
                }
            )
        return all_chunks

    def convert_to_target_format(self, sections: List[Section]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for section in sections:
            for chunk in self._build_chunks(section.content, self._clean_title_for_id(section.title)):
                result.append(
                    {
                        "id": chunk["chunk_id"],
                        "metadata": {
                            "lang": "en",
                            "section_title": section.title,
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                        },
                        "text": chunk["content"],
                    }
                )
        return result

    def _extract_sections_with_fallback(self, pdf_path: str) -> Tuple[Optional[List[Section]], str]:
        sections, status = self.split_by_bookmarks(pdf_path)
        if sections is None or len(sections) == 0:
            logger.info("Bookmark split unavailable, falling back to char-count chunking: %s", status)
            sections, status = self.split_by_char_count(pdf_path)
        return sections, status

    def _resolve_output_path(self, pdf_path: str, output_path: Optional[str]) -> str:
        if output_path:
            return output_path
        return f"{Path(pdf_path).stem}.json"

    def process_single_pdf(self, pdf_path: str, output_path: str) -> Dict[str, Any]:
        logger.info("Processing PDF: %s", pdf_path)

        try:
            sections, status = self._extract_sections_with_fallback(pdf_path)
            if sections is None:
                return {"success": False, "error": status}
            if not sections:
                return {"success": False, "error": "No valid sections"}

            result_data = self.convert_to_target_format(sections)
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(result_data, file, indent=2, ensure_ascii=False)

            logger.info("Saved %s sections / %s chunks to %s", len(sections), len(result_data), output_path)
            return {
                "success": True,
                "output_file": output_path,
                "sections": len(sections),
                "chunks": len(result_data),
                "result": result_data,
            }
        except Exception as exc:
            logger.error("Error while processing %s: %s", pdf_path, exc)
            return {"success": False, "error": str(exc)}

    def process_pdf_folder(self, input_folder: str, output_folder: str) -> ProcessingStats:
        os.makedirs(output_folder, exist_ok=True)
        stats = ProcessingStats()

        pdf_files = list(Path(input_folder).glob("*.pdf"))
        stats.total_files = len(pdf_files)
        logger.info("Found %s PDF files in %s", stats.total_files, input_folder)

        for index, pdf_file in enumerate(pdf_files, start=1):
            logger.info("Processing %s/%s: %s", index, stats.total_files, pdf_file.name)
            output_file = Path(output_folder) / f"{pdf_file.stem}.json"
            result = self.process_single_pdf(str(pdf_file), str(output_file))

            if result["success"]:
                stats.processed_successfully += 1
                continue

            error = result["error"]
            if "No bookmark structure found" in error:
                stats.skipped_no_bookmarks += 1
            elif "Too many bookmarks" in error:
                stats.skipped_too_many_bookmarks += 1
            else:
                stats.failed += 1
            stats.skipped_reasons.append(f"{pdf_file.name}: {error}")

        self._print_processing_stats(stats)
        return stats

    def _print_processing_stats(self, stats: ProcessingStats) -> None:
        logger.info("=" * 50)
        logger.info("Batch processing completed")
        logger.info("Total files: %s", stats.total_files)
        logger.info("Processed successfully: %s", stats.processed_successfully)
        logger.info("Skipped (no bookmarks): %s", stats.skipped_no_bookmarks)
        logger.info("Skipped (too many bookmarks): %s", stats.skipped_too_many_bookmarks)
        logger.info("Failed: %s", stats.failed)
        if stats.skipped_reasons:
            logger.info("Detailed skip reasons:")
            for reason in stats.skipped_reasons:
                logger.info("  - %s", reason)
        logger.info("=" * 50)

    def process_pdf(self, pdf_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Compatibility wrapper that now shares the same flat output schema."""
        resolved_output_path = self._resolve_output_path(pdf_path, output_path)
        return self.process_single_pdf(pdf_path, resolved_output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess PDF papers into section-aware JSON chunks")
    parser.add_argument("--input", type=str, default="data/raw_data_papers", help="Input PDF folder")
    parser.add_argument("--output", type=str, default="data/processed_papers", help="Output JSON folder")
    parser.add_argument("--max-bookmarks", type=int, default=20, help="Maximum accepted top-level bookmarks")
    parser.add_argument("--min-bookmarks", type=int, default=3, help="Minimum bookmarks required before fallback")
    parser.add_argument("--max-chunk-size", type=int, default=5000, help="Maximum chunk size in characters")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_folder = project_root / args.input
    output_folder = project_root / args.output

    if not input_folder.exists():
        raise SystemExit(f"Input folder does not exist: {input_folder}")

    splitter = BookmarkBasedSplitter(
        max_chunk_size=args.max_chunk_size,
        max_bookmarks=args.max_bookmarks,
        min_bookmarks=args.min_bookmarks,
    )
    splitter.process_pdf_folder(str(input_folder), str(output_folder))


if __name__ == "__main__":
    main()
