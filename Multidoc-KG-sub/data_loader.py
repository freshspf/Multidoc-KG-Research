"""
Data loader for processing biomedical literature JSON files.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from schema import Paper


class PaperDataLoader:
    """Load and process biomedical paper data from JSON files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing JSON paper files
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        print(f"[DataLoader] Initialized with directory: {data_dir}")
    
    def _read_json_file(self, file_path: Path) -> Any:
        """Read JSON payload from disk."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _normalize_keywords(self, raw_keywords: Any) -> List[str]:
        """Normalize keywords to a clean list of strings."""
        if isinstance(raw_keywords, str):
            separators = [";", ",", "|"]
            normalized = raw_keywords
            for separator in separators[1:]:
                normalized = normalized.replace(separator, separators[0])
            return [item.strip() for item in normalized.split(separators[0]) if item.strip()]

        if isinstance(raw_keywords, list):
            return [str(item).strip() for item in raw_keywords if str(item).strip()]

        return []

    def _extract_section_title(self, section: Dict[str, Any]) -> str:
        """Extract a readable section title from a section payload."""
        metadata = section.get('metadata', {}) or {}
        return (
            metadata.get('section_title')
            or section.get('section_title')
            or section.get('title')
            or ""
        ).strip()

    def _extract_sections(self, payload: Any) -> List[Dict[str, Any]]:
        """Normalize different JSON layouts into a section list."""
        if isinstance(payload, list):
            return [section for section in payload if isinstance(section, dict)]

        if isinstance(payload, dict):
            sections = payload.get("sections", [])
            if isinstance(sections, list):
                return [section for section in sections if isinstance(section, dict)]

        return []

    def _infer_title(self, payload: Any, sections: List[Dict[str, Any]], paper_id: str) -> str:
        """Infer the paper title from explicit fields or section metadata."""
        if isinstance(payload, dict):
            metadata = payload.get("metadata", {}) or {}
            candidates = [
                payload.get("title"),
                payload.get("paper_title"),
                payload.get("document_title"),
                metadata.get("title"),
                metadata.get("paper_title"),
                metadata.get("document_title"),
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        for section in sections:
            section_id = str(section.get("id", "")).strip().lower()
            metadata = section.get("metadata", {}) or {}
            section_title = self._extract_section_title(section)
            section_level_title = (
                metadata.get("document_title")
                or metadata.get("paper_title")
                or ""
            ).strip()

            if section_level_title:
                return section_level_title

            if section_id == "title" and section.get("text", "").strip():
                return section.get("text", "").strip()
            if section_id in {"paper_title", "document_title"} and section.get("text", "").strip():
                return section.get("text", "").strip()
            if section_title and section_title.lower() in {"title", "paper title", "document title"}:
                return section_title

        return paper_id

    def _infer_abstract(self, payload: Any, sections: List[Dict[str, Any]]) -> str:
        """Infer abstract text from explicit fields or abstract-like sections."""
        if isinstance(payload, dict):
            metadata = payload.get("metadata", {}) or {}
            candidates = [
                payload.get("abstract"),
                metadata.get("abstract"),
                payload.get("summary"),
            ]
            for candidate in candidates:
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        abstract_chunks: List[str] = []
        for section in sections:
            section_id = str(section.get("id", "")).strip().lower()
            section_title = self._extract_section_title(section).lower()
            if section_id == "abstract" or section_title == "abstract":
                text = section.get("text", "").strip()
                if text:
                    abstract_chunks.append(text)

        return "\n".join(abstract_chunks).strip()

    def _infer_keywords(self, payload: Any, sections: List[Dict[str, Any]]) -> List[str]:
        """Infer keywords from explicit fields or keyword-like sections."""
        if isinstance(payload, dict):
            metadata = payload.get("metadata", {}) or {}
            keyword_sources = [
                payload.get("keywords"),
                payload.get("keyword"),
                metadata.get("keywords"),
                metadata.get("keyword"),
                payload.get("mesh_terms"),
                metadata.get("mesh_terms"),
            ]
            for source in keyword_sources:
                keywords = self._normalize_keywords(source)
                if keywords:
                    return keywords

        for section in sections:
            section_id = str(section.get("id", "")).strip().lower()
            section_title = self._extract_section_title(section).lower()
            if "keyword" in section_id or "keyword" in section_title:
                keywords = self._normalize_keywords(section.get("text", ""))
                if keywords:
                    return keywords

        return []

    def _combine_section_text(self, sections: List[Dict[str, Any]]) -> str:
        """Combine section texts into a readable full-document string."""
        combined_content: List[str] = []

        for section in sections:
            text = section.get('text', '').strip()
            section_title = self._extract_section_title(section)

            if section_title:
                combined_content.append(f"## {section_title}\n")

            if text:
                combined_content.append(text)
                combined_content.append("\n\n")

        return "\n".join(combined_content).strip()

    def load_paper_from_json(self, json_file: str) -> Paper:
        """
        Load a single biomedical paper from JSON file.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            Paper object with combined full text plus title/abstract/keyword metadata
        """
        file_path = self.data_dir / json_file if not os.path.isabs(json_file) else Path(json_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        payload = self._read_json_file(file_path)
        paper_id = file_path.stem
        sections = self._extract_sections(payload)
        paper_title = self._infer_title(payload, sections, paper_id)
        abstract = self._infer_abstract(payload, sections)
        keywords = self._infer_keywords(payload, sections)
        combined_content = self._combine_section_text(sections)

        metadata: Dict[str, Any] = {
            "abstract": abstract,
            "keywords": keywords,
            "source_path": str(file_path),
            "section_count": len(sections),
            "sections": [
                {
                    "id": section.get("id", ""),
                    "section_title": self._extract_section_title(section),
                }
                for section in sections
            ],
        }

        if isinstance(payload, dict):
            source_metadata = payload.get("metadata", {}) or {}
            for field in ["pmid", "doi", "journal", "year", "authors", "mesh_terms"]:
                value = payload.get(field, source_metadata.get(field))
                if value not in (None, "", []):
                    metadata[field] = value

        if not combined_content:
            combined_content = abstract or paper_title

        paper = Paper(
            id=paper_id,
            title=paper_title,
            content=combined_content,
            metadata=metadata,
        )
        
        print(f"[DataLoader] Loaded paper: {paper_id}")
        print(f"[DataLoader]   - Title: {paper.title[:100]}")
        print(f"[DataLoader]   - Sections: {len(sections)}")
        print(f"[DataLoader]   - Abstract length: {len(abstract)} chars")
        print(f"[DataLoader]   - Keywords: {len(keywords)}")
        print(f"[DataLoader]   - Content length: {len(paper.content)} chars")
        
        return paper
    
    def load_all_papers(self, pattern: str = "*.json") -> List[Paper]:
        """
        Load all papers from the data directory.
        
        Args:
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            List of Paper objects
        """
        json_files = sorted(self.data_dir.glob(pattern))
        
        if not json_files:
            print(f"[DataLoader] Warning: No JSON files found in {self.data_dir}")
            return []
        
        print(f"[DataLoader] Found {len(json_files)} JSON files")
        
        papers = []
        for json_file in json_files:
            try:
                # Pass only the filename, not the full path
                paper = self.load_paper_from_json(json_file.name)
                papers.append(paper)
            except Exception as e:
                print(f"[DataLoader] Error loading {json_file.name}: {e}")
        
        print(f"[DataLoader] Successfully loaded {len(papers)} papers")
        return papers
    
    def load_specific_papers(self, filenames: List[str]) -> List[Paper]:
        """
        Load specific papers by filenames.
        
        Args:
            filenames: List of JSON filenames to load
            
        Returns:
            List of Paper objects
        """
        papers = []
        for filename in filenames:
            try:
                paper = self.load_paper_from_json(filename)
                papers.append(paper)
            except Exception as e:
                print(f"[DataLoader] Error loading {filename}: {e}")
        
        print(f"[DataLoader] Successfully loaded {len(papers)} papers")
        return papers
