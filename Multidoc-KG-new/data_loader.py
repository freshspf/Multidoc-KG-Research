"""
Data loader for processing JSON paper files.
"""
import json
import os
from typing import List, Optional
from pathlib import Path
from schema import Paper


class PaperDataLoader:
    """Load and process paper data from JSON files."""
    
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
    
    def load_paper_from_json(self, json_file: str) -> Paper:
        """
        Load a single paper from JSON file.
        
        Args:
            json_file: Path to JSON file
            
        Returns:
            Paper object with combined content from all sections
        """
        file_path = self.data_dir / json_file if not os.path.isabs(json_file) else Path(json_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        # Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            sections = json.load(f)
        
        # Extract paper ID from filename (remove extension and prefix)
        paper_id = file_path.stem  # e.g., "1_10.1109_ACCESS.2018.2825378"
        
        # Combine all section texts
        combined_content = []
        title_candidates = []
        
        for section in sections:
            section_id = section.get('id', '')
            metadata = section.get('metadata', {})
            text = section.get('text', '').strip()
            
            # Extract section title from metadata
            section_title = metadata.get('section_title', '')
            if section_title:
                title_candidates.append(section_title)
                combined_content.append(f"## {section_title}\n")
            
            # Add section text
            if text:
                combined_content.append(text)
                combined_content.append("\n\n")  # Add spacing between sections
        
        # Generate paper title (use first section title or filename)
        paper_title = title_candidates[0] if title_candidates else paper_id
        
        # Create Paper object
        paper = Paper(
            id=paper_id,
            title=paper_title,
            content="\n".join(combined_content)
        )
        
        print(f"[DataLoader] Loaded paper: {paper_id}")
        print(f"[DataLoader]   - Sections: {len(sections)}")
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
