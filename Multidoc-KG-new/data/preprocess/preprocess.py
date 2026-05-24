#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bookmark-based intelligent PDF section splitter
Supports single and double column layouts, uses PDF bookmark structure for precise segmentation
Batch processing of PDF files with standardized JSON output
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import pdfplumber
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BookmarkInfo:
    """Bookmark information"""
    title: str
    page_num: int
    level: int
    
@dataclass
class Section:
    """Section information"""
    title: str
    content: str
    page_start: int
    page_end: int
    section_id: str
    level: int = 1
    bookmark_title: str = ""

@dataclass
class ProcessingStats:
    """Processing statistics"""
    total_files: int = 0
    processed_successfully: int = 0
    skipped_no_bookmarks: int = 0
    skipped_too_many_bookmarks: int = 0
    failed: int = 0
    skipped_reasons: List[str] = None
    
    def __post_init__(self):
        if self.skipped_reasons is None:
            self.skipped_reasons = []

class BookmarkBasedSplitter:
    """Bookmark-based PDF splitter"""
    
    def __init__(self, max_chunk_size: int = 5000, max_bookmarks: int = 15):
        self.max_chunk_size = max_chunk_size
        self.max_bookmarks = max_bookmarks
        self.bookmarks = []
        
    def extract_bookmarks(self, pdf_path: str) -> List[BookmarkInfo]:
        """Extract PDF bookmark information"""
        bookmarks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if not pdf_reader.outline:
                    logger.warning("PDF has no bookmark structure")
                    return bookmarks
                
                def process_bookmarks(outline, level=1):
                    for item in outline:
                        if isinstance(item, list):
                            # Skip sub-level bookmarks, only process top-level
                            if level == 1:
                                process_bookmarks(item, level + 1)
                        else:
                            title = item.title if hasattr(item, 'title') else str(item)
                            
                            # Only process top-level bookmarks
                            if level > 1:
                                continue
                            
                            # Get page number
                            try:
                                if hasattr(item, 'page'):
                                    page_obj = item.page
                                    
                                    # Try multiple methods to get page number
                                    page_num = None
                                    
                                    # Method 1: Direct from page object
                                    if hasattr(page_obj, 'page'):
                                        page_num = page_obj.page
                                    
                                    # Method 2: Find by page reference
                                    elif hasattr(page_obj, 'idnum'):
                                        for i, page in enumerate(pdf_reader.pages):
                                            if hasattr(page, 'idnum') and page.idnum == page_obj.idnum:
                                                page_num = i
                                                break
                                    
                                    # Method 3: Use PyPDF2 built-in method
                                    if page_num is None:
                                        try:
                                            page_num = pdf_reader.get_destination_page_number(item)
                                        except:
                                            pass
                                    
                                    # Method 4: Try direct parsing from page object
                                    if page_num is None:
                                        try:
                                            # Get page object index in PDF
                                            for i, page in enumerate(pdf_reader.pages):
                                                if page == page_obj:
                                                    page_num = i
                                                    break
                                        except:
                                            pass
                                    
                                    # Method 5: Estimate page from bookmark position
                                    if page_num is None:
                                        # Estimate page position based on processed bookmarks
                                        estimated_page = len(bookmarks) * 2  # Rough estimate: one section per 2 pages
                                        if estimated_page < len(pdf_reader.pages):
                                            page_num = estimated_page
                                    
                                    if page_num is not None:
                                        bookmarks.append(BookmarkInfo(
                                            title=title,
                                            page_num=page_num,
                                            level=level
                                        ))
                                        logger.info(f"Found top-level bookmark: {title} (page {page_num + 1})")
                                    else:
                                        logger.warning(f"Cannot determine page number for bookmark '{title}'")
                                        
                            except Exception as e:
                                logger.warning(f"Error processing bookmark '{title}': {e}")
                
                process_bookmarks(pdf_reader.outline)
                
        except Exception as e:
            logger.error(f"Error extracting bookmarks: {e}")
            
        return bookmarks
    
    def filter_bookmarks_by_title(self, bookmarks: List[BookmarkInfo]) -> List[BookmarkInfo]:
        """Filter out section titles without capitalized first letter"""
        filtered_bookmarks = []
        
        for bookmark in bookmarks:
            title = bookmark.title.strip()
            
            # Special handling for Abstract (case-insensitive)
            if re.match(r'^abstract\b', title, re.IGNORECASE):
                filtered_bookmarks.append(bookmark)
                logger.info(f"Keeping Abstract section: {title}")
                continue
            
            # Check first letter of other sections
            if self._has_valid_title_case(title):
                filtered_bookmarks.append(bookmark)
                logger.info(f"Keeping section: {title}")
            else:
                logger.info(f"Filtering out non-capitalized section: {title}")
        
        return filtered_bookmarks
    
    def _has_valid_title_case(self, title: str) -> bool:
        """Check if title follows capitalization rules"""
        title = title.strip()
        
        # Remove leading numbers and symbols, find first letter
        match = re.match(r'^[\d\.\s]*([a-zA-Z])', title)
        if match:
            first_letter = match.group(1)
            return first_letter.isupper()
        
        return False
    
    def remove_references_and_after(self, bookmarks: List[BookmarkInfo]) -> List[BookmarkInfo]:
        """Remove Reference section and all sections after it"""
        reference_keywords = ['reference', 'references', 'bibliography']
        
        for i, bookmark in enumerate(bookmarks):
            title_lower = bookmark.title.lower().strip()
            
            # Check if contains reference-related keywords
            for keyword in reference_keywords:
                if keyword in title_lower:
                    logger.info(f"Found Reference section: {bookmark.title}, removing this and all following sections")
                    return bookmarks[:i]
        
        # If no reference section found, return all bookmarks
        return bookmarks
    
    def detect_column_layout(self, page) -> bool:
        """Detect if page is double-column layout"""
        chars = page.chars
        if not chars:
            return False
        
        # Count x-coordinate distribution
        x_positions = [char['x0'] for char in chars if char['text'].strip()]
        if not x_positions:
            return False
        
        page_width = page.width
        mid_x = page_width / 2
        
        # Count characters on left and right sides
        left_chars = sum(1 for x in x_positions if x < mid_x)
        right_chars = sum(1 for x in x_positions if x >= mid_x)
        
        total_chars = len(x_positions)
        return (left_chars / total_chars > 0.3 and right_chars / total_chars > 0.3)
    
    def extract_page_text_smart(self, page) -> str:
        """Smart text extraction (supports single/double column)"""
        is_two_column = self.detect_column_layout(page)
        
        if is_two_column:
            return self._extract_two_column_text(page)
        else:
            return page.extract_text() or ""
    
    def _extract_two_column_text(self, page) -> str:
        """Extract double-column text"""
        chars = page.chars
        if not chars:
            return ""
        
        page_width = page.width
        mid_x = page_width / 2
        
        # Separate left and right column characters
        left_chars = [c for c in chars if c['x0'] < mid_x]
        right_chars = [c for c in chars if c['x0'] >= mid_x]
        
        # Process left and right columns separately
        left_text = self._chars_to_text(left_chars)
        right_text = self._chars_to_text(right_chars)
        
        # Merge left and right column text
        if left_text and right_text:
            return left_text + "\n\n" + right_text
        elif left_text:
            return left_text
        else:
            return right_text
    
    def _chars_to_text(self, chars) -> str:
        """Convert character list to text"""
        if not chars:
            return ""
        
        # Sort by y-coordinate (top to bottom) and x-coordinate (left to right)
        sorted_chars = sorted(chars, key=lambda c: (-c['y0'], c['x0']))
        
        # Group into lines
        lines = []
        current_line_chars = []
        current_y = None
        
        for char in sorted_chars:
            y = round(char['y0'], 1)
            
            if current_y is None:
                current_y = y
                current_line_chars.append(char)
            elif abs(y - current_y) <= 3:  # Same line
                current_line_chars.append(char)
            else:  # New line
                if current_line_chars:
                    lines.append(self._build_spaced_line_text(current_line_chars))
                current_line_chars = [char]
                current_y = y
        
        if current_line_chars:
            lines.append(self._build_spaced_line_text(current_line_chars))
        
        return '\n'.join(lines)
    
    def _build_spaced_line_text(self, chars) -> str:
        """Build line text with appropriate spacing"""
        if not chars:
            return ""
        
        # Sort by x-coordinate to ensure correct order
        chars = sorted(chars, key=lambda c: c['x0'])
        
        result = []
        prev_char = None
        
        for char in chars:
            current_text = char['text']
            
            if prev_char is not None:
                # Calculate character spacing
                gap = char['x0'] - prev_char['x1']
                
                # Calculate average character width
                prev_width = prev_char['x1'] - prev_char['x0'] if prev_char['x1'] > prev_char['x0'] else 6
                current_width = char['x1'] - char['x0'] if char['x1'] > char['x0'] else 6
                avg_char_width = (prev_width + current_width) / 2
                
                # Add space if gap is larger than certain proportion of character width
                if gap > avg_char_width * 0.3:  # Lower threshold for easier spacing
                    if gap > avg_char_width * 1.5:
                        result.append('  ')  # Large gap: two spaces
                    else:
                        result.append(' ')   # Normal gap: one space
            
            result.append(current_text)
            prev_char = char
        
        return ''.join(result)
    
    def split_by_bookmarks(self, pdf_path: str) -> Tuple[Optional[List[Section]], str]:
        """Split PDF by bookmarks, return section list and status"""
        logger.info(f"Starting bookmark-based PDF splitting: {pdf_path}")
        
        # 1. Extract bookmarks
        bookmarks = self.extract_bookmarks(pdf_path)
        if not bookmarks:
            return None, "No bookmark structure found"
        
        logger.info(f"Found {len(bookmarks)} raw bookmarks")
        
        # 2. Filter non-capitalized sections
        bookmarks = self.filter_bookmarks_by_title(bookmarks)
        logger.info(f"After capitalization filter: {len(bookmarks)} bookmarks remaining")
        
        # 3. Check bookmark count
        if len(bookmarks) < 3:
            return None, f"Too few bookmarks: {len(bookmarks)} < 3"
        if len(bookmarks) > self.max_bookmarks:
            return None, f"Too many bookmarks: {len(bookmarks)} > {self.max_bookmarks}"
        
        # 4. Remove Reference section and after
        bookmarks = self.remove_references_and_after(bookmarks)
        logger.info(f"After removing References: {len(bookmarks)} bookmarks remaining")
        
        # 5. Sort bookmarks by page number
        bookmarks.sort(key=lambda x: x.page_num)
        
        # 6. Extract content for each section
        sections = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for i, bookmark in enumerate(bookmarks):
                    # Determine section start and end pages
                    start_page = bookmark.page_num
                    
                    # Find next same-level or higher bookmark as end point
                    end_page = total_pages - 1  # Default to document end
                    next_bookmark = None
                    for j in range(i + 1, len(bookmarks)):
                        candidate = bookmarks[j]
                        if candidate.level <= bookmark.level:
                            next_bookmark = candidate
                            # If next bookmark is on same page, use current page
                            if candidate.page_num == bookmark.page_num:
                                end_page = bookmark.page_num
                            else:
                                end_page = candidate.page_num - 1
                            break
                    
                    # Extract section content
                    try:
                        content = self._extract_section_content(
                            pdf, start_page, end_page, bookmark, next_bookmark
                        )
                        
                        if content.strip():  # Only add non-empty sections
                            section = Section(
                                title=self._clean_title(bookmark.title),
                                content=content,
                                page_start=start_page + 1,  # Convert to 1-based
                                page_end=end_page + 1,
                                section_id=f"section_{i+1}",
                                level=bookmark.level,
                                bookmark_title=bookmark.title
                            )
                            sections.append(section)
                            logger.info(f"Successfully extracted section '{bookmark.title}': {len(content)} characters")
                        else:
                            logger.warning(f"Skipping empty section '{bookmark.title}'")
                    except Exception as e:
                        logger.warning(f"Skipping section '{bookmark.title}': {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Error splitting PDF: {e}")
            return None, f"Processing error: {e}"
        
        logger.info(f"Successfully split into {len(sections)} sections")
        return sections, "success"
    
    def _extract_section_content(self, pdf, start_page: int, end_page: int, bookmark: BookmarkInfo, next_bookmark: BookmarkInfo = None) -> str:
        """Extract section content"""
        content_parts = []
        
        for page_num in range(start_page, min(end_page + 1, len(pdf.pages))):
            page = pdf.pages[page_num]
            page_text = self.extract_page_text_smart(page)
            
            if page_text:
                # If first page, try to remove title part
                if page_num == start_page:
                    page_text = self._remove_title_from_content(page_text, bookmark.title)
                
                # If next bookmark is on same page, extract until next title
                if next_bookmark and next_bookmark.page_num == page_num:
                    page_text = self._extract_content_until_next_title(page_text, next_bookmark.title)
                
                # Always check and remove References section
                page_text = self._remove_references_content(page_text)
                
                content_parts.append(page_text)
        
        return '\n\n'.join(content_parts)
    
    def _extract_content_until_next_title(self, text: str, next_title: str) -> str:
        """Extract text until next title appears"""
        # Clean title (remove number prefix, etc.)
        clean_next_title = self._clean_title(next_title)
        
        lines = text.split('\n')
        result_lines = []
        
        # Check for reference keywords
        reference_keywords = ['reference', 'references', 'bibliography']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if contains next title
            if clean_next_title.lower() in line.lower():
                # Found next title, stop adding content
                break
            
            # Check if contains reference keywords
            is_reference_line = False
            for keyword in reference_keywords:
                if keyword in line_lower and len(line.strip()) < 50:  # Short lines more likely to be titles
                    is_reference_line = True
                    break
            
            if is_reference_line:
                # Found References title, stop adding content
                break
                
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_references_content(self, text: str) -> str:
        """Remove References section and content after it"""
        reference_keywords = ['references', 'reference', 'bibliography']
        
        lines = text.split('\n')
        result_lines = []
        reference_pattern_detected = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Check if it's References title line
            is_reference_title = False
            for keyword in reference_keywords:
                # Exact match or short line starting with keyword
                if (line_lower == keyword or 
                    (line_lower.startswith(keyword) and len(line_stripped) < 30) or
                    (keyword in line_lower and len(line_stripped) < 20)):
                    is_reference_title = True
                    break
            
            # Extra check: if line only contains "References" and some numbers/symbols
            if not is_reference_title and len(line_stripped) < 30:
                for keyword in reference_keywords:
                    clean_line = ''.join(c for c in line_lower if c.isalpha())
                    if clean_line == keyword:
                        is_reference_title = True
                        break
            
            # Check if it's citation list start (e.g., [1], [32] patterns)
            is_reference_list_start = False
            if not is_reference_title and line_stripped:
                # Check if starts with [number]
                import re
                if re.match(r'^\[\d+\]', line_stripped):
                    # Check if following lines also have similar pattern
                    citation_count = 0
                    # Check next few lines
                    for j in range(i, min(i + 5, len(lines))):
                        if re.match(r'^\[\d+\]', lines[j].strip()):
                            citation_count += 1
                    
                    # If multiple consecutive lines are in citation format, consider this as citation list start
                    if citation_count >= 2:
                        is_reference_list_start = True
            
            if is_reference_title or is_reference_list_start:
                # Found References title or citation list start, stop adding content
                break
                
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_title_from_content(self, text: str, title: str) -> str:
        """Remove title part from content"""
        lines = text.split('\n')
        
        # Find line containing title
        title_clean = self._clean_title(title).lower()
        
        for i, line in enumerate(lines):
            line_clean = self._clean_title(line).lower()
            
            # If title line found, return content from next line
            if title_clean in line_clean or line_clean in title_clean:
                if len(title_clean) > 5 and len(line_clean) > 5:  # Avoid mismatching short text
                    return '\n'.join(lines[i+1:])
        
        return text  # If title not found, return original text
    
    def _clean_title(self, title: str) -> str:
        """Clean title text"""
        # Remove number prefix
        title = re.sub(r'^[\d\.]+\s*', '', title)
        # Remove extra whitespace
        title = ' '.join(title.split())
        return title.strip()
    
    def _fallback_to_full_text(self, pdf_path: str) -> List[Section]:
        """Fallback: extract entire document"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                full_content = []
                for page in pdf.pages:
                    page_text = self.extract_page_text_smart(page)
                    if page_text:
                        full_content.append(page_text)
                
                content = '\n\n'.join(full_content)
                
                return [Section(
                    title="Complete Document",
                    content=content,
                    page_start=1,
                    page_end=len(pdf.pages),
                    section_id="section_1"
                )]
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            return []
    
    def split_sections_into_chunks(self, sections: List[Section]) -> List[Dict[str, Any]]:
        """Further split sections into smaller chunks"""
        all_chunks = []
        
        for section in sections:
            chunks = self._chunk_text_by_section(section.content, section.title)
            
            section_data = {
                "section_id": section.section_id,
                "title": section.title,
                "bookmark_title": section.bookmark_title,
                "page_start": section.page_start,
                "page_end": section.page_end,
                "level": section.level,
                "content_length": len(section.content),
                "chunks": chunks
            }
            all_chunks.append(section_data)
        
        return all_chunks
    
    def _clean_title_for_id(self, title: str) -> str:
        """Clean title for ID, convert to lowercase and remove special characters"""
        # Remove number prefix
        title = re.sub(r'^[\d\.]+\s*', '', title)
        # Remove extra whitespace
        title = ' '.join(title.split())
        # Convert to lowercase
        title = title.lower()
        # Remove special characters, keep only letters and numbers
        title = re.sub(r'[^a-z0-9]', '', title)
        return title.strip()
    
    def _chunk_text_by_section(self, text: str, section_title: str) -> List[Dict[str, Any]]:
        """Split text by section name + number"""
        # Clean section title for ID
        clean_title = self._clean_title_for_id(section_title)
        
        if len(text) <= self.max_chunk_size:
            return [{
                "chunk_id": clean_title,
                "content": text,
                "start_char": 0,
                "end_char": len(text),
                "word_count": len(text.split())
            }]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        chunk_start = 0
        chunk_num = 1
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{clean_title}{chunk_num}",
                        "content": current_chunk.strip(),
                        "start_char": chunk_start,
                        "end_char": chunk_start + len(current_chunk),
                        "word_count": len(current_chunk.split())
                    })
                    chunk_start += len(current_chunk)
                    chunk_num += 1
                
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append({
                "chunk_id": f"{clean_title}{chunk_num}",
                "content": current_chunk.strip(),
                "start_char": chunk_start,
                "end_char": chunk_start + len(current_chunk),
                "word_count": len(current_chunk.split())
            })
        
        return chunks
    
    def _chunk_text(self, text: str, section_id: str) -> List[Dict[str, Any]]:
        """Split text into smaller chunks"""
        if len(text) <= self.max_chunk_size:
            return [{
                "chunk_id": f"{section_id}_chunk_1",
                "content": text,
                "start_char": 0,
                "end_char": len(text),
                "word_count": len(text.split())
            }]
        
        chunks = []
        sentences = text.split('. ')
        current_chunk = ""
        chunk_start = 0
        chunk_num = 1
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append({
                        "chunk_id": f"{section_id}_chunk_{chunk_num}",
                        "content": current_chunk.strip(),
                        "start_char": chunk_start,
                        "end_char": chunk_start + len(current_chunk),
                        "word_count": len(current_chunk.split())
                    })
                    chunk_start += len(current_chunk)
                    chunk_num += 1
                
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append({
                "chunk_id": f"{section_id}_chunk_{chunk_num}",
                "content": current_chunk.strip(),
                "start_char": chunk_start,
                "end_char": chunk_start + len(current_chunk),
                "word_count": len(current_chunk.split())
            })
        
        return chunks
    
    def convert_to_target_format(self, sections: List[Section]) -> List[Dict[str, Any]]:
        """Convert to target JSON format using new chunking logic"""
        result = []
        
        for section in sections:
            # Use new chunking method
            chunks = self._chunk_text_by_section(section.content, section.title)
            
            # Create separate entry for each chunk
            for chunk in chunks:
                section_data = {
                    "id": chunk["chunk_id"],  # Use new ID format (section name + number)
                    "metadata": {
                        "lang": "en",
                        "section_title": section.title,
                        "page_start": section.page_start,
                        "page_end": section.page_end
                    },
                    "text": chunk["content"]
                }
                result.append(section_data)
        
        return result
    
    def process_single_pdf(self, pdf_path: str, output_path: str) -> Dict[str, Any]:
        """Process single PDF file"""
        logger.info(f"Starting to process PDF: {pdf_path}")
        
        try:
            # Split sections
            sections, status = self.split_by_bookmarks(pdf_path)
            
            if sections is None:
                return {"success": False, "error": status}
            
            if not sections:
                return {"success": False, "error": "No valid sections"}
            
            # Convert to target format
            result_data = self.convert_to_target_format(sections)
            
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processing complete, results saved to: {output_path}")
            logger.info(f"Stats: {len(sections)} sections, {len(result_data)} chunks")
            
            return {"success": True, "output_file": output_path, "sections": len(sections), "chunks": len(result_data)}
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"success": False, "error": str(e)}
    
    def process_pdf_folder(self, input_folder: str, output_folder: str) -> ProcessingStats:
        """Batch process PDF folder"""
        logger.info(f"Starting batch processing of PDF folder: {input_folder}")
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Statistics
        stats = ProcessingStats()
        
        # Get all PDF files
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        stats.total_files = len(pdf_files)
        
        logger.info(f"Found {stats.total_files} PDF files")
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"\nProcessing progress: {i+1}/{stats.total_files} - {pdf_file.name}")
            
            # Generate output filename: use original filename (without .pdf extension) + .json
            output_file = Path(output_folder) / f"{pdf_file.stem}.json"
            
            # Process single PDF
            result = self.process_single_pdf(str(pdf_file), str(output_file))
            
            if result["success"]:
                stats.processed_successfully += 1
                logger.info(f"Successfully processed: {pdf_file.name}")
            else:
                error = result["error"]
                if "No bookmark structure found" in error:
                    stats.skipped_no_bookmarks += 1
                    stats.skipped_reasons.append(f"{pdf_file.name}: No bookmark structure")
                elif "Too many bookmarks" in error:
                    stats.skipped_too_many_bookmarks += 1
                    stats.skipped_reasons.append(f"{pdf_file.name}: {error}")
                else:
                    stats.failed += 1
                    stats.skipped_reasons.append(f"{pdf_file.name}: {error}")
                
                logger.warning(f"Skipped file: {pdf_file.name} - {error}")
        
        # Print statistics
        self._print_processing_stats(stats)
        
        return stats
    
    def _print_processing_stats(self, stats: ProcessingStats):
        """Print processing statistics"""
        logger.info(f"\n{'='*50}")
        logger.info(f"Batch processing complete:")
        logger.info(f"Total files: {stats.total_files}")
        logger.info(f"Successfully processed: {stats.processed_successfully}")
        logger.info(f"Skipped (no bookmarks): {stats.skipped_no_bookmarks}")
        logger.info(f"Skipped (too many bookmarks): {stats.skipped_too_many_bookmarks}")
        logger.info(f"Failed: {stats.failed}")
        
        if stats.skipped_reasons:
            logger.info(f"\nDetailed skip reasons:")
            for reason in stats.skipped_reasons:
                logger.info(f"  - {reason}")
        
        logger.info(f"{'='*50}")
    
    def process_pdf(self, pdf_path: str, output_path: str = None) -> Dict[str, Any]:
        """Complete PDF file processing"""
        logger.info(f"Starting to process PDF: {pdf_path}")
        
        if not output_path:
            pdf_name = Path(pdf_path).stem
            output_path = f"{pdf_name}_bookmark_sections.json"
        
        try:
            # Split sections
            sections = self.split_by_bookmarks(pdf_path)
            
            if not sections:
                logger.error("Failed to split into any sections")
                return {"success": False, "error": "No sections found"}
            
            # Chunk
            section_chunks = self.split_sections_into_chunks(sections)
            
            # Statistics
            total_chunks = sum(len(section["chunks"]) for section in section_chunks)
            total_words = sum(chunk["word_count"] for section in section_chunks for chunk in section["chunks"])
            
            # Build result
            result = {
                "metadata": {
                    "title": "",
                    "authors": [],
                    "abstract": "",
                    "keywords": [],
                    "total_pages": 0,
                    "extraction_method": "bookmark_based",
                    "extraction_time": datetime.now().isoformat()
                },
                "statistics": {
                    "total_sections": len(section_chunks),
                    "total_chunks": total_chunks,
                    "total_words": total_words,
                    "processing_time": datetime.now().isoformat()
                },
                "sections": section_chunks
            }
            
            # Save results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processing complete, results saved to: {output_path}")
            logger.info(f"Stats: {len(section_chunks)} sections, {total_chunks} chunks, {total_words} words")
            
            return {"success": True, "output_file": output_path, "result": result}
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"success": False, "error": str(e)}

def main():
    """Main function - default batch processing mode"""
    # Set default paths
    input_folder = "/path/to/input/papers"
    output_folder = "/path/to/output/processed_papers"
    max_bookmarks = 200
    
    print(f"PDF Batch Processor")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Max bookmarks: {max_bookmarks}")
    print(f"Min bookmarks: 1")
    print("=" * 50)
    
    # Check input path
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return
    
    if not input_path.is_dir():
        print(f"Error: Input path is not a directory: {input_path}")
        return
    
    # Create splitter and start batch processing
    splitter = BookmarkBasedSplitter(max_bookmarks=max_bookmarks)
    output_path = Path(output_folder)
    
    stats = splitter.process_pdf_folder(str(input_path), str(output_path))
    
    print(f"\nFinal Statistics:")
    print(f"Successfully processed: {stats.processed_successfully}")
    print(f"Skipped files: {stats.skipped_no_bookmarks + stats.skipped_too_many_bookmarks + stats.failed}")
    print(f"Total files: {stats.total_files}")
    print("=" * 50)

if __name__ == "__main__":
    main()
