#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于PDF书签的智能章节分割器
支持单栏和双栏文章，使用PDF书签结构进行精确分段
批量处理PDF文件，输出标准化的JSON格式
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BookmarkInfo:
    """书签信息"""
    title: str
    page_num: int
    level: int
    
@dataclass
class Section:
    """章节信息"""
    title: str
    content: str
    page_start: int
    page_end: int
    section_id: str
    level: int = 1
    bookmark_title: str = ""

@dataclass
class ProcessingStats:
    """处理统计信息"""
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
    """基于书签的PDF分割器"""
    
    def __init__(self, max_chunk_size: int = 5000, max_bookmarks: int = 15):
        self.max_chunk_size = max_chunk_size
        self.max_bookmarks = max_bookmarks
        self.bookmarks = []
        
    def extract_bookmarks(self, pdf_path: str) -> List[BookmarkInfo]:
        """提取PDF书签信息"""
        bookmarks = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                if not pdf_reader.outline:
                    logger.warning("PDF没有书签结构")
                    return bookmarks
                
                def process_bookmarks(outline, level=1):
                    for item in outline:
                        if isinstance(item, list):
                            # 跳过子级书签，只处理一级标题
                            if level == 1:
                                process_bookmarks(item, level + 1)
                        else:
                            title = item.title if hasattr(item, 'title') else str(item)
                            
                            # 只处理一级标题
                            if level > 1:
                                continue
                            
                            # 获取页面号
                            try:
                                if hasattr(item, 'page'):
                                    page_obj = item.page
                                    
                                    # 尝试多种方法获取页面号
                                    page_num = None
                                    
                                    # 方法1: 直接从page对象获取
                                    if hasattr(page_obj, 'page'):
                                        page_num = page_obj.page
                                    
                                    # 方法2: 通过页面引用查找
                                    elif hasattr(page_obj, 'idnum'):
                                        for i, page in enumerate(pdf_reader.pages):
                                            if hasattr(page, 'idnum') and page.idnum == page_obj.idnum:
                                                page_num = i
                                                break
                                    
                                    # 方法3: 使用PyPDF2的内置方法
                                    if page_num is None:
                                        try:
                                            page_num = pdf_reader.get_destination_page_number(item)
                                        except:
                                            pass
                                    
                                    # 方法4: 尝试从页面对象直接解析
                                    if page_num is None:
                                        try:
                                            # 获取页面对象在PDF中的索引
                                            for i, page in enumerate(pdf_reader.pages):
                                                if page == page_obj:
                                                    page_num = i
                                                    break
                                        except:
                                            pass
                                    
                                    # 方法5: 使用书签的位置信息推断页面
                                    if page_num is None:
                                        # 根据已处理的书签估算页面位置
                                        estimated_page = len(bookmarks) * 2  # 粗略估计每2页一个章节
                                        if estimated_page < len(pdf_reader.pages):
                                            page_num = estimated_page
                                    
                                    if page_num is not None:
                                        bookmarks.append(BookmarkInfo(
                                            title=title,
                                            page_num=page_num,
                                            level=level
                                        ))
                                        logger.info(f"找到一级书签: {title} (页面 {page_num + 1})")
                                    else:
                                        logger.warning(f"无法确定书签 '{title}' 的页面号")
                                        
                            except Exception as e:
                                logger.warning(f"处理书签 '{title}' 时出错: {e}")
                
                process_bookmarks(pdf_reader.outline)
                
        except Exception as e:
            logger.error(f"提取书签时出错: {e}")
            
        return bookmarks
    
    def filter_bookmarks_by_title(self, bookmarks: List[BookmarkInfo]) -> List[BookmarkInfo]:
        """过滤首字母未大写的章节标题"""
        filtered_bookmarks = []
        
        for bookmark in bookmarks:
            title = bookmark.title.strip()
            
            # 特殊处理Abstract（不区分大小写）
            if re.match(r'^abstract\b', title, re.IGNORECASE):
                filtered_bookmarks.append(bookmark)
                logger.info(f"保留Abstract章节: {title}")
                continue
            
            # 检查其他章节的首字母
            if self._has_valid_title_case(title):
                filtered_bookmarks.append(bookmark)
                logger.info(f"保留章节: {title}")
            else:
                logger.info(f"过滤掉首字母未大写的章节: {title}")
        
        return filtered_bookmarks
    
    def _has_valid_title_case(self, title: str) -> bool:
        """检查标题是否符合首字母大写规则"""
        title = title.strip()
        
        # 移除开头的数字和符号，找到第一个字母
        match = re.match(r'^[\d\.\s]*([a-zA-Z])', title)
        if match:
            first_letter = match.group(1)
            return first_letter.isupper()
        
        return False
    
    def remove_references_and_after(self, bookmarks: List[BookmarkInfo]) -> List[BookmarkInfo]:
        """移除Reference章节及之后的所有章节"""
        reference_keywords = ['reference', 'references', 'bibliography']
        
        for i, bookmark in enumerate(bookmarks):
            title_lower = bookmark.title.lower().strip()
            
            # 检查是否包含reference相关关键词
            for keyword in reference_keywords:
                if keyword in title_lower:
                    logger.info(f"找到Reference章节: {bookmark.title}，移除该章节及之后内容")
                    return bookmarks[:i]
        
        # 如果没找到reference章节，返回所有章节
        return bookmarks
    
    def detect_column_layout(self, page) -> bool:
        """检测页面是否为双栏布局"""
        chars = page.chars
        if not chars:
            return False
        
        # 统计字符的x坐标分布
        x_positions = [char['x0'] for char in chars if char['text'].strip()]
        if not x_positions:
            return False
        
        page_width = page.width
        mid_x = page_width / 2
        
        # 统计左右两侧的字符数量
        left_chars = sum(1 for x in x_positions if x < mid_x)
        right_chars = sum(1 for x in x_positions if x >= mid_x)
        
        total_chars = len(x_positions)
        return (left_chars / total_chars > 0.3 and right_chars / total_chars > 0.3)
    
    def extract_page_text_smart(self, page) -> str:
        """智能提取页面文本（支持单栏/双栏）"""
        is_two_column = self.detect_column_layout(page)
        
        if is_two_column:
            return self._extract_two_column_text(page)
        else:
            return page.extract_text() or ""
    
    def _extract_two_column_text(self, page) -> str:
        """提取双栏文本"""
        chars = page.chars
        if not chars:
            return ""
        
        page_width = page.width
        mid_x = page_width / 2
        
        # 分离左右栏字符
        left_chars = [c for c in chars if c['x0'] < mid_x]
        right_chars = [c for c in chars if c['x0'] >= mid_x]
        
        # 分别处理左右栏
        left_text = self._chars_to_text(left_chars)
        right_text = self._chars_to_text(right_chars)
        
        # 合并左右栏文本
        if left_text and right_text:
            return left_text + "\n\n" + right_text
        elif left_text:
            return left_text
        else:
            return right_text
    
    def _chars_to_text(self, chars) -> str:
        """将字符列表转换为文本"""
        if not chars:
            return ""
        
        # 按y坐标（从上到下）和x坐标（从左到右）排序
        sorted_chars = sorted(chars, key=lambda c: (-c['y0'], c['x0']))
        
        # 分组成行
        lines = []
        current_line_chars = []
        current_y = None
        
        for char in sorted_chars:
            y = round(char['y0'], 1)
            
            if current_y is None:
                current_y = y
                current_line_chars.append(char)
            elif abs(y - current_y) <= 3:  # 同一行
                current_line_chars.append(char)
            else:  # 新行
                if current_line_chars:
                    lines.append(self._build_spaced_line_text(current_line_chars))
                current_line_chars = [char]
                current_y = y
        
        if current_line_chars:
            lines.append(self._build_spaced_line_text(current_line_chars))
        
        return '\n'.join(lines)
    
    def _build_spaced_line_text(self, chars) -> str:
        """构建带适当空格的行文本"""
        if not chars:
            return ""
        
        # 按x坐标排序确保正确顺序
        chars = sorted(chars, key=lambda c: c['x0'])
        
        result = []
        prev_char = None
        
        for char in chars:
            current_text = char['text']
            
            if prev_char is not None:
                # 计算字符间距
                gap = char['x0'] - prev_char['x1']
                
                # 计算字符平均宽度
                prev_width = prev_char['x1'] - prev_char['x0'] if prev_char['x1'] > prev_char['x0'] else 6
                current_width = char['x1'] - char['x0'] if char['x1'] > char['x0'] else 6
                avg_char_width = (prev_width + current_width) / 2
                
                # 如果间距大于字符宽度的一定比例，添加空格
                if gap > avg_char_width * 0.3:  # 降低阈值，更容易添加空格
                    if gap > avg_char_width * 1.5:
                        result.append('  ')  # 大间距用两个空格
                    else:
                        result.append(' ')   # 正常间距用一个空格
            
            result.append(current_text)
            prev_char = char
        
        return ''.join(result)
    
    def split_by_bookmarks(self, pdf_path: str) -> Tuple[Optional[List[Section]], str]:
        """根据书签分割PDF，返回章节列表和状态信息"""
        logger.info(f"开始基于书签分割PDF: {pdf_path}")
        
        # 1. 提取书签
        bookmarks = self.extract_bookmarks(pdf_path)
        if not bookmarks:
            return None, "没有找到书签结构"
        
        logger.info(f"找到 {len(bookmarks)} 个原始书签")
        
        # 2. 过滤首字母未大写的章节
        bookmarks = self.filter_bookmarks_by_title(bookmarks)
        logger.info(f"首字母过滤后剩余 {len(bookmarks)} 个书签")
        
        # 3. 检查书签数量
        if len(bookmarks) < 3:
            return None, f"书签数量过少: {len(bookmarks)} < 3"
        if len(bookmarks) > self.max_bookmarks:
            return None, f"书签数量过多: {len(bookmarks)} > {self.max_bookmarks}"
        
        # 4. 移除Reference及之后的章节
        bookmarks = self.remove_references_and_after(bookmarks)
        logger.info(f"移除Reference后剩余 {len(bookmarks)} 个书签")
        
        # 5. 按页码排序书签
        bookmarks.sort(key=lambda x: x.page_num)
        
        # 6. 提取每个章节的内容
        sections = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for i, bookmark in enumerate(bookmarks):
                    # 确定章节的起始和结束页面
                    start_page = bookmark.page_num
                    
                    # 找到下一个同级或更高级书签作为结束点
                    end_page = total_pages - 1  # 默认到文档末尾
                    next_bookmark = None
                    for j in range(i + 1, len(bookmarks)):
                        candidate = bookmarks[j]
                        if candidate.level <= bookmark.level:
                            next_bookmark = candidate
                            # 如果下一个书签在同一页，仍使用当前页
                            if candidate.page_num == bookmark.page_num:
                                end_page = bookmark.page_num
                            else:
                                end_page = candidate.page_num - 1
                            break
                    
                    # 提取章节内容
                    try:
                        content = self._extract_section_content(
                            pdf, start_page, end_page, bookmark, next_bookmark
                        )
                        
                        if content.strip():  # 只添加非空章节
                            section = Section(
                                title=self._clean_title(bookmark.title),
                                content=content,
                                page_start=start_page + 1,  # 转为1-based
                                page_end=end_page + 1,
                                section_id=f"section_{i+1}",
                                level=bookmark.level,
                                bookmark_title=bookmark.title
                            )
                            sections.append(section)
                            logger.info(f"成功提取章节 '{bookmark.title}': {len(content)} 字符")
                        else:
                            logger.warning(f"跳过空章节 '{bookmark.title}'")
                    except Exception as e:
                        logger.warning(f"跳过章节 '{bookmark.title}': {e}")
                        continue
        
        except Exception as e:
            logger.error(f"分割PDF时出错: {e}")
            return None, f"处理错误: {e}"
        
        logger.info(f"成功分割为 {len(sections)} 个章节")
        return sections, "success"
    
    def _extract_section_content(self, pdf, start_page: int, end_page: int, bookmark: BookmarkInfo, next_bookmark: BookmarkInfo = None) -> str:
        """提取章节内容"""
        content_parts = []
        
        for page_num in range(start_page, min(end_page + 1, len(pdf.pages))):
            page = pdf.pages[page_num]
            page_text = self.extract_page_text_smart(page)
            
            if page_text:
                # 如果是第一页，尝试去除标题部分
                if page_num == start_page:
                    page_text = self._remove_title_from_content(page_text, bookmark.title)
                
                # 如果下一个书签在同一页，需要截取到下一个标题之前
                if next_bookmark and next_bookmark.page_num == page_num:
                    page_text = self._extract_content_until_next_title(page_text, next_bookmark.title)
                
                # 无论如何，都要检查并移除References部分
                page_text = self._remove_references_content(page_text)
                
                content_parts.append(page_text)
        
        return '\n\n'.join(content_parts)
    
    def _extract_content_until_next_title(self, text: str, next_title: str) -> str:
        """提取文本直到下一个标题出现"""
        # 清理标题（去除数字前缀等）
        clean_next_title = self._clean_title(next_title)
        
        lines = text.split('\n')
        result_lines = []
        
        # 检查references关键词
        reference_keywords = ['reference', 'references', 'bibliography']
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # 检查是否包含下一个标题
            if clean_next_title.lower() in line.lower():
                # 找到下一个标题，停止添加内容
                break
            
            # 检查是否包含reference相关关键词
            is_reference_line = False
            for keyword in reference_keywords:
                if keyword in line_lower and len(line.strip()) < 50:  # 简短的行更可能是标题
                    is_reference_line = True
                    break
            
            if is_reference_line:
                # 找到References标题，停止添加内容
                break
                
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_references_content(self, text: str) -> str:
        """从文本中移除References部分及之后的内容"""
        reference_keywords = ['references', 'reference', 'bibliography']
        
        lines = text.split('\n')
        result_lines = []
        reference_pattern_detected = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # 检查是否是References标题行
            is_reference_title = False
            for keyword in reference_keywords:
                # 精确匹配或者以关键词开头的短行
                if (line_lower == keyword or 
                    (line_lower.startswith(keyword) and len(line_stripped) < 30) or
                    (keyword in line_lower and len(line_stripped) < 20)):
                    is_reference_title = True
                    break
            
            # 额外检查：如果行只包含"References"和一些数字/符号
            if not is_reference_title and len(line_stripped) < 30:
                for keyword in reference_keywords:
                    clean_line = ''.join(c for c in line_lower if c.isalpha())
                    if clean_line == keyword:
                        is_reference_title = True
                        break
            
            # 检查是否是引用列表开始（如 [1], [32] 等模式）
            is_reference_list_start = False
            if not is_reference_title and line_stripped:
                # 检查是否以 [数字] 开头
                import re
                if re.match(r'^\[\d+\]', line_stripped):
                    # 检查前面几行和后面几行是否也有类似模式
                    citation_count = 0
                    # 检查接下来的几行
                    for j in range(i, min(i + 5, len(lines))):
                        if re.match(r'^\[\d+\]', lines[j].strip()):
                            citation_count += 1
                    
                    # 如果连续多行都是引用格式，认为这是引用列表开始
                    if citation_count >= 2:
                        is_reference_list_start = True
            
            if is_reference_title or is_reference_list_start:
                # 找到References标题或引用列表开始，停止添加内容
                break
                
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _remove_title_from_content(self, text: str, title: str) -> str:
        """从内容中移除标题部分"""
        lines = text.split('\n')
        
        # 查找包含标题的行
        title_clean = self._clean_title(title).lower()
        
        for i, line in enumerate(lines):
            line_clean = self._clean_title(line).lower()
            
            # 如果找到标题行，从下一行开始返回内容
            if title_clean in line_clean or line_clean in title_clean:
                if len(title_clean) > 5 and len(line_clean) > 5:  # 避免误匹配短文本
                    return '\n'.join(lines[i+1:])
        
        return text  # 如果没找到标题，返回原文本
    
    def _clean_title(self, title: str) -> str:
        """清理标题文本"""
        # 移除编号前缀
        title = re.sub(r'^[\d\.]+\s*', '', title)
        # 移除多余空白
        title = ' '.join(title.split())
        return title.strip()
    
    def _fallback_to_full_text(self, pdf_path: str) -> List[Section]:
        """回退方案：提取整个文档"""
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
            logger.error(f"回退方案也失败了: {e}")
            return []
    
    def split_sections_into_chunks(self, sections: List[Section]) -> List[Dict[str, Any]]:
        """将章节进一步分割为小块"""
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
        """清理标题用于ID，转换为小写并移除特殊字符"""
        # 移除编号前缀
        title = re.sub(r'^[\d\.]+\s*', '', title)
        # 移除多余空白
        title = ' '.join(title.split())
        # 转换为小写
        title = title.lower()
        # 移除特殊字符，只保留字母和数字
        title = re.sub(r'[^a-z0-9]', '', title)
        return title.strip()
    
    def _chunk_text_by_section(self, text: str, section_title: str) -> List[Dict[str, Any]]:
        """按章节名+数字的方式分割文本"""
        # 清理章节标题，用于ID
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
        """将文本分割为小块"""
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
        """转换为目标JSON格式，使用新的分块逻辑"""
        result = []
        
        for section in sections:
            # 使用新的分块方法
            chunks = self._chunk_text_by_section(section.content, section.title)
            
            # 为每个块创建单独的条目
            for chunk in chunks:
                section_data = {
                    "id": chunk["chunk_id"],  # 使用新的ID格式（章节名+数字）
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
        """处理单个PDF文件"""
        logger.info(f"开始处理PDF: {pdf_path}")
        
        try:
            # 分割章节
            sections, status = self.split_by_bookmarks(pdf_path)
            
            if sections is None:
                return {"success": False, "error": status}
            
            if not sections:
                return {"success": False, "error": "没有有效章节"}
            
            # 转换为目标格式
            result_data = self.convert_to_target_format(sections)
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"处理完成，结果已保存到: {output_path}")
            logger.info(f"统计: {len(sections)}个章节, {len(result_data)}个块")
            
            return {"success": True, "output_file": output_path, "sections": len(sections), "chunks": len(result_data)}
            
        except Exception as e:
            logger.error(f"处理PDF时出错: {e}")
            return {"success": False, "error": str(e)}
    
    def process_pdf_folder(self, input_folder: str, output_folder: str) -> ProcessingStats:
        """批量处理PDF文件夹"""
        logger.info(f"开始批量处理PDF文件夹: {input_folder}")
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 统计信息
        stats = ProcessingStats()
        
        # 获取所有PDF文件
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        stats.total_files = len(pdf_files)
        
        logger.info(f"找到 {stats.total_files} 个PDF文件")
        
        for i, pdf_file in enumerate(pdf_files):
            logger.info(f"\n处理进度: {i+1}/{stats.total_files} - {pdf_file.name}")
            
            # 生成输出文件名：使用原文件名（去掉.pdf扩展名）+ .json
            output_file = Path(output_folder) / f"{pdf_file.stem}.json"
            
            # 处理单个PDF
            result = self.process_single_pdf(str(pdf_file), str(output_file))
            
            if result["success"]:
                stats.processed_successfully += 1
                logger.info(f"✅ 成功处理: {pdf_file.name}")
            else:
                error = result["error"]
                if "没有找到书签结构" in error:
                    stats.skipped_no_bookmarks += 1
                    stats.skipped_reasons.append(f"{pdf_file.name}: 无书签结构")
                elif "书签数量过多" in error:
                    stats.skipped_too_many_bookmarks += 1
                    stats.skipped_reasons.append(f"{pdf_file.name}: {error}")
                else:
                    stats.failed += 1
                    stats.skipped_reasons.append(f"{pdf_file.name}: {error}")
                
                logger.warning(f"❌ 跳过文件: {pdf_file.name} - {error}")
        
        # 打印统计信息
        self._print_processing_stats(stats)
        
        return stats
    
    def _print_processing_stats(self, stats: ProcessingStats):
        """打印处理统计信息"""
        logger.info(f"\n{'='*50}")
        logger.info(f"批量处理完成统计:")
        logger.info(f"总文件数: {stats.total_files}")
        logger.info(f"成功处理: {stats.processed_successfully}")
        logger.info(f"跳过(无书签): {stats.skipped_no_bookmarks}")
        logger.info(f"跳过(书签过多): {stats.skipped_too_many_bookmarks}")
        logger.info(f"处理失败: {stats.failed}")
        
        if stats.skipped_reasons:
            logger.info(f"\n详细跳过原因:")
            for reason in stats.skipped_reasons:
                logger.info(f"  - {reason}")
        
        logger.info(f"{'='*50}")
    
    def process_pdf(self, pdf_path: str, output_path: str = None) -> Dict[str, Any]:
        """完整处理PDF文件"""
        logger.info(f"开始处理PDF: {pdf_path}")
        
        if not output_path:
            pdf_name = Path(pdf_path).stem
            output_path = f"{pdf_name}_bookmark_sections.json"
        
        try:
            # 分割章节
            sections, status = self.split_by_bookmarks(pdf_path)
            
            if sections is None or not sections:
                logger.error(f"未能分割出任何章节: {status}")
                return {"success": False, "error": status or "No sections found"}
            
            # 分块
            section_chunks = self.split_sections_into_chunks(sections)
            
            # 统计信息
            total_chunks = sum(len(section["chunks"]) for section in section_chunks)
            total_words = sum(chunk["word_count"] for section in section_chunks for chunk in section["chunks"])
            
            # 构建结果
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
            
            # 保存结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"处理完成，结果已保存到: {output_path}")
            logger.info(f"统计: {len(section_chunks)}个章节, {total_chunks}个块, {total_words}个单词")
            
            return {"success": True, "output_file": output_path, "result": result}
            
        except Exception as e:
            logger.error(f"处理PDF时出错: {e}")
            return {"success": False, "error": str(e)}

def main():
    """主函数 - 默认批处理模式"""
    import argparse
    
    parser = argparse.ArgumentParser(description="基于PDF书签的智能章节分割器")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_data_papers",
        help="输入PDF文件夹路径 (default: data/raw_data_papers)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/papers",
        help="输出JSON文件夹路径 (default: data/papers)"
    )
    parser.add_argument(
        "--max-bookmarks",
        type=int,
        default=20,
        help="最大书签数量 (default: 20)"
    )
    
    args = parser.parse_args()
    
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).parent.parent.parent
    input_folder = script_dir / args.input
    output_folder = script_dir / args.output
    max_bookmarks = args.max_bookmarks
    
    print(f"🚀 PDF批处理分割器")
    print(f"📁 输入文件夹: {input_folder}")
    print(f"📁 输出文件夹: {output_folder}")
    print(f"📊 最大书签数: {max_bookmarks}")
    print(f"📖 最小书签数: 3")
    print("=" * 50)
    
    # 检查输入路径
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ 错误：输入路径不存在: {input_path}")
        print(f"   请确保路径正确，或使用 --input 参数指定")
        return
    
    if not input_path.is_dir():
        print(f"❌ 错误：输入路径不是文件夹: {input_path}")
        return
    
    # 创建输出目录
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"✅ 输出目录已创建/确认: {output_path}")
    
    # 创建分割器并开始批处理
    splitter = BookmarkBasedSplitter(max_bookmarks=max_bookmarks)
    
    stats = splitter.process_pdf_folder(str(input_path), str(output_path))
    
    print(f"\n📊 最终统计:")
    print(f"✅ 成功处理: {stats.processed_successfully}")
    print(f"⏭️  跳过文件: {stats.skipped_no_bookmarks + stats.skipped_too_many_bookmarks + stats.failed}")
    print(f"📄 总文件数: {stats.total_files}")
    print("=" * 50)

if __name__ == "__main__":
    main()
