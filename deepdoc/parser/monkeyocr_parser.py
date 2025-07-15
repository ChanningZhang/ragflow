#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import io
import json
import logging
import os
import re
import zipfile
import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from PIL import Image

# MonkeyOCR configuration from environment variables
MONKEYOCR_URL = os.environ.get("MONKEYOCR_URL", "http://localhost:6006")
MONKEYOCR_TIMEOUT = int(os.environ.get("MONKEYOCR_TIMEOUT", "300"))


class MonkeyOCRClient:
    """MonkeyOCR HTTP客户端"""
    
    def __init__(self, base_url: str = None, timeout: int = None):
        if base_url is None:
            base_url = MONKEYOCR_URL
        if timeout is None:
            timeout = MONKEYOCR_TIMEOUT
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    def parse_document(self, file_data: bytes, filename: str) -> Dict:
        """
        调用MonkeyOCR解析文档
        
        Args:
            file_data: 文件二进制数据
            filename: 文件名
        
        Returns:
            解析响应字典
        """
        files = {'file': (filename, file_data, 'application/octet-stream')}
        
        try:
            response = self.session.post(f"{self.base_url}/parse", files=files, timeout=self.timeout)
            
            if response.status_code != 200:
                raise Exception(f"MonkeyOCR request failed with status {response.status_code}: {response.text}")
            
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"Network error calling MonkeyOCR: {str(e)}")
    
    def download_result(self, download_url: str) -> bytes:
        """
        下载解析结果ZIP文件
        
        Args:
            download_url: 下载URL
        
        Returns:
            ZIP文件二进制数据
        """
        if download_url.startswith('/'):
            download_url = urljoin(self.base_url + '/', download_url.lstrip('/'))
        
        try:
            response = self.session.get(download_url, timeout=self.timeout)
            
            if response.status_code != 200:
                raise Exception(f"Download failed with status {response.status_code}")
            
            return response.content
        except requests.RequestException as e:
            raise Exception(f"Network error downloading result: {str(e)}")
    
    def close(self):
        """关闭会话"""
        self.session.close()


class MonkeyOCRResultParser:
    """MonkeyOCR结果解析器"""
    
    def __init__(self):
        self.image_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')
        self.markdown_extensions = {'.md', '.markdown'}
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.middle_json = None  # 存储middle.json数据
        self.content_list = None  # 存储content_list.json数据
    
    def extract_zip_content(self, zip_data: bytes) -> Dict[str, bytes]:
        """解压ZIP文件并提取内容"""
        content = {}
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if not file_info.is_dir():
                        try:
                            file_content = zip_file.read(file_info.filename)
                            content[file_info.filename] = file_content
                        except Exception as e:
                            logging.warning(f"Failed to extract {file_info.filename}: {e}")
        except zipfile.BadZipFile as e:
            raise Exception(f"Invalid ZIP file: {e}")
        
        return content
    
    # def get_position_data(self) -> Dict:
    #     """获取位置数据，用于图片合并"""
    #     return {
    #         'middle_json': self.middle_json,
    #         'content_list': self.content_list
    #     }
    
    def load_content_list(self, content_list_data: List[Dict]):
        """加载content_list.json数据"""
        self.content_list = content_list_data
        logging.info(f"Loaded content_list with {len(content_list_data)} entries")
    
    def get_content_list(self) -> List[Dict]:
        """获取content_list数据"""
        return self.content_list if self.content_list else []
    
    def parse_markdown_with_images(self, markdown_content: str, image_files: Dict[str, bytes], middle_json_data: Dict = None) -> List[Tuple[str, List[Image.Image]]]:
        """解析Markdown内容，提取文本和关联的图片列表（不合并）"""
        sections = []
        
        # 记录可用的图片文件（只在开始时打印一次）
        if image_files:
            available_images = list(image_files.keys())
            logging.info(f"Available images for parsing: {available_images}")
        
        # 分离表格和普通文本
        text_content, tables = self._extract_tables_from_markdown(markdown_content)
        
        # 处理普通文本内容
        paragraphs = self._split_markdown_paragraphs(text_content)
        
        for i, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            logging.debug(f"Processing paragraph {i}: {paragraph[:100]}...")
            
            # 查找段落中的图片引用
            images, image_paths = self._extract_images_from_paragraph(paragraph, image_files, middle_json_data)
            
            # 如果发现图片，记录段落索引
            if images:
                logging.debug(f"Paragraph {i} contains {len(images)} images")
                for j, img_path in enumerate(image_paths):
                    page_info = self._get_image_page_info(img_path, middle_json_data)
                    logging.debug(f"  Image {j}: {img_path} (from {page_info})")
            
            # 清理Markdown语法，保留纯文本
            clean_text = self._clean_markdown_syntax(paragraph)
            
            if clean_text.strip():
                # 不合并图片，直接返回图片列表，让chunk级别处理
                sections.append((clean_text.strip(), images))
                logging.debug(f"Added section {len(sections)-1}: text_len={len(clean_text.strip())}, images={len(images)}")
            else:
                # 如果清理后文本为空但有图片，仍然添加一个包含图片的片段
                if images:
                    sections.append(("{图片}", images))
                    logging.debug(f"Added image-only section {len(sections)-1}: {len(images)} images")
        
        # 将表格添加到sections中（保持与RAGFlow格式一致）
        for table_text in tables:
            if table_text.strip():
                sections.append((table_text.strip(), []))  # 表格没有图片
        
        return sections
    
    def _split_markdown_paragraphs(self, content: str) -> List[str]:
        """分割Markdown段落"""
        paragraphs = re.split(r'\n\s*\n', content)
        
        refined_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 处理标题（独立成段）
            if re.match(r'^#{1,6}\s+', para):
                refined_paragraphs.append(para)
            # 处理列表项
            elif re.match(r'^[\-\*\+]\s+', para, re.MULTILINE):
                refined_paragraphs.append(para)
            # 处理表格
            elif '|' in para and para.count('\n') > 0:
                refined_paragraphs.append(para)
            else:
                # 普通段落，按句子分割避免过长
                sentences = self._split_long_paragraph(para)
                refined_paragraphs.extend(sentences)
        
        return refined_paragraphs
    
    def _split_long_paragraph(self, paragraph: str, max_length: int = 1000) -> List[str]:
        """分割过长的段落"""
        if len(paragraph) <= max_length:
            return [paragraph]
        
        sentences = re.split(r'[.!?。！？]\s*', paragraph)
        
        result = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence = sentence.strip()
            if not sentence.endswith(('.', '!', '?', '。', '！', '？')):
                sentence += '。'
            
            if len(current_chunk) + len(sentence) <= max_length:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    result.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            result.append(current_chunk.strip())
        
        return result if result else [paragraph]
    
    def _extract_tables_from_markdown(self, markdown_content: str) -> Tuple[str, List[str]]:
        """从Markdown内容中提取表格，返回(文本内容, 表格列表)"""
        import re
        
        # 表格模式：| 列1 | 列2 | 列3 |
        table_pattern = r'(\|[^\n]*\|(?:\n\|[^\n]*\|)+)'
        
        tables = []
        text_content = markdown_content
        
        # 查找所有表格
        for match in re.finditer(table_pattern, markdown_content):
            table_text = match.group(1)
            tables.append(table_text)
            
            # 从原文本中移除表格
            text_content = text_content.replace(table_text, '')
        
        return text_content, tables
    
    def extract_tables_from_markdown(self, markdown_content: str) -> List:
        """
        从Markdown内容中提取表格，返回RAGFlow格式的表格列表
        
        Returns:
            List: RAGFlow格式的表格列表 [(img, rows), poss]
        """
        text_content, table_texts = self._extract_tables_from_markdown(markdown_content)
        
        tables = []
        for table_text in table_texts:
            if table_text.strip():
                # 将Markdown表格转换为RAGFlow格式
                # RAGFlow期望的格式: ((img, rows), poss)
                # 其中 img 是表格图片（可选），rows 是表格行数据，poss 是位置信息
                
                # 解析Markdown表格为行数据
                rows = self._parse_markdown_table(table_text)
                
                # 创建RAGFlow格式的表格条目
                table_entry = ((None, rows), [])  # (img, rows), poss
                tables.append(table_entry)
        
        return tables
    
    def _parse_markdown_table(self, table_text: str) -> List[str]:
        """
        解析Markdown表格为行数据
        
        Args:
            table_text: Markdown格式的表格文本
            
        Returns:
            List[str]: 表格行数据列表
        """
        lines = table_text.strip().split('\n')
        rows = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('|'):
                # 移除开头的 | 和结尾的 |
                line = line.strip('|')
                if line:
                    # 分割单元格并清理
                    cells = [cell.strip() for cell in line.split('|')]
                    # 跳过分隔行（只包含 - 和 | 的行）
                    if not all(cell.replace('-', '').replace(':', '').strip() == '' for cell in cells):
                        row_text = ' | '.join(cells)
                        if row_text.strip():
                            rows.append(row_text)
        
        return rows
    
    def _get_image_page_info(self, image_path: str, middle_json_data: Dict = None) -> str:
        """根据图片路径获取页面信息"""
        if not middle_json_data:
            return "unknown"
        
        # 从图片路径中提取文件名
        image_filename = os.path.basename(image_path)
        
        # 在 middle.json 中查找图片的页面信息
        for page_info in middle_json_data.get("pdf_info", []):
            page_idx = page_info.get("page_idx", 0)
            
            # 在 preproc_blocks 中查找
            for block in page_info.get("preproc_blocks", []):
                if block.get("type") == "image":
                    for block_item in block.get("blocks", []):
                        for line in block_item.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("type") == "image":
                                    span_image_path = span.get("image_path", "")
                                    if span_image_path and image_filename in span_image_path:
                                        return f"page_{page_idx}"
            
            # 在 images 数组中查找
            for img_info in page_info.get("images", []):
                if img_info.get("type") == "image":
                    for block_item in img_info.get("blocks", []):
                        for line in block_item.get("lines", []):
                            for span in line.get("spans", []):
                                if span.get("type") == "image":
                                    span_image_path = span.get("image_path", "")
                                    if span_image_path and image_filename in span_image_path:
                                        return f"page_{page_idx}"
        
        return "unknown"

    def _extract_images_from_paragraph(self, paragraph: str, image_files: Dict[str, bytes], middle_json_data: Dict = None) -> Tuple[List[Image.Image], List[str]]:
        """从段落中提取图片"""
        images = []
        image_paths = []
        
        matches = self.image_pattern.findall(paragraph)
        
        if matches:
            logging.debug(f"Found {len(matches)} image references in paragraph: {[path for _, path in matches]}")
            logging.debug(f"Paragraph text preview: {paragraph[:200]}...")
        
        for alt_text, image_path in matches:
            logging.debug(f"Processing image reference: {image_path}")
            
            # 获取图片文件名（不包含路径）
            image_filename = os.path.basename(image_path)
            logging.debug(f"Looking for image filename: {image_filename}")
            
            # 尝试多种匹配策略
            matched_image = None
            
            # 策略1：直接路径匹配
            if image_path in image_files:
                matched_image = image_path
                logging.debug(f"Direct path match: {image_path}")
            
            # 策略2：文件名匹配（忽略前缀）
            elif not matched_image:
                for file_path in image_files.keys():
                    if file_path.endswith(image_filename):
                        matched_image = file_path
                        logging.debug(f"Filename match: {file_path} -> {image_filename}")
                        break
            
            # 策略3：文件名匹配（包含前缀）
            elif not matched_image:
                for file_path in image_files.keys():
                    if os.path.basename(file_path) == image_filename:
                        matched_image = file_path
                        logging.debug(f"Basename match: {file_path} -> {image_filename}")
                        break
            
            # 策略4：模糊匹配（包含文件名）
            elif not matched_image:
                for file_path in image_files.keys():
                    if image_filename in file_path:
                        matched_image = file_path
                        logging.debug(f"Fuzzy match: {file_path} contains {image_filename}")
                        break
            
            if matched_image:
                try:
                    img = Image.open(io.BytesIO(image_files[matched_image])).convert('RGB')
                    images.append(img)
                    image_paths.append(matched_image) # 记录匹配到的图片路径
                    
                    # 获取图片的页面信息
                    page_info = self._get_image_page_info(matched_image, middle_json_data)
                    
                    logging.info(f"Successfully loaded image: {matched_image} (from {page_info})")
                    
                    # 如果图片来自第二页但出现在早期段落中，发出警告
                    if page_info == "page_1":  # 第二页（索引从0开始）
                        logging.warning(f"🔍 PAGE 2 IMAGE found in paragraph: {paragraph[:200]}...")
                        logging.warning(f"🔍 Image path: {matched_image}")
                    
                except Exception as e:
                    logging.warning(f"Failed to load image {matched_image}: {e}")
            else:
                logging.warning(f"No matching image found for: {image_path}")
        
        return images, image_paths
    
    def _clean_markdown_syntax(self, text: str) -> str:
        """清理Markdown语法标记"""
        # 移除图片引用
        text = self.image_pattern.sub('', text)
        
        # 移除标题标记
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # 移除粗体和斜体标记
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # 移除链接标记（保留链接文本）
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # 移除代码标记
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
        
        # 移除列表标记
        text = re.sub(r'^[\-\*\+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # 移除表格标记（但保留表格内容）
        text = re.sub(r'^\|.*\|$', '', text, flags=re.MULTILINE)
        
        # 规范化空白
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _combine_images(self, images: List[Image.Image], image_paths: List[str] = None, middle_json_data: Dict = None) -> Optional[Image.Image]:
        """
        根据位置信息合并多个图片
        
        Args:
            images: 图片列表
            image_paths: 图片路径列表，用于在middle.json中查找位置信息
            middle_json_data: middle.json的解析数据，包含位置信息
            
        Returns:
            Optional[Image.Image]: 合并后的图片
        """
        if not images:
            return None
        
        if len(images) == 1:
            return images[0]
        
        # 优先使用位置信息进行拼接
        if middle_json_data and image_paths:
            try:
                # 从middle.json中提取位置信息
                all_positions = self._extract_image_positions_from_middle(middle_json_data, image_paths)
                
                if all_positions:
                    # 根据图片路径匹配位置信息，确保顺序正确
                    matched_positions = []
                    matched_images = []
                    
                    for i, image_path in enumerate(image_paths):
                        if i < len(images):
                            # 查找对应的位置信息
                            matched_pos = None
                            for pos in all_positions:
                                if pos["image_path"] == image_path:
                                    matched_pos = pos
                                    break
                            
                            if matched_pos:
                                matched_positions.append(matched_pos)
                                matched_images.append(images[i])
                                logging.info(f"Matched image {i+1}: {image_path} -> bbox {matched_pos['bbox']}")
                            else:
                                logging.warning(f"No position found for image: {image_path}")
                    
                    if matched_positions and len(matched_positions) == len(matched_images):
                        # 根据位置信息合并图片
                        logging.info(f"Using position-based combination for {len(matched_images)} images")
                        return self._combine_images_by_position(matched_images, matched_positions)
                    else:
                        logging.warning("Position matching failed, falling back to vertical combination")
                else:
                    logging.info("No position information found in middle.json")
            except Exception as e:
                logging.warning(f"Failed to combine images by position: {e}, falling back to vertical combination")
        
        # 回退到垂直拼接
        logging.info("Using vertical combination for images")
        return self._combine_images_vertical(images)
    
    def _combine_images_vertical(self, images: List[Image.Image]) -> Image.Image:
        """垂直拼接图片（原来的方法）"""
        if len(images) == 1:
            return images[0]
        
        # 垂直拼接图片
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        combined = Image.new('RGB', (max_width, total_height), 'white')
        
        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height
        
        return combined
    
    def _extract_image_positions_from_middle(self, middle_json_data: Dict, image_paths: List[str]) -> List[Dict]:
        """
        从middle.json中提取图片位置信息 - 保持原始页面相对坐标
        
        Args:
            middle_json_data: middle.json的解析数据
            image_paths: 图片路径列表
            
        Returns:
            List[Dict]: 位置信息列表，每个元素包含 {image_path, bbox, page_idx, page_size}
        """
        positions = []
        
        # 遍历所有页面
        for page_info in middle_json_data.get("pdf_info", []):
            page_idx = page_info.get("page_idx", 0)
            page_size = page_info.get("page_size", [595.3, 841.9])  # 默认A4尺寸
            page_width = page_size[0] if len(page_size) > 0 else 595.3
            page_height = page_size[1] if len(page_size) > 1 else 841.9
            
            logging.info(f"Processing page {page_idx}, page_size={page_size}")
            
            # 从preproc_blocks中提取图片位置
            for block in page_info.get("preproc_blocks", []):
                if block.get("type") == "image":
                    bbox = block.get("bbox", [])
                    if len(bbox) == 4:
                        # 查找对应的图片路径
                        for block_item in block.get("blocks", []):
                            for line in block_item.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("type") == "image":
                                        image_path = span.get("image_path", "")
                                        if image_path:
                                            # 改进的路径匹配逻辑
                                            matched_path = self._match_image_path(image_path, image_paths)
                                            if matched_path:
                                                # 检查是否已经添加过（避免重复）
                                                if not any(pos["image_path"] == matched_path and pos["page_idx"] == page_idx for pos in positions):
                                                    # 保持原始页面相对坐标
                                                    positions.append({
                                                        "image_path": matched_path,
                                                        "bbox": bbox,  # 保持原始页面相对坐标
                                                        "page_idx": page_idx,
                                                        "page_size": [page_width, page_height],
                                                        "left": bbox[0],
                                                        "top": bbox[1],
                                                        "right": bbox[2],
                                                        "bottom": bbox[3]
                                                    })
                                                    
                                                    logging.debug(f"Added position for {matched_path}: page={page_idx}, bbox={bbox}, page_size={[page_width, page_height]}")
                                                else:
                                                    logging.debug(f"Skipped duplicate position for {matched_path} on page {page_idx}")
            
            # 从images数组中提取图片位置（备用）
            for img_info in page_info.get("images", []):
                if img_info.get("type") == "image":
                    bbox = img_info.get("bbox", [])
                    if len(bbox) == 4:
                        for block_item in img_info.get("blocks", []):
                            for line in block_item.get("lines", []):
                                for span in line.get("spans", []):
                                    if span.get("type") == "image":
                                        image_path = span.get("image_path", "")
                                        if image_path:
                                            matched_path = self._match_image_path(image_path, image_paths)
                                            if matched_path:
                                                # 检查是否已经添加过
                                                if not any(pos["image_path"] == matched_path for pos in positions):
                                                    # 保持原始页面相对坐标
                                                    positions.append({
                                                        "image_path": matched_path,
                                                        "bbox": bbox,  # 保持原始页面相对坐标
                                                        "page_idx": page_idx,
                                                        "page_size": [page_width, page_height],
                                                        "left": bbox[0],
                                                        "top": bbox[1],
                                                        "right": bbox[2],
                                                        "bottom": bbox[3]
                                                    })
                                                    
                                                    logging.debug(f"Added position for {matched_path} (from images): page={page_idx}, bbox={bbox}, page_size={[page_width, page_height]}")
        
        return positions
    
    def _match_image_path(self, middle_path: str, zip_paths: List[str]) -> Optional[str]:
        """
        匹配middle.json中的图片路径和ZIP文件中的图片路径
        
        Args:
            middle_path: middle.json中的图片路径
            zip_paths: ZIP文件中的图片路径列表
            
        Returns:
            Optional[str]: 匹配到的路径，如果没有匹配到则返回None
        """
        # 提取文件名（不包含路径）
        middle_filename = os.path.basename(middle_path)
        
        # 策略1：直接文件名匹配
        for zip_path in zip_paths:
            zip_filename = os.path.basename(zip_path)
            if zip_filename == middle_filename:
                return zip_path
        
        # 策略2：包含文件名匹配
        for zip_path in zip_paths:
            if middle_filename in zip_path:
                return zip_path
        
        # 策略3：移除扩展名后匹配
        middle_name_without_ext = os.path.splitext(middle_filename)[0]
        for zip_path in zip_paths:
            zip_name_without_ext = os.path.splitext(os.path.basename(zip_path))[0]
            if zip_name_without_ext == middle_name_without_ext:
                return zip_path
        
        # 策略4：哈希值匹配（如果文件名是哈希值）
        if len(middle_filename.split('.')[0]) >= 32:  # 可能是哈希值
            for zip_path in zip_paths:
                zip_filename = os.path.basename(zip_path)
                if len(zip_filename.split('.')[0]) >= 32:
                    # 比较哈希值部分
                    middle_hash = middle_filename.split('.')[0]
                    zip_hash = zip_filename.split('.')[0]
                    if middle_hash == zip_hash:
                        return zip_path
        
        logging.warning(f"No matching path found for {middle_path} in {zip_paths}")
        return None
    
    def _combine_images_by_position(self, images: List[Image.Image], positions: List[Dict]) -> Image.Image:
        """
        根据位置信息合并图片 - 不缩放，保持原始比例和页面相对位置
        
        Args:
            images: 图片列表
            positions: 位置信息列表
            
        Returns:
            Image.Image: 合并后的图片
        """
        logging.info(f"_combine_images_by_position called with {len(images)} images and {len(positions)} positions")
        
        if not positions:
            logging.warning("No positions provided, falling back to vertical combination")
            return self._combine_images_vertical(images)
        
        # 确保图片数量和位置信息数量匹配
        if len(images) != len(positions):
            logging.warning(f"Images count ({len(images)}) doesn't match positions count ({len(positions)}), falling back to vertical combination")
            return self._combine_images_vertical(images)
        
        # 详细记录输入信息
        # for i, (img, pos) in enumerate(zip(images, positions)):
        #     logging.info(f"Input {i}: image_size={img.size}, position={pos}")
        
        try:
            # 按位置排序（先按页面，再按top，最后按left）
            sorted_items = sorted(zip(images, positions), key=lambda x: (x[1]["page_idx"], x[1]["top"], x[1]["left"]))
            images, positions = zip(*sorted_items)
            images = list(images)
            positions = list(positions)
            
            logging.info("After sorting by position:")
            # for i, (img, pos) in enumerate(zip(images, positions)):
            #     logging.info(f"Sorted {i}: image_size={img.size}, position={pos}")
            
            # 检查是否有跨页面的图片
            pages = set(pos["page_idx"] for pos in positions)
            if len(pages) > 1:
                logging.info(f"Cross-page images detected: pages {sorted(pages)}")
                return self._combine_images_cross_page(images, positions)
            
            # 单页面图片：使用完整的页面尺寸作为画布
            page_width = positions[0]["page_size"][0] if positions[0].get("page_size") else 595.3
            page_height = positions[0]["page_size"][1] if positions[0].get("page_size") else 841.9
            
            # 计算图片的实际位置范围（使用bbox信息）
            min_left = min(pos["left"] for pos in positions)
            min_top = min(pos["top"] for pos in positions)
            max_right = max(pos["right"] for pos in positions)
            max_bottom = max(pos["bottom"] for pos in positions)
            
            # 使用统一的画布尺寸计算逻辑（单张和多张图片都一样）
            canvas_width = max(page_width, max_right)
            canvas_height = max_bottom - min_top
            
            logging.info(f"Page size: {page_width} x {page_height}")
            logging.info(f"Image bounds (using bbox): min_left={min_left}, min_top={min_top}, max_right={max_right}, max_bottom={max_bottom}")
            logging.info(f"Canvas size: {canvas_width} x {canvas_height}")
            
            # 确保画布尺寸合理
            if canvas_width <= 0 or canvas_height <= 0:
                logging.warning(f"Invalid canvas size: {canvas_width}x{canvas_height}, falling back to vertical combination")
                return self._combine_images_vertical(images)
            
            logging.info(f"Creating canvas with size: {canvas_width} x {canvas_height}")
            
            # 创建画布
            combined = Image.new('RGB', (int(canvas_width), int(canvas_height)), 'white')
            
            # 放置图片（保持在页面中的原始位置）
            for i, (img, pos) in enumerate(zip(images, positions)):
                # 使用统一的位置计算逻辑（单张和多张图片都一样）
                x = int(pos["left"])  # 直接使用页面坐标
                y = int(pos["top"] - min_top)  # Y坐标相对于最小top值
                
                # 计算目标尺寸（使用bbox信息）
                target_width = int(pos["right"] - pos["left"])
                target_height = int(pos["bottom"] - pos["top"])
                
                # 缩放图片到目标尺寸
                if img.size != (target_width, target_height):
                    resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                    logging.info(f"Image {i+1}: Resized from {img.size} to {resized_img.size}")
                else:
                    resized_img = img
                    logging.info(f"Image {i+1}: No resize needed, size={img.size}")
                
                # 边界检查：确保图片不超出画布
                paste_x = max(0, min(x, int(canvas_width) - target_width))
                paste_y = max(0, min(y, int(canvas_height) - target_height))
                
                # 记录关键信息
                # logging.info(f"Image {i+1}: Position check")
                # logging.info(f"  Target position: ({x}, {y}), Target size: {target_width}x{target_height}")
                # logging.info(f"  Canvas size: {canvas_width}x{canvas_height}")
                # logging.info(f"  Final position: ({paste_x}, {paste_y})")
                
                if paste_x != x or paste_y != y:
                    logging.warning(f"  ⚠️  Position adjusted: ({x}, {y}) -> ({paste_x}, {paste_y})")
                
                # 粘贴缩放后的图片
                combined.paste(resized_img, (paste_x, paste_y))
            
            logging.info(f"Successfully combined {len(images)} images using position information")
            return combined
            
        except Exception as e:
            logging.error(f"Error in position-based image combination: {e}")
            return self._combine_images_vertical(images)
    
    def _combine_images_cross_page(self, images: List[Image.Image], positions: List[Dict]) -> Image.Image:
        """
        处理跨页面的图片合并 - 使用相对位置计算，不缩放，保持原始比例和页面相对位置
        
        核心思路：
        1. 按页面分组图片
        2. 计算每个页面内图片的相对位置布局
        3. 将后续页面的图片坐标转换为相对于前一页底部的位置
        4. 在统一画布上按转换后的位置放置图片（不缩放）
        5. 保持图片在各自页面中的相对位置
        
        Args:
            images: 图片列表
            positions: 位置信息列表
            
        Returns:
            Image.Image: 合并后的图片
        """
        logging.info(f"_combine_images_cross_page called with {len(images)} images")
        
        # 按页面分组并排序
        page_groups = {}
        for img, pos in zip(images, positions):
            page_idx = pos["page_idx"]
            if page_idx not in page_groups:
                page_groups[page_idx] = []
            page_groups[page_idx].append((img, pos))
        
        # 按页面顺序排序
        sorted_pages = sorted(page_groups.keys())
        logging.info(f"Page groups: {sorted_pages}")
        
        # 计算跨页面的相对位置布局
        converted_positions = []
        converted_images = []
        accumulated_height = 0  # 累计高度偏移
        max_page_width = 0  # 记录最大页面宽度
        
        for page_idx in sorted_pages:
            page_items = page_groups[page_idx]
            page_imgs = [item[0] for item in page_items]
            page_positions = [item[1] for item in page_items]
            
            logging.info(f"Processing page {page_idx} with {len(page_imgs)} images, accumulated_height={accumulated_height}")
            
            # 获取当前页面的尺寸
            if page_positions:
                page_size = page_positions[0]["page_size"]
                page_width, page_height = page_size[0], page_size[1]
                max_page_width = max(max_page_width, page_width)
                logging.info(f"Page {page_idx} size: {page_width} x {page_height}")
            else:
                continue
            
            # 处理当前页面的每个图片
            for img, pos in zip(page_imgs, page_positions):
                # 计算转换后的位置：保持X坐标（页面相对位置），Y坐标加上累计高度偏移
                converted_pos = {
                    "image_path": pos["image_path"],
                    "page_idx": pos["page_idx"],
                    "page_size": pos["page_size"],
                    "original_bbox": pos["bbox"],
                    # 转换后的坐标：X坐标保持不变（保持页面相对位置），Y坐标加上累计高度偏移
                    "left": pos["left"],  # 保持页面相对位置
                    "top": pos["top"] + accumulated_height,
                    "right": pos["right"],  # 保持页面相对位置
                    "bottom": pos["bottom"] + accumulated_height,
                    "bbox": [
                        pos["left"],  # 保持页面相对位置
                        pos["top"] + accumulated_height,
                        pos["right"],  # 保持页面相对位置
                        pos["bottom"] + accumulated_height
                    ]
                }
                
                converted_positions.append(converted_pos)
                converted_images.append(img)
                
                # logging.info(f"Converted image from page {page_idx}: original_bbox={pos['bbox']}, converted_bbox={converted_pos['bbox']}")
                # logging.info(f"  Page relative position: X={pos['left']}/{page_width} ({pos['left']/page_width*100:.1f}%)")
            
            # 更新累计高度偏移：加上当前页面的高度
            accumulated_height += page_height
            logging.info(f"Updated accumulated_height to {accumulated_height} after page {page_idx}")
        
        # 现在使用转换后的位置信息进行拼接
        if not converted_positions:
            logging.warning("No converted positions available")
            return self._combine_images_vertical(images)
        
        # 计算整体画布大小 - 基于bbox信息
        min_left = min(pos["left"] for pos in converted_positions)
        min_top = min(pos["top"] for pos in converted_positions)
        
        # 使用bbox信息计算右边界和底边界
        max_right = max(pos["right"] for pos in converted_positions)
        max_bottom = max(pos["bottom"] for pos in converted_positions)
        
        # 使用最大页面宽度作为画布宽度，但确保能容纳所有图片
        canvas_width = max(max_page_width, max_right)
        canvas_height = max_bottom - min_top
        
        logging.info(f"Cross-page canvas (using bbox): {canvas_width} x {canvas_height} (max_page_width: {max_page_width})")
        logging.info(f"Cross-page bounds: min_left={min_left}, min_top={min_top}, max_right={max_right}, max_bottom={max_bottom}")
        
        # 确保画布尺寸合理
        if canvas_width <= 0 or canvas_height <= 0:
            logging.warning(f"Invalid canvas size: {canvas_width}x{canvas_height}, falling back to vertical combination")
            return self._combine_images_vertical(images)
        
        # 创建画布
        combined = Image.new('RGB', (int(canvas_width), int(canvas_height)), 'white')
        
        # 按转换后的位置放置图片（保持原始尺寸）
        for i, (img, pos) in enumerate(zip(converted_images, converted_positions)):
            # 计算在画布上的位置 - 保持图片在页面中的原始X位置
            x = int(pos["left"])  # 直接使用转换后的X坐标（保持页面相对位置）
            y = int(pos["top"] - min_top)  # Y坐标相对于最小top值
            
            # 计算目标尺寸（使用bbox信息）
            target_width = int(pos["right"] - pos["left"])
            target_height = int(pos["bottom"] - pos["top"])
            
            # 缩放图片到目标尺寸
            if img.size != (target_width, target_height):
                resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                # logging.info(f"Cross-page image {i+1}: Resized from {img.size} to {resized_img.size}")
            else:
                resized_img = img
                # logging.info(f"Cross-page image {i+1}: No resize needed, size={img.size}")
            
            # 边界检查：确保图片不超出画布
            paste_x = max(0, min(x, int(canvas_width) - target_width))
            paste_y = max(0, min(y, int(canvas_height) - target_height))
            
            # 记录关键信息
            # logging.info(f"Cross-page image {i+1} (page {pos['page_idx']}): Position check")
            # logging.info(f"  Target position: ({x}, {y}), Target size: {target_width}x{target_height}")
            # logging.info(f"  Canvas size: {canvas_width}x{canvas_height}")
            # logging.info(f"  Final position: ({paste_x}, {paste_y})")
            
            if paste_x != x or paste_y != y:
                logging.warning(f"  ⚠️  Position adjusted: ({x}, {y}) -> ({paste_x}, {paste_y})")
            
            # 粘贴缩放后的图片
            combined.paste(resized_img, (paste_x, paste_y))
        
        logging.info(f"Cross-page combination completed: final size={combined.size}")
        return combined
    

class MonkeyOCRParser:
    """MonkeyOCR PDF解析器 - 底层解析器，与DeepDOC parser同级"""
    
    def __init__(self, monkeyocr_url: str = None, timeout: int = None, kb_id: str = None):
        if monkeyocr_url is None:
            monkeyocr_url = MONKEYOCR_URL
        if timeout is None:
            timeout = MONKEYOCR_TIMEOUT
        self.monkeyocr_url = monkeyocr_url
        self.timeout = timeout
        self.kb_id = kb_id
    
    def crop(self, text, ZM=3, need_position=False):
        """
        MonkeyOCR的crop方法实现
        由于MonkeyOCR返回的是已经处理好的文本和图片，这里返回None表示没有对应的图片区域
        
        Args:
            text: 文本内容
            ZM: 缩放因子（未使用）
            need_position: 是否需要位置信息
            
        Returns:
            Tuple[Optional[Image.Image], Optional[List]]: (图片, 位置信息)
        """
        # MonkeyOCR已经将图片和文本分离处理，所以这里返回None
        if need_position:
            return None, None
        return None
    
    def remove_tag(self, txt):
        """
        MonkeyOCR的remove_tag方法实现
        由于MonkeyOCR返回的是清理后的文本，这里直接返回原文本
        
        Args:
            txt: 文本内容
            
        Returns:
            str: 清理后的文本
        """
        # MonkeyOCR已经返回清理后的文本，不需要额外的标签清理
        return txt
    
    def __call__(self, filename, binary=None, from_page=0, to_page=100000, callback=None):
        """
        MonkeyOCR解析器主函数
        
        Args:
            filename: 文件名或文件路径
            binary: 文件二进制数据
            from_page: 起始页（暂未使用）
            to_page: 结束页（暂未使用）
            callback: 回调函数
        
        Returns:
            Tuple[List[Tuple[str, Optional[Image.Image]]], List]: (文本片段列表, 表格列表)
            与其他PDF解析器保持一致的返回格式
        """
        if callback is None:
            callback = lambda prog, msg: None
        
        # 验证输入
        if not binary:
            if not os.path.exists(filename):
                callback(-1, f"文件不存在: {filename}")
                return [], []
            
            with open(filename, 'rb') as f:
                binary = f.read()
        else:
            # 如果提供了binary参数，确保filename是字符串（文件名）
            if isinstance(filename, bytes):
                # 如果filename是二进制数据，说明参数传递错误
                callback(-1, f"参数错误：filename应该是文件名，但收到了二进制数据")
                return [], []
        
        client = None
        try:
            callback(0.1, "连接MonkeyOCR服务...")
            
            client = MonkeyOCRClient(self.monkeyocr_url, self.timeout)
            callback(0.2, "发送文档到MonkeyOCR...")
            
            # 调用MonkeyOCR解析
            response = client.parse_document(binary, os.path.basename(filename))
            
            if not response.get('success'):
                raise Exception(f"MonkeyOCR parsing failed: {response.get('message', 'Unknown error')}")
            
            callback(0.4, "下载解析结果...")
            
            # 下载结果
            download_url = response.get('download_url')
            if not download_url:
                raise Exception("No download URL in MonkeyOCR response")
            
            zip_data = client.download_result(download_url)
            
            # 将ZIP数据落盘以便排查
            import tempfile
            
            # 创建临时目录用于存储调试文件
            debug_dir = os.path.join(os.getcwd(), "monkeyocr_debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # 生成带时间戳的文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"monkeyocr_result_{timestamp}.zip"
            zip_path = os.path.join(debug_dir, zip_filename)
            
            # 保存ZIP文件
            with open(zip_path, 'wb') as f:
                f.write(zip_data)
            
            logging.info(f"MonkeyOCR ZIP文件已保存到: {zip_path}")
            callback(0.6, f"解析返回数据... (ZIP已保存到: {zip_path})")
            
            # 解析ZIP内容
            parser = MonkeyOCRResultParser()
            content = parser.extract_zip_content(zip_data)
            
            callback(0.7, "处理文档内容...")
            
            # 分离Markdown文件和图片文件
            markdown_files = {}
            image_files = {}
            middle_json_data = {}
            content_list_data = None
            minio_image_locations = []  # 存储MinIO图片位置对象数组
            
            # 调试：打印所有解压的文件
            # print(f"🔍 DEBUG: All extracted files: {list(content.keys())}")
            logging.info(f"All extracted files: {list(content.keys())}")
            
            # 导入MinIO工具类
            try:
                from rag.utils.storage_factory import STORAGE_IMPL
                minio_available = True
                logging.info("MinIO storage available")
            except ImportError:
                minio_available = False
                logging.warning("MinIO storage not available, skipping image upload")
            
            # 调试信息
            logging.info(f"MinIO available: {minio_available}")
            logging.info(f"KB ID: {self.kb_id}")
            logging.info(f"Image extensions: {parser.image_extensions}")
            
            for file_path, file_content in content.items():
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == ".json":
                    try:
                        json_data = json.loads(file_content.decode('utf-8'))
                        filename = os.path.basename(file_path)
                        if filename.endswith('_middle.json'):
                            middle_json_data = json_data
                            # print(f"📄 DEBUG: Found middle.json file: {file_path}")
                            logging.info(f"Found middle.json file: {file_path}")
                        elif filename.endswith('_content_list.json'):
                            content_list_data = json_data
                            logging.info(f"Found content_list.json file: {file_path}")
                        else:
                            # 如果不是明确的文件类型，根据内容判断
                            if isinstance(json_data, dict) and 'pdf_info' in json_data:
                                middle_json_data = json_data
                                logging.info(f"Found middle.json file (by content): {file_path}")
                            elif isinstance(json_data, list) and len(json_data) > 0 and 'type' in json_data[0]:
                                content_list_data = json_data
                                logging.info(f"Found content_list.json file (by content): {file_path}")
                            else:
                                logging.warning(f"Unknown JSON file type: {file_path}")
                    except json.JSONDecodeError:
                        logging.warning(f"Failed to decode JSON file {file_path}")
                elif ext in parser.markdown_extensions:
                    try:
                        markdown_files[file_path] = file_content.decode('utf-8')
                        # print(f"📄 DEBUG: Found markdown file: {file_path}")
                        logging.info(f"Found markdown file: {file_path}")
                    except UnicodeDecodeError:
                        logging.warning(f"Failed to decode markdown file {file_path}")
                elif ext in parser.image_extensions:
                    image_files[file_path] = file_content
                    logging.info(f"Found image file: {file_path}")
                    
                    # 上传图片到MinIO
                    if minio_available and self.kb_id:
                        try:
                            import uuid
                            image_filename = f"monkeyocr_{uuid.uuid4().hex}_{os.path.basename(file_path)}"
                            STORAGE_IMPL.put(self.kb_id, image_filename, file_content)
                            minio_path = f"{self.kb_id}/{image_filename}"
                            minio_image_locations.append({
                                "image_name": os.path.basename(file_path),
                                "location": minio_path
                            })
                            logging.info(f"Successfully uploaded image to MinIO: {minio_path}")
                        except Exception as e:
                            logging.error(f"Failed to upload image {file_path} to MinIO: {e}")
                    else:
                        logging.warning(f"Skipping MinIO upload for {file_path}: minio_available={minio_available}, kb_id={self.kb_id}")
            
            # 将数据加载到解析器中
            if middle_json_data:
                parser.middle_json = middle_json_data
            if content_list_data:
                parser.load_content_list(content_list_data)
            
            # print(f"📊 DEBUG: Total markdown files: {len(markdown_files)}")
            logging.info(f"Total markdown files: {len(markdown_files)}")
            # print(f"📊 DEBUG: Total image files: {len(image_files)}")
            logging.info(f"Total image files: {len(image_files)}")
            
            callback(0.8, "生成文档片段...")
            
            # 解析所有Markdown文件
            all_sections = []
            all_tables = []
            
            for md_path, md_content in markdown_files.items():
                # print(f"📝 DEBUG: Processing markdown file: {md_path}")
                # print(f"📝 DEBUG: Markdown content length: {len(md_content)} characters")
                logging.info(f"Processing markdown file: {md_path}")
                logging.info(f"Markdown content length: {len(md_content)} characters")
                
                sections = parser.parse_markdown_with_images(md_content, image_files, middle_json_data)
                # print(f"📝 DEBUG: Generated {len(sections)} sections from {md_path}")
                logging.info(f"Generated {len(sections)} sections from {md_path}")
                
                # 统计包含图片的片段
                sections_with_images = sum(1 for _, img in sections if img is not None)
                # print(f"🖼️ DEBUG: Sections with images: {sections_with_images}")
                logging.info(f"Sections with images: {sections_with_images}")
                
                all_sections.extend(sections)
                
                # 提取表格数据
                tables = parser.extract_tables_from_markdown(md_content)
                all_tables.extend(tables)
            
            # 如果没有Markdown文件，尝试直接处理图片
            if not all_sections and image_files:
                callback(0.85, "处理独立图片...")
                for img_path, img_data in image_files.items():
                    try:
                        img = Image.open(io.BytesIO(img_data)).convert('RGB')
                        all_sections.append((f"Image: {os.path.basename(img_path)}", img))
                    except Exception as e:
                        logging.warning(f"Failed to process image {img_path}: {e}")
            
            callback(0.9, "MonkeyOCR处理完成")
            
            # 将位置信息存储到解析器实例中，供后续使用
            self._position_data = {
                'middle_json': middle_json_data,
                'content_list': content_list_data,
                'image_locations': minio_image_locations,
                'markdown_files': markdown_files
            }
            
            # 调试信息
            logging.info(f"Final minio_image_locations count: {len(minio_image_locations)}")
            logging.info(f"Final minio_image_locations: {minio_image_locations}")
            
            # 返回格式与其他PDF解析器一致
            return all_sections, all_tables  # (sections, tables)
            
        except Exception as e:
            error_msg = f"MonkeyOCR parsing failed: {str(e)}"
            logging.error(error_msg)
            callback(-1, error_msg)
            return [], []
        finally:
            if client:
                client.close()
    
    def get_position_data(self):
        """获取位置信息数据"""
        return getattr(self, '_position_data', None)
    
    def get_content_list(self):
        """获取content_list数据"""
        position_data = self.get_position_data()
        if position_data and 'content_list' in position_data:
            return position_data['content_list']
        return None

    def get_minio_image_locations(self):
        """获取MinIO图片位置对象数组"""
        position_data = self.get_position_data()
        if position_data and 'image_locations' in position_data:
            return position_data['image_locations']
        return []

    def _save_markdown_files(self, doc_id, position_data):
        """
        保存 Markdown 文件到数据库
        
        Args:
            doc_id: 文档ID
            position_data: 位置数据，包含 markdown_files 和 image_locations
        """
        if not doc_id:
            logging.warning("doc_id 为空，跳过 Markdown 文件保存")
            return
        
        markdown_files = position_data.get('markdown_files', {})
        if not markdown_files:
            logging.info("没有找到 Markdown 文件，跳过保存")
            return
        
        logging.info(f"开始保存 {len(markdown_files)} 个 Markdown 文件，文档ID: {doc_id}")
        
        try:
            # 导入 DocumentContentService
            from api.db.services.document_content_service import DocumentContentService
            
            # 获取中间数据
            middle_json_data = position_data.get('middle_json', {})
            content_list_data = position_data.get('content_list', [])
            image_locations = position_data.get('image_locations', [])
            
            # 保存每个 Markdown 文件
            for file_path, content in markdown_files.items():
                try:
                    # 创建文档内容记录
                    DocumentContentService.create_document_content(
                        doc_id=doc_id,
                        markdown=content,
                        monkeyocr_middle_json=middle_json_data,
                        monkeyocr_content_list=content_list_data,
                        monkeyocr_image_locations=image_locations,
                        file_path=file_path,
                        file_name=os.path.basename(file_path)
                    )
                    
                    logging.info(f"已保存 Markdown 文件: {file_path} (大小: {len(content)} 字符)")
                    if image_locations:
                        logging.info(f"关联的图片位置: {image_locations}")
                        
                except Exception as e:
                    logging.error(f"保存 Markdown 文件 {file_path} 失败: {e}")
                    
        except ImportError as e:
            logging.error(f"导入 DocumentContentService 失败: {e}")
            logging.error("请确保 DocumentContentService 已正确配置")
        except Exception as e:
            logging.error(f"保存 Markdown 文件过程中发生错误: {e}")

    def save_image_locations_to_db(self, doc_id):
        """
        将图片位置保存到数据库
        Args:
            doc_id: 文档ID
        Returns:
            bool: 保存是否成功
        """
        if not doc_id:
            logging.warning("doc_id 为空，跳过图片位置保存")
            return False
        position_data = self.get_position_data()
        if not position_data:
            logging.warning("没有位置数据，跳过图片位置保存")
            return False
        image_locations = position_data.get('image_locations', [])
        if not image_locations:
            logging.info("没有图片位置，跳过保存")
            return True
        try:
            from api.db.services.document_content_service import DocumentContentService
            middle_json_data = position_data.get('middle_json', {})
            content_list_data = position_data.get('content_list', [])
            content_record = DocumentContentService.create_document_content(
                doc_id=doc_id,
                markdown=None,
                monkeyocr_middle_json=middle_json_data,
                monkeyocr_content_list=content_list_data,
                monkeyocr_image_locations=image_locations,
                file_path="monkeyocr_images",
                file_name="image_locations"
            )
            logging.info(f"成功保存图片位置到数据库: {content_record['id']}")
            logging.info(f"图片位置: {image_locations}")
            return True
        except ImportError as e:
            logging.error(f"导入 DocumentContentService 失败: {e}")
            return False
        except Exception as e:
            logging.error(f"保存图片位置到数据库失败: {e}")
            return False 