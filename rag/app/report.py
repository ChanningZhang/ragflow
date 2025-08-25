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

import logging
import os
import re
from typing import Dict, List, Tuple, Optional

from PIL import Image
from deepdoc.parser import MonkeyOCRParser
from rag.nlp import rag_tokenizer, tokenize_table, tokenize_chunks, tokenize_chunks_with_images, num_tokens_from_string, combine_images_with_monkeyocr_position, concat_img_with_page_limit


def report_merge_with_monkeyocr_images(texts, image_lists, monkeyocr_parser=None, chunk_token_num=128, delimiter="\n。；！？"):
    """
    Report 专用的图片合并函数，基于 naive_merge_with_image_lists
    主要区别：按页合并（每页一个 chunk）
    """
    logging.info(f"report_merge_with_monkeyocr_images called with {len(texts)} texts, {len(image_lists)} image_lists")
    
    if not texts or len(texts) != len(image_lists):
        logging.error(f"Mismatch: texts={len(texts)}, image_lists={len(image_lists)}")
        return [], []
    
    # 确保 texts 是字符串列表
    if isinstance(texts[0], tuple):
        texts = [t[0] for t in texts]
        logging.info("Converted tuple texts to string list")
    
    # 获取页面尺寸信息
    page_width, page_height = None, None
    if monkeyocr_parser:
        try:
            parse_result = monkeyocr_parser.get_parse_result()
            if parse_result and parse_result.get('middle_json'):
                middle_json_data = parse_result['middle_json']
                pdf_info = middle_json_data.get("pdf_info", [])
                if pdf_info:
                    page_size = pdf_info[0].get("page_size")
                    if page_size and len(page_size) == 2:
                        page_width, page_height = page_size[0], page_size[1]
                        logging.info(f"Got page size from parser: {page_width}x{page_height}")
                    else:
                        logging.error(f"Invalid page_size from parser: {page_size}")
                        raise ValueError(f"Invalid page_size from parser: {page_size}")
                else:
                    logging.error("No pdf_info in middle_json")
                    raise ValueError("No pdf_info in middle_json, cannot determine page size")
            else:
                logging.error("No parse result from parser")
                raise ValueError("No parse result from parser, cannot determine page size")
        except Exception as e:
            logging.error(f"Failed to get page size from parser: {e}")
            raise ValueError(f"Failed to get page size from parser: {e}")
    else:
        logging.error("No monkeyocr_parser provided")
        raise ValueError("No monkeyocr_parser provided, cannot determine page size")
    
    if not page_width or not page_height or page_width <= 0 or page_height <= 0:
        logging.error(f"Invalid page dimensions: {page_width}x{page_height}")
        raise ValueError(f"Invalid page dimensions: {page_width}x{page_height}")
    
    # 尝试使用 content_list.json 进行基于页面的分割
    content_list = monkeyocr_parser.get_content_list() if monkeyocr_parser else None
    if content_list:
        logging.info(f"Using content_list.json with {len(content_list)} paragraphs for page-based chunk splitting")
        try:
            return _report_merge_with_content_list(texts, image_lists, content_list, monkeyocr_parser, page_width, page_height)
        except Exception as e:
            logging.warning(f"Content list based splitting failed: {e}, falling back to simple page-based splitting")
            return _report_merge_fallback(texts, image_lists, monkeyocr_parser, page_width, page_height)
    else:
        logging.warning("No content_list.json available, falling back to simple page-based splitting")
        return _report_merge_fallback(texts, image_lists, monkeyocr_parser, page_width, page_height)


def _report_merge_with_content_list(texts, image_lists, content_list, monkeyocr_parser, page_width, page_height) -> Tuple[List[str], List[Optional[Image.Image]]]:
    """
    基于 content_list.json 进行 chunk 分割（Report 专用）
    合并规则：
    1. page_idx相同表示同一页
    2. 同一页中的"type": "text"或"table"项，拼接为一个文本chunk
    3. 同一页中的"type": "image"项，合并为一个图片chunk
    4. 合并时，文本拼接按原始content_list顺序
    每页最多两个chunk：文本chunk和图片chunk
    """
    logging.info(f"Processing {len(content_list)} content paragraphs with per-page text+image merging")

    # 收集所有图片（保持顺序）
    all_images = []
    for img_list in image_lists:
        all_images.extend(img_list)

    current_image_index = 0

    # 输出容器
    cks: List[str] = []
    result_images: List[Optional[Image.Image]] = []

    # 当前页缓冲
    current_page_idx: Optional[int] = None
    current_page_texts: List[str] = []
    current_page_images: List[Image.Image] = []
    current_page_image_paths: List[Optional[str]] = []

    def _flush_current_page():
        """输出当前页，将文本与图片合并为一个chunk（文本为内容拼接，图片为合并图）。"""
        nonlocal current_page_idx, current_page_texts, current_page_images, current_page_image_paths

        if current_page_idx is None:
            return

        # 合并文本（为空则使用占位，避免下游丢弃图片）
        page_text = "\n".join(current_page_texts).strip()
        if not page_text:
            page_text = f"[Page {current_page_idx + 1}]"

        # 合并图片（若有）
        combined_img = None
        if current_page_images:
            try:
                parse_result = monkeyocr_parser.get_parse_result()
                if parse_result and parse_result.get('middle_json'):
                    # logging.info(f"MonkeyOCR: Combining {len(current_page_images)} images on page {current_page_idx} with position info")
                    combined_img = combine_images_with_monkeyocr_position(current_page_images, parse_result, current_page_image_paths)
                else:
                    # logging.info(f"MonkeyOCR: No parse result, vertical combining images on page {current_page_idx}")
                    combined_img = concat_img_with_page_limit(current_page_images, max_width=page_width, max_height=page_height)
            except Exception as e:
                logging.error(f"Error processing images for page {current_page_idx}: {e}")
                combined_img = concat_img_with_page_limit(current_page_images, max_width=page_width, max_height=page_height)

        # 生成单一chunk（文本+图片）
        cks.append(page_text)
        result_images.append(combined_img)
        # logging.info(f"Page {current_page_idx}: Created unified chunk with text_len={len(page_text)} and {'image' if combined_img else 'no image'}")

        # 重置缓冲
        current_page_texts = []
        current_page_images = []
        current_page_image_paths = []

    # 按顺序处理每个 content 项
    for i, content_item in enumerate(content_list):
        page_idx = content_item.get('page_idx', -1)
        if page_idx < 0:
            continue

        content_type = content_item.get('type', '')

        # 页面变化则刷新上一页
        if current_page_idx is not None and current_page_idx != page_idx:
            _flush_current_page()

        # 设定当前页
        current_page_idx = page_idx

        if content_type == 'text':
            content_text = content_item.get('text', '').strip()
            if content_text:
                current_page_texts.append(content_text)
                logging.debug(f"Added text to page {page_idx}: {content_text[:50]}...")
                # 如果文本中带有图片引用，顺序分配一张图片到本页图片集合
                if '![' in content_text and '](' in content_text and current_image_index < len(all_images):
                    current_page_images.append(all_images[current_image_index])
                    current_page_image_paths.append(getattr(all_images[current_image_index], 'path', None))
                    current_image_index += 1

        elif content_type == 'table':
            table_parts: List[str] = []
            captions = content_item.get('table_caption') or []
            if isinstance(captions, list) and captions:
                caption_text = "\n".join(str(c).strip() for c in captions if str(c).strip())
                if caption_text:
                    table_parts.append(caption_text)
            table_body = (content_item.get('table_body') or '').strip()
            if table_body:
                table_parts.append(table_body)
            if table_parts:
                table_text = "\n".join(table_parts)
                current_page_texts.append(table_text)
                logging.debug(f"Added table to page {page_idx}: {table_text[:50]}...")

        elif content_type == 'image':
            # 仅收集图片，稍后在 flush 时合并为一个图片chunk
            img_path = content_item.get('img_path', '')
            if current_image_index < len(all_images):
                img = all_images[current_image_index]
                current_page_images.append(img)
                current_page_image_paths.append(img_path or getattr(img, 'path', None))
                current_image_index += 1
                logging.debug(f"Added image for page {page_idx}: {os.path.basename(img_path) if img_path else 'N/A'}")

            # 将图片的说明文字拼接到当前页文本中
            captions = content_item.get('img_caption')
            if captions:
                if isinstance(captions, list):
                    caption_text = "\n".join(str(c).strip() for c in captions if str(c).strip())
                else:
                    caption_text = str(captions).strip()
                if caption_text:
                    current_page_texts.append(caption_text)

    # 处理最后一页
    _flush_current_page()

    logging.info(f"report_merge_with_content_list completed: {len(cks)} chunks; {sum(1 for x in result_images if x is not None)} image chunks")
    return cks, result_images


def _report_merge_fallback(texts, image_lists, monkeyocr_parser, page_width, page_height) -> Tuple[List[str], List[Optional[Image.Image]]]:
    """
    简单的按页回退方案（当 content_list 不可用时）
    """
    logging.info("Using simple page-based chunk splitting as fallback")
    
    chunks = []
    images = []
    
    for i, (text, img_list) in enumerate(zip(texts, image_lists)):
        page_text = (text or "").strip()
        if page_text:
            chunks.append(page_text)
        else:
            chunks.append(f"[Page {i+1}]")  # 空页面占位符
            
        # 处理该页的图片
        if not img_list:
            images.append(None)
        else:
            try:
                combined_img = concat_img_with_page_limit(img_list, max_width=page_width, max_height=page_height)
                images.append(combined_img)
            except Exception as e:
                logging.error(f"Error processing images for page {i}: {e}")
                images.append(None)
    
    logging.info(f"Fallback merge completed: {len(chunks)} page-based chunks")
    return chunks, images


def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):
    """
    Report 专用 chunker：
    - 仅支持 PDF + MonkeyOCR。
    - 表格沿用与 naive 相同的 tokenize 逻辑。
    - 文本基于 content_list，按 page_idx 合并为每页一个 chunk。
    """

    if not re.search(r"\.pdf$", filename, re.IGNORECASE):
        raise NotImplementedError("report chunker 仅支持 PDF 文件（MonkeyOCR）")

    parser_config = kwargs.get(
        "parser_config", {
            "chunk_token_num": 128, "delimiter": "\n!?。；！？", "layout_recognize": "MonkeyOCR"})

    layout_recognizer = parser_config.get("layout_recognize", "MonkeyOCR")
    if layout_recognizer != "MonkeyOCR":
        raise NotImplementedError("report chunker 仅支持 MonkeyOCR 模式")

    is_english = (lang.lower() == "english")
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))
    }
    doc["title_sm_tks"] = rag_tokenizer.fine_grained_tokenize(doc["title_tks"])

    # 初始化 MonkeyOCR 解析器
    monkeyocr_url = os.environ.get('MONKEYOCR_URL', 'http://localhost:6006')
    timeout = int(os.environ.get('MONKEYOCR_TIMEOUT', '300'))
    kb_id = kwargs.get('kb_id')
    pdf_parser = MonkeyOCRParser(monkeyocr_url=monkeyocr_url, timeout=timeout, kb_id=kb_id)
    logging.info(f"[report] MonkeyOCR parser initialized - URL: {monkeyocr_url}, Timeout: {timeout}, KB ID: {kb_id}")

    callback(0.1, "Start to parse.")
    sections, tables = pdf_parser(filename, binary, from_page=from_page, to_page=to_page,
                                  callback=callback)

    # 表格先按 naive 逻辑处理
    res = tokenize_table(tables, doc, is_english)
    callback(0.7, "Tables tokenized.")

    # 检查是否有图片信息
    has_images = False
    if sections:
        # MonkeyOCR 返回的 sections 格式是 [(text, image_list), ...]，其中 image_list 是图片列表
        has_images = any(len(section[1]) > 0 for section in sections if len(section) > 1)
        logging.info(f"MonkeyOCR image check: has_images={has_images}, sections with images: {sum(1 for section in sections if len(section) > 1 and section[1])}")
    
    if has_images:
        # MonkeyOCR: sections 格式是 [(text, image_list), ...]，其中 image_list 是图片列表
        texts = [section[0] for section in sections]
        # 直接使用图片列表，无需转换
        image_lists = [section[1] if len(section) > 1 else [] for section in sections]
        
        logging.info(f"MonkeyOCR: Processing {len(sections)} sections with {sum(1 for img_list in image_lists if img_list)} image lists")
        
        # 详细记录每个section的图片信息
        for i, (text, img_list) in enumerate(zip(texts, image_lists)):
            if img_list:
                logging.debug(f"Section {i}: {len(img_list)} images, text preview: {text[:50]}...")
                for j, img in enumerate(img_list):
                    if hasattr(img, 'size'):
                        logging.debug(f"  Image {j}: size={img.size}")
                    else:
                        logging.debug(f"  Image {j}: type={type(img)}")
                        
                    # 尝试获取图片的文件名信息（如果有的话）
                    if hasattr(img, 'filename'):
                        logging.debug(f"  Image {j} filename: {img.filename}")
                    elif hasattr(img, 'info') and img.info:
                        logging.debug(f"  Image {j} info: {img.info}")
            else:
                logging.debug(f"Section {i}: no images, text preview: {text[:50]}...")
        
        chunks, images = report_merge_with_monkeyocr_images(texts, image_lists, pdf_parser,
                                        int(parser_config.get(
                                            "chunk_token_num", 128)), parser_config.get(
                                            "delimiter", "\n!?。；！？"))
        
        logging.info(f"MonkeyOCR: Generated {len(chunks)} chunks with {sum(1 for img in images if img is not None)} images")
        
        # 详细记录每个chunk的信息
        for i, (chunk, img) in enumerate(zip(chunks, images)):
            if img:
                logging.debug(f"Chunk {i}: has image (size={img.size}), text length: {len(chunk)}")
                logging.debug(f"Chunk {i} text preview: {chunk[:100]}...")
            else:
                logging.debug(f"Chunk {i}: no image, text length: {len(chunk)}")
        
        if kwargs.get("section_only", False):
            return chunks
        
        res.extend(tokenize_chunks_with_images(chunks, doc, is_english, images))
    else:
        logging.info("No images found, using simple page-based merging")
        # fallback：将 sections 中每页文本合并（如果有）
        # sections 通常为 [(text, image_or_list), ...]，此处仅抽取文本
        page_texts: List[str] = []
        for txt, *_ in sections:
            page_texts.append((txt or "").strip())
        chunks = page_texts
        
        if kwargs.get("section_only", False):
            return chunks

        res.extend(tokenize_chunks(chunks, doc, is_english, pdf_parser))
    
    callback(0.95, "Finish parsing.")
    return res


if __name__ == "__main__":
    import sys

    def dummy(prog=None, msg=""):
        pass

    chunk(sys.argv[1], from_page=0, to_page=10, callback=dummy)


