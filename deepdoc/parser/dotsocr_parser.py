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
import tempfile
from io import BytesIO
import re
import base64
import uuid
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from PIL import Image

from dots_ocr.parser import DotsOCRParser as BaseDotsOCRParser
from dots_ocr.utils.doc_utils import load_images_from_pdf

# 导入任务取消异常类
class TaskCanceledException(Exception):
    """任务取消异常"""
    def __init__(self, msg):
        self.msg = msg
        super().__init__(msg)

class DotsOCRParser:
    """DotsOCR PDF解析器 - 调用 dots_ocr 模块进行文档解析"""
    
    def __init__(self, 
                #  ip='localhost',
                #  port=8000,
                 addr='localhost:8000',
                 model_name='model',
                 temperature=0.1,
                 top_p=1.0,
                 max_completion_tokens=16384,
                 num_thread=64,
                 dpi=200,
                 output_dir=None,
                 min_pixels=None,
                 max_pixels=None,
                 kb_id=None,  # 添加 kb_id 参数，用于 MinIO 上传
                 **kwargs):
        """
        初始化 DotsOCRParser
        
        Args:
            addr: vLLM 服务器地址
            model_name: 模型名称
            temperature: 生成温度
            top_p: top_p 参数
            max_completion_tokens: 最大完成令牌数
            num_thread: 线程数
            dpi: PDF 转图片的 DPI
            output_dir: 输出目录（可选）
            min_pixels: 最小像素数
            max_pixels: 最大像素数
            kb_id: 知识库 ID，用于 MinIO 上传
        """
        self.temp_dir = None
        if output_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix="dotsocr_")
            output_dir = self.temp_dir
        
        self.kb_id = kb_id  # 保存知识库 ID
        
        self.dots_parser = BaseDotsOCRParser(
            # ip=ip,
            # port=port,
            addr=addr,
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=max_completion_tokens,
            num_thread=num_thread,
            dpi=dpi,
            output_dir=output_dir,
            min_pixels=min_pixels,
            max_pixels=max_pixels
        )
        
        # 存储解析结果，用于 get_parse_result() 方法
        self._parse_result = None
        
        # 存储原始文件名信息，用于后续保存到document_content表
        self._original_filename = None
        
        logging.info(f"DotsOCRParser 初始化完成，服务器: {addr}, 模型: {model_name}")
    
    def _process_single_page(self, page_idx, image, start_page, prompt_mode, lock=None):
        """
        处理单页图片的函数，用于并发执行
        
        Args:
            page_idx: 页面索引（相对于selected_images的索引）
            image: 页面图片对象
            start_page: 起始页码
            prompt_mode: 提示模式
            lock: 线程锁（可选）
            
        Returns:
            dict: 包含页面处理结果的字典
        """
        actual_page = start_page + page_idx
        logging.debug(f"正在解析第 {actual_page + 1} 页 (索引: {page_idx})")
        
        try:
            # 使用基础 DotsOCRParser 解析单页图片
            page_result = self.dots_parser._parse_single_image(
                origin_image=image,
                prompt_mode=prompt_mode,
                save_dir=self.dots_parser.output_dir,
                save_name=f"page_{actual_page}",
                source="pdf",
                page_idx=actual_page
            )
                                    
            # 收集页面尺寸信息
            input_width = page_result.get('input_width', image.width)
            input_height = page_result.get('input_height', image.height)
            page_info = {
                "page_no": actual_page,
                "page_size": [input_width, input_height]
            }
            
            # 提取 Markdown 内容（优先使用内存数据）
            raw_page_text = page_result.get('md_content_data', "")
            if raw_page_text:
                # 处理 base64 图片
                page_text = self._process_base64_images(raw_page_text, actual_page)
            else:
                page_text = f"第 {actual_page + 1} 页解析失败"
            
            # 处理 JSON 数据
            layout_info_data = page_result.get('layout_info_data')
            
            # 处理图片数据并上传到 MinIO
            dotsocr_page_filename = None
            layout_image_data = page_result.get('layout_image_data')
            if layout_image_data and self.kb_id:
                try:
                    from rag.utils.storage_factory import STORAGE_IMPL
                except ImportError:
                    STORAGE_IMPL = None
                    
                if STORAGE_IMPL:
                    try:
                        image_filename = f"dotsocr_page_{page_idx}_{uuid.uuid4().hex}.jpg"
                        STORAGE_IMPL.put(self.kb_id, image_filename, layout_image_data)
                        dotsocr_page_filename = image_filename
                        logging.info(f"Successfully uploaded DotsOCR page image to MinIO: {self.kb_id}/{image_filename}")
                    except Exception as e:
                        logging.error(f"Failed to upload page image to MinIO: {e}")
            
            # 创建页面图片对象用于chunk预览
            page_image = None
            if layout_image_data:
                try:
                    from PIL import Image
                    import io
                    image_buffer = io.BytesIO(layout_image_data)
                    page_image = Image.open(image_buffer).convert('RGB')
                    logging.debug(f"成功创建第 {actual_page + 1} 页图片对象")
                except Exception as e:
                    logging.warning(f"创建第 {actual_page + 1} 页图片对象失败: {e}")
                    page_image = None
            
            logging.debug(f"页面 {actual_page + 1} 解析完成，文本长度: {len(page_text)}")
            
            return {
                'success': True,
                'page_idx': page_idx,
                'actual_page': actual_page,
                'page_info': page_info,
                'page_text': page_text,
                'page_image': page_image,
                'layout_info_data': layout_info_data,
                'dotsocr_page_filename': dotsocr_page_filename
            }
            
        except Exception as e:
            error_msg = f"解析第 {actual_page + 1} 页时出错: {str(e)}"
            logging.error(error_msg)
            return {
                'success': False,
                'page_idx': page_idx,
                'actual_page': actual_page,
                'error_msg': error_msg
            }
    
    def __call__(self, filename, binary=None, from_page=0, to_page=100000, 
                 prompt_mode="prompt_layout_all_en", callback=None, **kwargs):
        """
        DotsOCR解析器主函数
        
        Args:
            filename: 文件名或文件路径
            binary: 文件二进制数据
            from_page: 起始页（从0开始）
            to_page: 结束页
            prompt_mode: 提示模式，默认为 "prompt_layout_all_en"
            callback: 回调函数
            
        Returns:
            Tuple[List[Tuple[str, Optional[Image.Image]]], List]: (文本片段列表, 表格列表)
        """
        if callback is None:
            callback = lambda prog, msg: None
        
        # 存储原始文件名信息，用于后续保存到document_content表
        self._original_filename = filename
        
        # 验证输入
        if not binary:
            if not os.path.exists(filename):
                callback(-1, f"文件不存在: {filename}")
                return [], []
            
            with open(filename, 'rb') as f:
                binary = f.read()
        
        try:
            callback(0.03, "开始 DotsOCR 解析...")
            
            # 创建临时文件用于处理
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(binary)
                temp_file_path = temp_file.name
            
            try:
                callback(0.08, "PDF 转换为图片...")
                
                # 使用 dots_ocr 的工具将 PDF 转换为图片
                images = load_images_from_pdf(temp_file_path, dpi=self.dots_parser.dpi)
                total_pages = len(images)
                
                if total_pages == 0:
                    callback(-1, "PDF 转换失败，没有页面")
                    return [], []
                
                # 限制页面范围
                start_page = max(0, from_page)
                end_page = min(to_page, total_pages)
                selected_images = images[start_page:end_page]
                
                logging.info(f"PDF 转换完成，总页数: {total_pages}, 处理页数: {len(selected_images)}")
                callback(0.1, f"总共 {len(selected_images)} 页，开始解析...")
                
                # 逐页解析并收集结果（使用并发处理）
                all_tables = []
                
                logging.info(f"开始解析 {len(selected_images)} 页，从第 {start_page + 1} 页到第 {start_page + len(selected_images)} 页")
                
                # 初始化结果列表，预分配空间以确保按索引存储
                num_pages = len(selected_images)
                page_infos = [None] * num_pages
                dotsocr_md_list = [None] * num_pages
                dotsocr_json_data_list = [None] * num_pages
                dotsocr_json_list = [None] * num_pages
                dotsocr_page_list = [None] * num_pages
                all_sections = [None] * num_pages
                
                # 创建线程池并发执行
                # 从环境变量读取线程池大小配置，如果没有配置则使用默认值，但不超过页面数
                configured_thread_pool_size = int(os.environ.get('DOTSOCR_THREAD_NUM', '4'))
                thread_pool_size = min(configured_thread_pool_size, num_pages)
                logging.info(f"使用线程池大小: {thread_pool_size} 并发处理 {num_pages} 页")
                
                with ThreadPoolExecutor(max_workers=thread_pool_size) as executor:
                    # 提交所有任务
                    future_to_page = {}
                    for page_idx, image in enumerate(selected_images):
                        future = executor.submit(self._process_single_page, page_idx, image, start_page, prompt_mode)
                        future_to_page[future] = page_idx
                    
                    # 收集结果（Fork and Join模式）
                    completed_count = 0
                    for future in as_completed(future_to_page):
                        page_idx = future_to_page[future]
                        
                        try:
                            # 更新进度回调
                            completed_count += 1
                            callback(0.1 + 0.8 * (completed_count / num_pages), 
                                   f"已完成 {completed_count}/{num_pages} 页解析...")
                            
                            result = future.result()
                            
                            if result['success']:
                                # 成功处理的页面，按索引存储结果
                                page_infos[page_idx] = result['page_info']
                                dotsocr_md_list[page_idx] = result['page_text'] if result['page_text'] != f"第 {result['actual_page'] + 1} 页解析失败" else None
                                dotsocr_json_data_list[page_idx] = result['layout_info_data']
                                dotsocr_json_list[page_idx] = None  # 不再需要文件路径
                                dotsocr_page_list[page_idx] = result['dotsocr_page_filename']
                                all_sections[page_idx] = (result['page_text'], result['page_image'])
                            else:
                                # 失败的页面，存储错误信息
                                error_msg = result['error_msg']
                                logging.error(error_msg)
                                
                                # 为失败的页面存储默认值（按索引）
                                page_infos[page_idx] = None
                                dotsocr_md_list[page_idx] = None
                                dotsocr_json_data_list[page_idx] = None
                                dotsocr_json_list[page_idx] = None
                                dotsocr_page_list[page_idx] = None
                                all_sections[page_idx] = (error_msg, None)
                                
                        except TaskCanceledException as e:
                            # 任务取消异常需要向上传播，中断解析
                            logging.info(f"[DotsOCR] 任务在解析第 {result.get('actual_page', page_idx) + 1} 页时被取消: {e.msg}")
                            executor.shutdown(wait=False)  # 立即停止执行器
                            raise e
                        except Exception as e:
                            # 处理其他异常
                            actual_page = start_page + page_idx
                            error_msg = f"解析第 {actual_page + 1} 页时出错: {str(e)}"
                            logging.error(error_msg)
                            
                            # 为异常页面存储错误信息（按索引）
                            page_infos[page_idx] = None
                            dotsocr_md_list[page_idx] = None
                            dotsocr_json_data_list[page_idx] = None
                            dotsocr_json_list[page_idx] = None
                            dotsocr_page_list[page_idx] = None
                            all_sections[page_idx] = (error_msg, None)
                
                # 转换结果列表为原来代码期望的格式（过滤掉None值）
                page_infos = [info for info in page_infos if info is not None]
                logging.info(f"并发解析完成，成功处理 {len(page_infos)} 页")
                
                if all_sections:
                    callback(0.9, "生成页面级别的chunks...")
                    
                    # 为每页生成独立的chunk，保持页面内容完整性
                    final_sections = []
                    logging.info(f"开始处理 {len(all_sections)} 个解析结果")
                    for page_idx, (page_text, page_image) in enumerate(all_sections):
                        logging.debug(f"处理第 {page_idx + 1} 个解析结果，文本长度: {len(page_text) if page_text else 0}")
                        if page_text and page_text.strip():  # 只保留有内容的页面
                            final_sections.append((page_text.strip(), page_image))
                            logging.debug(f"添加第 {page_idx + 1} 个有效chunk")
                        else:
                            logging.debug(f"跳过第 {page_idx + 1} 个空内容")
                    
                    if not final_sections:
                        final_sections = [("解析失败：没有提取到有效内容", None)]
                        logging.warning("没有有效内容，添加失败提示chunk")
                        
                    logging.info(f"DotsOCR 解析完成，从 {len(all_sections)} 个解析结果生成 {len(final_sections)} 个页面chunk")
                else:
                    final_sections = [("解析失败：没有提取到内容", None)]
                
                # 设置解析结果，用于 get_parse_result() 方法
                self._parse_result = {
                    'middle_json': {
                        'pdf_info': page_infos
                    },
                    'content_list': [],
                    'total_pages': total_pages,
                    'processed_pages': len(selected_images),
                    'dotsocr_md_list': dotsocr_md_list,
                    'dotsocr_json_list': dotsocr_json_list,
                    'dotsocr_json_data': dotsocr_json_data_list,  # 添加内存中的 JSON 数据
                    'dotsocr_page_list': dotsocr_page_list,
                    'original_filename': self._original_filename  # 添加原始文件名信息
                }
                
                callback(1.0, "DotsOCR 解析完成")
                return final_sections, all_tables
                
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
        except TaskCanceledException as e:
            # 任务取消异常需要向上传播，不应该被当作普通错误处理
            logging.info(f"[DotsOCR] 任务被取消，停止解析: {e.msg}")
            raise e
        except Exception as e:
            error_msg = f"DotsOCR 解析失败: {str(e)}"
            logging.error(error_msg)
            callback(-1, error_msg)
            
            # 即使解析失败，也设置一个基础的 _parse_result 以避免后续错误
            self._parse_result = {
                'middle_json': {
                    'pdf_info': []
                },
                'content_list': [],
                'total_pages': 0,
                'processed_pages': 0,
                'error': error_msg,
                'original_filename': self._original_filename  # 添加原始文件名信息
            }
            
            return [], []
    
    def _combine_images_vertical(self, images):
        """垂直拼接多个图片"""
        if not images:
            return None
        if len(images) == 1:
            return images[0]
        
        # 计算总尺寸
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        # 创建合并后的图片
        combined = Image.new('RGB', (max_width, total_height), 'white')
        
        y_offset = 0
        for img in images:
            x_offset = (max_width - img.width) // 2  # 居中对齐
            combined.paste(img, (x_offset, y_offset))
            y_offset += img.height
        
        return combined
    

    def crop(self, text, ZM=3, need_position=False):
        """
        DotsOCR 的 crop 方法实现
        由于 DotsOCR 返回的是已经处理好的文本和图片，这里返回 None 表示没有对应的图片区域
        
        Args:
            text: 文本内容
            ZM: 缩放因子（未使用）
            need_position: 是否需要位置信息
            
        Returns:
            Tuple[Optional[Image.Image], Optional[List]]: (图片, 位置信息)
        """
        # DotsOCR 已经将图片和文本分离处理，所以这里返回 None
        if need_position:
            return None, None
        return None
    
    def remove_tag(self, txt):
        """
        DotsOCR 的 remove_tag 方法实现
        由于 DotsOCR 返回的是清理后的文本，这里直接返回原文本
        
        Args:
            txt: 文本内容
            
        Returns:
            str: 清理后的文本
        """
        # DotsOCR 已经返回清理后的文本，不需要额外的标签清理
        return txt
    
    def get_parse_result(self):
        """
        获取解析结果数据，与 MonkeyOCRParser 接口保持一致
        
        Returns:
            Dict: 解析结果字典，包含 middle_json 等信息
        """
        return self._parse_result
    
    def get_content_list(self):
        """
        获取 content_list 数据，与 MonkeyOCRParser 接口保持一致
        
        Returns:
            List[Dict]: content_list 数据，如果没有则返回 None
        """
        parse_result = self.get_parse_result()
        if parse_result and 'content_list' in parse_result:
            return parse_result['content_list']
        return None
    
    def _save_dotsocr_files(self, doc_id, parse_result):
        """
        保存 DotsOCR 解析结果到数据库
        
        Args:
            doc_id: 文档ID
            parse_result: 解析结果字典，包含 dotsocr_md_list、dotsocr_json_list、dotsocr_page_list 等数据
        """
        if not parse_result:
            logging.warning(f"DotsOCR 解析结果为空，文档ID: {doc_id}")
            return

        logging.info(f"开始保存 DotsOCR 解析结果，文档ID: {doc_id}")
        
        try:
            # 导入相关服务
            from api.db.services.document_content_service import DocumentContentService
            try:
                from rag.utils.storage_factory import STORAGE_IMPL
                logging.info(f"STORAGE_IMPL imported successfully: {STORAGE_IMPL is not None}")
            except ImportError as e:
                logging.error(f"Failed to import STORAGE_IMPL: {e}")
                STORAGE_IMPL = None
            
            # 获取 DotsOCR 数据
            dotsocr_md_list = parse_result.get('dotsocr_md_list', [])
            dotsocr_json_paths = parse_result.get('dotsocr_json_list', [])
            dotsocr_page_list = parse_result.get('dotsocr_page_list', [])
            middle_json_data = parse_result.get('middle_json', {})
            
            # 获取原始文件名信息，参考MonkeyOCR的实现
            original_filename = parse_result.get('original_filename')
            
            # 获取 JSON 数据内容（从内存中）
            dotsocr_json_data = parse_result.get('dotsocr_json_data', [])
            
            logging.info(f"DotsOCR 数据统计: md页数={len(dotsocr_md_list)}, json内容数={len(dotsocr_json_data)}, page图片数={len(dotsocr_page_list)}")
            logging.info(f"JSON 内容统计: {len([j for j in dotsocr_json_data if j is not None])} 个有效 JSON 对象")
            
            # 检查是否有有效的解析结果
            if not dotsocr_md_list or not any(dotsocr_md_list):
                logging.warning(f"DotsOCR 解析结果为空，文档ID: {doc_id}")
                return
                
            logging.info(f"DotsOCR 页面图片已在解析过程中上传到 MinIO: {len([p for p in dotsocr_page_list if p])} 个图片")
            
            # 合并所有页面的 Markdown 内容
            combined_content = ""
            for page_idx, md_content in enumerate(dotsocr_md_list):
                if md_content and md_content.strip():
                    if combined_content:
                        combined_content += f"\n\n{md_content}"
                    else:
                        combined_content = f"\n\n{md_content}"
            
            try:
                logging.info(f"准备创建文档内容记录，参数:")
                logging.info(f"  - doc_id: {doc_id}")
                logging.info(f"  - combined_content length: {len(combined_content.strip())}")
                logging.info(f"  - file_path: {original_filename}")
                logging.info(f"  - file_name: {os.path.basename(original_filename) if original_filename else None}")
                logging.info(f"  - dotsocr_md_list: {[md[:50] + '...' if md and len(md) > 50 else md for md in dotsocr_md_list]}")
                logging.info(f"  - dotsocr_json_list: {len([j for j in dotsocr_json_data if j is not None])} JSON objects (showing first 100 chars of each): {[str(j)[:100] + '...' if j and len(str(j)) > 100 else j for j in dotsocr_json_data[:2]]}")
                logging.info(f"  - dotsocr_page_list: {dotsocr_page_list}")
                
                # 创建文档内容记录，参考MonkeyOCR的实现传递file_path和file_name
                result = DocumentContentService.create_document_content(
                    doc_id=doc_id,
                    content=combined_content.strip(),
                    dotsocr_md_list=dotsocr_md_list,
                    dotsocr_json_list=dotsocr_json_data,
                    dotsocr_page_list=dotsocr_page_list,
                    file_path=original_filename,
                    file_name=os.path.basename(original_filename) if original_filename else None,
                    layout_recognize="DotsOCR",
                    content_type="markdown"
                )
                
                logging.info(f"成功保存 DotsOCR 解析结果: 文档ID={doc_id}, 返回结果: {result}")
                logging.info(f"  - Markdown 页数: {len([md for md in dotsocr_md_list if md])}")
                logging.info(f"  - JSON 文件数: {len([json for json in dotsocr_json_data if json])}")
                logging.info(f"  - 图片文件数: {len([page for page in dotsocr_page_list if page])}")
                    
            except Exception as e:
                logging.error(f"保存 DotsOCR 解析结果失败: {e}")
                import traceback
                logging.error(f"详细错误信息: {traceback.format_exc()}")
                    
        except ImportError as e:
            logging.error(f"导入 DocumentContentService 失败: {e}")
            logging.error("请确保 DocumentContentService 已正确配置")
        except Exception as e:
            logging.error(f"保存 DotsOCR 解析结果过程中发生错误: {e}")

    def _process_base64_images(self, markdown_content: str, page_num: int) -> str:
        """
        处理 Markdown 内容中的 base64 图片，将其上传到 MinIO 并替换为引用
        
        Args:
            markdown_content: 包含 base64 图片的原始 Markdown 内容
            page_num: 页面编号
            
        Returns:
            str: 处理后的 Markdown 内容
        """
        # 导入MinIO工具类（仅在需要时导入）
        try:
            from rag.utils.storage_factory import STORAGE_IMPL
            minio_available = True
            logging.debug("MinIO storage available for base64 image processing")
        except ImportError:
            logging.debug("STORAGE_IMPL not available, will use placeholder for base64 images")
            STORAGE_IMPL = None
            minio_available = False
        
        # base64 图片的正则表达式
        base64_pattern = r'!\[([^\]]*)\]\(data:image[^;]*;base64,([^)]+)\)'
        
        # 图片计数器，用于区分同一页面上的不同图片
        image_counter = 0
        
        def replace_base64_image(match):
            nonlocal image_counter
            image_counter += 1  # 每处理一张图片就递增
            alt_text = match.group(1)
            base64_data = match.group(2)
            
            try:
                # 解码 base64 数据
                image_data = base64.b64decode(base64_data)
                
                # 检测图片格式
                image_format = "png"  # 默认格式
                if image_data.startswith(b'\xff\xd8\xff'):
                    image_format = "jpg"
                elif image_data.startswith(b'\x89PNG'):
                    image_format = "png"
                elif image_data.startswith(b'GIF'):
                    image_format = "gif"
                elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
                    image_format = "webp"
                
                # 如果有 kb_id 且 MinIO 可用，则上传图片
                if minio_available and self.kb_id and STORAGE_IMPL:
                    # 生成唯一的图片文件名，使用检测到的格式
                    image_hash = uuid.uuid4().hex
                    image_filename = f"{image_hash}.{image_format}"
                    
                    try:
                        # 上传到 MinIO
                        STORAGE_IMPL.put(self.kb_id, image_filename, image_data)
                        minio_path = f"{self.kb_id}/{image_filename}"
                        
                        logging.info(f"Successfully uploaded base64 image to MinIO: {minio_path}")
                        
                        # 返回引用，使用images/前缀格式（与MonkeyOCR保持一致）
                        return f"![{alt_text}](images/{image_hash}.{image_format})"
                    
                    except Exception as e:
                        logging.error(f"Failed to upload base64 image to MinIO: {e}")
                        # 如果上传失败，返回带hash的占位符
                        image_hash = uuid.uuid4().hex
                        return f"![{alt_text}](images/{image_hash}.{image_format})"
                else:
                    # 如果没有 MinIO，返回带hash的图片路径格式（与MonkeyOCR保持一致）
                    image_hash = uuid.uuid4().hex
                    return f"![{alt_text}](images/{image_hash}.{image_format})"
                    
            except Exception as e:
                logging.error(f"Failed to process base64 image: {e}")
                # 处理失败时也返回hash格式的占位符（默认使用png）
                image_hash = uuid.uuid4().hex
                return f"![{alt_text}](images/{image_hash}.png)"
        
        # 替换所有 base64 图片
        processed_content = re.sub(base64_pattern, replace_base64_image, markdown_content)
        
        # 统计处理的图片数量
        base64_count = len(re.findall(base64_pattern, markdown_content))
        if base64_count > 0:
            logging.info(f"Processed {base64_count} base64 images in page {page_num}")
        
        return processed_content
    
    def __del__(self):
        """清理临时目录"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                logging.warning(f"清理临时目录失败: {e}")
