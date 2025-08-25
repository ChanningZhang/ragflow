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

from api.db.services.common_service import CommonService
from api.db.db_models import DocumentContent, DB
from api.utils import get_uuid


class DocumentContentService(CommonService):
    model = DocumentContent

    @classmethod
    @DB.connection_context()
    def create_document_content(cls, doc_id, monkeyocr_middle_json=None, 
                               monkeyocr_content_list=None, monkeyocr_image_locations=None, 
                               dotsocr_md=None, dotsocr_json=None, dotsocr_page=None,
                               file_path=None, file_name=None, content=None, 
                               layout_recognize=None, content_type=None):
        """
        创建文档内容记录
        
        Args:
            doc_id: 文档ID
            monkeyocr_middle_json: MonkeyOCR 中间 JSON 数据
            monkeyocr_content_list: MonkeyOCR 内容列表数据
            monkeyocr_image_locations: MonkeyOCR 图片位置数组（对象结构: {image_name, location}）
            file_path: 原始文件路径
            file_name: 文件名
            content: 通用内容字段
            layout_recognize: 布局识别类型
            content_type: 内容类型 (markdown, text, json等)
            
        Returns:
            DocumentContent: 创建的文档内容记录
        """
        try:
            # 计算内容大小
            content_size = 0
            if content:
                content_size += len(content.encode('utf-8'))
            
            # 创建记录
            content_record = {
                "id": get_uuid(),
                "doc_id": doc_id,
                "monkeyocr_middle_json": monkeyocr_middle_json,
                "monkeyocr_content_list": monkeyocr_content_list,
                "monkeyocr_image_locations": monkeyocr_image_locations,
                "dotsocr_md": dotsocr_md,
                "dotsocr_json": dotsocr_json,
                "dotsocr_page": dotsocr_page,
                "file_path": file_path,
                "file_name": file_name or (os.path.basename(file_path) if file_path else None),
                "content_size": content_size,
                "content": content,
                "layout_recognize": layout_recognize,
                "content_type": content_type or "text"
            }
            
            # 插入数据库
            cls.model.create(**content_record)
            
            logging.info(f"成功创建文档内容记录: {content_record['id']}, 文档ID: {doc_id}, 布局识别: {layout_recognize}")
            return content_record
            
        except Exception as e:
            logging.error(f"创建文档内容记录失败: {e}")
            raise

    @classmethod
    @DB.connection_context()
    def get_by_doc_id(cls, doc_id):
        """
        根据文档ID获取文档内容记录
        
        Args:
            doc_id: 文档ID
            
        Returns:
            List[DocumentContent]: 文档内容记录列表
        """
        try:
            return list(cls.model.select().where(cls.model.doc_id == doc_id))
        except Exception as e:
            logging.error(f"获取文档内容记录失败: {e}")
            return []

    @classmethod
    @DB.connection_context()
    def update_document_content(cls, content_id, **kwargs):
        """
        更新文档内容记录
        
        Args:
            content_id: 内容记录ID
            **kwargs: 要更新的字段
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 如果更新了 content 内容，重新计算大小
            if 'content' in kwargs:
                content_size = len(kwargs['content'].encode('utf-8')) if kwargs['content'] else 0
                kwargs['content_size'] = content_size
            
            cls.update_by_id(content_id, kwargs)
            logging.info(f"成功更新文档内容记录: {content_id}")
            return True
            
        except Exception as e:
            logging.error(f"更新文档内容记录失败: {e}")
            return False

    @classmethod
    @DB.connection_context()
    def delete_by_doc_id(cls, doc_id):
        """
        根据文档ID删除文档内容记录
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 删除是否成功
        """
        try:
            deleted_count = cls.model.delete().where(cls.model.doc_id == doc_id).execute()
            logging.info(f"成功删除文档内容记录: doc_id={doc_id}, 删除数量={deleted_count}")
            return True
            
        except Exception as e:
            logging.error(f"删除文档内容记录失败: {e}")
            return False

    @classmethod
    @DB.connection_context()
    def delete_by_doc_id(cls, doc_id):
        """
        根据文档ID删除文档内容记录
        
        Args:
            doc_id: 文档ID
            
        Returns:
            int: 删除的记录数
        """
        try:
            deleted_count = cls.model.delete().where(cls.model.doc_id == doc_id).execute()
            logging.info(f"成功删除文档内容记录: {deleted_count} 条, 文档ID: {doc_id}")
            return deleted_count
            
        except Exception as e:
            logging.error(f"删除文档内容记录失败: {e}")
            return 0

    @classmethod
    @DB.connection_context()
    def get_content_statistics(cls, doc_id):
        """
        获取文档内容统计信息
        
        Args:
            doc_id: 文档ID
            
        Returns:
            dict: 统计信息
        """
        try:
            contents = cls.get_by_doc_id(doc_id)
            
            stats = {
                "total_records": len(contents),
                "total_size": sum(c.content_size for c in contents),
                "has_content": any(c.content for c in contents),
                "has_middle_json": any(c.monkeyocr_middle_json for c in contents),
                "has_content_list": any(c.monkeyocr_content_list for c in contents),
                "has_image_locations": any(c.monkeyocr_image_locations for c in contents),
                "total_images": sum(len(c.monkeyocr_image_locations) if c.monkeyocr_image_locations else 0 for c in contents)
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"获取文档内容统计信息失败: {e}")
            return {}

    @classmethod
    @DB.connection_context()
    def get_image_locations_by_doc_id(cls, doc_id):
        """
        根据文档ID获取图片位置对象数组
        
        Args:
            doc_id: 文档ID
            
        Returns:
            List[dict]: 图片位置对象数组
        """
        try:
            contents = cls.get_by_doc_id(doc_id)
            image_locations = []
            for content in contents:
                if content.monkeyocr_image_locations:
                    image_locations.extend(content.monkeyocr_image_locations)
            return image_locations
        except Exception as e:
            logging.error(f"获取图片位置失败: {e}")
            return []

    @classmethod
    @DB.connection_context()
    def update_image_locations(cls, content_id, image_locations):
        """
        更新文档内容的图片位置
        
        Args:
            content_id: 内容记录ID
            image_locations: 图片位置对象数组
            
        Returns:
            bool: 更新是否成功
        """
        try:
            cls.update_by_id(content_id, {"monkeyocr_image_locations": image_locations})
            logging.info(f"成功更新图片位置: {content_id}")
            return True
        except Exception as e:
            logging.error(f"更新图片位置失败: {e}")
            return False 