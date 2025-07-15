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
    def create_document_content(cls, doc_id, markdown=None, monkeyocr_middle_json=None, 
                               monkeyocr_content_list=None, file_path=None, file_name=None):
        """
        创建文档内容记录
        
        Args:
            doc_id: 文档ID
            markdown: Markdown 文件内容
            monkeyocr_middle_json: MonkeyOCR 中间 JSON 数据
            monkeyocr_content_list: MonkeyOCR 内容列表数据
            file_path: 原始文件路径
            file_name: 文件名
            
        Returns:
            DocumentContent: 创建的文档内容记录
        """
        try:
            # 计算内容大小
            content_size = 0
            if markdown:
                content_size += len(markdown.encode('utf-8'))
            
            # 创建记录
            content_record = {
                "id": get_uuid(),
                "doc_id": doc_id,
                "markdown": markdown,
                "monkeyocr_middle_json": monkeyocr_middle_json,
                "monkeyocr_content_list": monkeyocr_content_list,
                "file_path": file_path,
                "file_name": file_name or (os.path.basename(file_path) if file_path else None),
                "content_size": content_size
            }
            
            # 插入数据库
            cls.model.create(**content_record)
            
            logging.info(f"成功创建文档内容记录: {content_record['id']}, 文档ID: {doc_id}")
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
            # 如果更新了 markdown 内容，重新计算大小
            if 'markdown' in kwargs:
                content_size = len(kwargs['markdown'].encode('utf-8')) if kwargs['markdown'] else 0
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
                "has_markdown": any(c.markdown for c in contents),
                "has_middle_json": any(c.monkeyocr_middle_json for c in contents),
                "has_content_list": any(c.monkeyocr_content_list for c in contents)
            }
            
            return stats
            
        except Exception as e:
            logging.error(f"获取文档内容统计信息失败: {e}")
            return {} 