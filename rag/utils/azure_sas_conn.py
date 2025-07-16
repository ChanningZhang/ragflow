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
import time
from io import BytesIO
from rag import settings
from rag.utils import singleton
from azure.storage.blob import ContainerClient


@singleton
class RAGFlowAzureSasBlob:
    def __init__(self):
        self.conn = None
        self.container_url = os.getenv('CONTAINER_URL', settings.AZURE["container_url"])
        self.sas_token = os.getenv('SAS_TOKEN', settings.AZURE["sas_token"])
        self.__open__()

    def __open__(self):
        try:
            if self.conn:
                self.__close__()
        except Exception:
            pass

        try:
            self.conn = ContainerClient.from_container_url(self.container_url + "?" + self.sas_token)
        except Exception:
            logging.exception("Fail to connect %s " % self.container_url)

    def __close__(self):
        del self.conn
        self.conn = None

    def health(self):
        _bucket, fnm, binary = "txtxtxtxt1", "txtxtxtxt1", b"_t@@@1"
        return self.conn.upload_blob(name=fnm, data=BytesIO(binary), length=len(binary))

    def put(self, bucket, fnm, binary):
        for _ in range(3):
            try:
                return self.conn.upload_blob(name=fnm, data=BytesIO(binary), length=len(binary))
            except Exception:
                logging.exception(f"Fail put {bucket}/{fnm}")
                self.__open__()
                time.sleep(1)

    def rm(self, bucket, fnm):
        try:
            self.conn.delete_blob(fnm)
        except Exception:
            logging.exception(f"Fail rm {bucket}/{fnm}")

    def get(self, bucket, fnm):
        for _ in range(1):
            try:
                r = self.conn.download_blob(fnm)
                return r.read()
            except Exception:
                logging.exception(f"fail get {bucket}/{fnm}")
                self.__open__()
                time.sleep(1)
        return

    def obj_exist(self, bucket, fnm):
        try:
            return self.conn.get_blob_client(fnm).exists()
        except Exception:
            logging.exception(f"Fail put {bucket}/{fnm}")
        return False

    def get_presigned_url(self, bucket, fnm, expires):
        for _ in range(10):
            try:
                return self.conn.get_presigned_url("GET", bucket, fnm, expires)
            except Exception:
                logging.exception(f"fail get {bucket}/{fnm}")
                self.__open__()
                time.sleep(1)
        return

    def copy_object(self, source_bucket, source_object, dest_bucket, dest_object):
        """
        Copy an object within Azure Blob Storage.
        Since Azure SAS only supports one container, this will copy objects within the same container.
        
        Args:
            source_bucket (str): Source bucket name (ignored in Azure SAS)
            source_object (str): Source object name/path
            dest_bucket (str): Destination bucket name (ignored in Azure SAS)
            dest_object (str): Destination object name/path
        
        Returns:
            Object copy result or None if failed
        """
        for _ in range(3):
            try:
                # Get source blob client
                source_blob_client = self.conn.get_blob_client(source_object)
                
                # Check if source exists
                if not source_blob_client.exists():
                    raise Exception(f"Source object {source_object} does not exist")
                
                # Get destination blob client
                dest_blob_client = self.conn.get_blob_client(dest_object)
                
                # Start copy operation using the blob URL
                source_url = source_blob_client.url
                copy_result = dest_blob_client.start_copy_from_url(source_url)
                
                return copy_result
            except Exception:
                logging.exception(f"Fail to copy {source_bucket}/{source_object} to {dest_bucket}/{dest_object}:")
                self.__open__()
                time.sleep(1)
        return None
