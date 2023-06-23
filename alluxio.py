import io
import os
import cv2
import hashlib
import requests
import humanfriendly

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class AlluxioDataset(Dataset):
    def __init__(self, local_path, alluxio_ufs_path, alluxio_rest, transform, _logger):
        self.alluxio_rest = alluxio_rest
        self.transform = transform
        self._logger = _logger
        self.data = []
        classes = [name for name in os.listdir(local_path) if os.path.isdir(os.path.join(local_path, name))]
        index_to_class = {i:j for i, j in enumerate(classes)}
        self.class_to_index = {value:key for key,value in index_to_class.items()}
        for class_name in classes:
            local_class_path = os.path.join(local_path, class_name)
            image_names = [name for name in os.listdir(local_class_path) if os.path.isfile(os.path.join(local_class_path, name))]
            for image_name in image_names:
                self.data.append([alluxio_ufs_path.rstrip("/") + "/" + class_name + "/" + image_name, class_name])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, class_name = self.data[index]
        image_content = self.alluxio_rest.read_whole_file(image_path)
        try:
            image = Image.open(io.BytesIO(image_content)).convert("RGB")
        except Exception as e:
            self._logger.error(
                f"Error when decoding image: {image_path}, error: {e}"
            )
            return None

        if self.transform is not None:
            image = self.transform(image)
        
        class_id = self.class_to_index[class_name]
        return image, class_id
    

# TODO support multiple workers
class AlluxioRest:
    def __init__(self, alluxio_workers, alluxio_page_size, _logger):
        self.alluxio_workers = [item.strip() for item in alluxio_workers.split(",")]
        self.alluxio_page_size = humanfriendly.parse_size(alluxio_page_size)
        self._logger = _logger
    
    def read_whole_file(self, file_path):
        file_id = self.get_file_id(file_path)
        worker_address = self.get_worker_address(file_id)
        content = b""
        page_index = 0
        while True:
            page_content = self.read_file(worker_address, file_id, page_index)
            if page_content is None or page_content == b"":
                break
            elif len(page_content) < self.alluxio_page_size: # last page
                content += page_content
                break
            else:
                content += page_content
                page_index += 1

        return content
        
    
    def read_file(self, worker_address, file_id, page_index):
        url = f"http://{worker_address}/page"
        params = {
            'fileId': file_id,
            'pageIndex': page_index
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            self._logger.error(
                f"Error when requesting image {file_id} page {page_index}: error {e}"
            )
            return None
        
    def get_file_id(self, uri):
        try:
            sha256_hash = hashlib.sha256()
            sha256_hash.update(uri.encode('utf-8'))
            return sha256_hash.hexdigest().lower()
        except hashlib.AlgorithmNotAvailable:
            # Continue with other hash method
            pass

        try:
            md5_hash = hashlib.md5()
            md5_hash.update(uri.encode('utf-8'))
            return md5_hash.hexdigest().lower()
        except hashlib.AlgorithmNotAvailable:
            # Continue with other hash method
            pass

        # Fallback to simple hashCode
        return hex(hash(uri))[2:].lower()
    
    def get_worker_address(self, file_path):
        return self.alluxio_workers[0]
        