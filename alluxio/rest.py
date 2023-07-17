import hashlib
import io
import os
import json

import humanfriendly
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset

class AlluxioRestDataset(Dataset):
    def __init__(
        self, alluxio_rest, dataset_path, transform, _logger
    ):
        self.alluxio_rest = alluxio_rest
        self.transform = transform
        self._logger = _logger
        self.data = []
        
        classes = [item['mName'] for item in json.loads(self.alluxio_rest.list_dir(dataset_path)) if item['mType'] == 'directory']

        index_to_class = {i: j for i, j in enumerate(classes)}
        
        self.class_to_index = {
            value: key for key, value in index_to_class.items()
        }

        for class_name in classes:
            class_path = dataset_path.rstrip("/") + "/" + class_name
            image_names = [item['mName'] for item in json.loads(self.alluxio_rest.list_dir(class_path)) if item['mType'] == 'file']
            for image_name in image_names:
                self.data.append(
                    [
                        class_path
                        + "/"
                        + image_name,
                        class_name,
                    ]
                )

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
    def __init__(
        self, endpoint, dora_root, page_size, concurrency, _logger
    ):
        self.workers = [item.strip() for item in endpoint.split(",")]
        self.dora_root = dora_root
        self.page_size = humanfriendly.parse_size(page_size)
        self._logger = _logger
        self.session = self.create_session(concurrency)

    def create_session(self, concurrency):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency
        )
        session.mount("http://", adapter)
        return session

    def list_dir(self, path):
        worker_address = self.get_worker_address()
        url = f"http://{worker_address}/files"
        rel_path = self.subtract_path(path, self.dora_root)
        params = {"path": rel_path}
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            self._logger.error(
                f"Error when listing path {rel_path}: error {e}"
            )
            return None  

    def read_whole_file(self, file_path):
        file_id = self.get_file_id(file_path)
        worker_address = self.get_worker_address()
        page_index = 0

        def page_generator():
            nonlocal page_index
            while True:
                page_content = self.read_page(
                    worker_address, file_id, page_index
                )
                if not page_content:
                    return
                yield page_content
                if len(page_content) < self.page_size:  # last page
                    return
                page_index += 1

        content = b"".join(page_generator())
        return content

    def read_page(self, worker_address, file_id, page_index):
        url = f"http://{worker_address}/page"
        params = {"fileId": file_id, "pageIndex": page_index}

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            self._logger.error(
                f"Error when requesting file {file_id} page {page_index}: error {e}"
            )
            return None

    def get_file_id(self, uri):
        hash_functions = [
            hashlib.sha256,
            hashlib.md5,
            lambda x: hex(hash(x))[2:].lower(),  # Fallback to simple hashCode
        ]
        for hash_function in hash_functions:
            try:
                hash_obj = hash_function()
                hash_obj.update(uri.encode("utf-8"))
                return hash_obj.hexdigest().lower()
            except AttributeError:
                continue

    def get_worker_address(self):
        return self.workers[0]

    def subtract_path(self, path, parent_path):
        if '://' in path and '://' in parent_path:
            # Remove the parent_path from path
            relative_path = path[len(parent_path):]
        else:
            # Get the relative path for local paths
            relative_path = os.path.relpath(path, start=parent_path)
        return relative_path
