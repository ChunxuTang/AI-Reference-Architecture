import hashlib
import io
import os

import humanfriendly
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor


class AlluxioDataset(Dataset):
    def __init__(
        self, local_path, alluxio_ufs_path, alluxio_rest, transform, _logger
    ):
        self.alluxio_rest = alluxio_rest
        self.transform = transform
        self._logger = _logger
        self.data = []
        classes = [
            name
            for name in os.listdir(local_path)
            if os.path.isdir(os.path.join(local_path, name))
        ]
        index_to_class = {i: j for i, j in enumerate(classes)}
        self.class_to_index = {
            value: key for key, value in index_to_class.items()
        }
        for class_name in classes:
            local_class_path = os.path.join(local_path, class_name)
            image_names = [
                name
                for name in os.listdir(local_class_path)
                if os.path.isfile(os.path.join(local_class_path, name))
            ]
            for image_name in image_names:
                self.data.append(
                    [
                        alluxio_ufs_path.rstrip("/")
                        + "/"
                        + class_name
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
        self, alluxio_workers, alluxio_page_size, concurrency, _logger
    ):
        self.workers = [item.strip() for item in alluxio_workers.split(",")]
        self.page_size = humanfriendly.parse_size(alluxio_page_size)
        self._logger = _logger
        self.session = self.create_session(concurrency)
        self.executor = ThreadPoolExecutor(max_workers=concurrency)


    def create_session(self, concurrency):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency
        )
        session.mount("http://", adapter)
        return session

    def read_whole_file(self, file_path, page_number):
        file_id = self.get_file_id(file_path)
        worker_address = self.get_worker_address(file_id)
        page_index = 0

        def page_generator(page_index):
            page_content = self.read_file(worker_address, file_id, page_index)
            if not page_content:
                return None
            if len(page_content) < self.page_size:  # last page
                return page_content
            return page_content

        # Use the executor to map the page_generator function to the data
        pages = list(self.executor.map(page_generator, range(page_number)))

        # Remove None values from the list
        pages = [page for page in pages if page is not None]

        content = b"".join(pages)

        return content

    def read_file(self, worker_address, file_id, page_index):
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

    def get_worker_address(self, file_path):
        return self.workers[0]
