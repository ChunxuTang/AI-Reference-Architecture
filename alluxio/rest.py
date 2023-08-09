import hashlib
import io
import json
import logging
import os

import humanfriendly
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset

from alluxio.workerring import ConsistentHashProvider
from alluxio.workerring import WorkerNetAddress

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AlluxioRestDataset(Dataset):
    def __init__(self, alluxio_rest, dataset_path, transform, logger):
        self.alluxio_rest = alluxio_rest
        self.transform = transform
        self.logger = logger or logging.getLogger("AlluxioRestDataset")
        self.data = []

        classes = [
            item["mName"]
            for item in json.loads(self.alluxio_rest.list_dir(dataset_path))
            if item["mType"] == "directory"
        ]

        index_to_class = {i: j for i, j in enumerate(classes)}

        self.class_to_index = {
            value: key for key, value in index_to_class.items()
        }

        for class_name in classes:
            class_path = dataset_path.rstrip("/") + "/" + class_name
            image_names = [
                item["mName"]
                for item in json.loads(self.alluxio_rest.list_dir(class_path))
                if item["mType"] == "file"
            ]
            for image_name in image_names:
                self.data.append(
                    [
                        class_path + "/" + image_name,
                        class_name,
                    ]
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, class_name = self.data[index]
        image_content = self.alluxio_rest.read_file(image_path)
        try:
            image = Image.open(io.BytesIO(image_content)).convert("RGB")
            self.logger.info(
                f"Succeed to get image: {image_path}"
            )
        except Exception as e:
            self.logger.error(
                f"Error when decoding image: {image_path}, error: {e}"
            )
            return None

        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_to_index[class_name]
        return image, class_id


# TODO support multiple workers
class AlluxioRest:
    ALLUXIO_PAGE_SIZE_KEY = "alluxio.worker.page.store.page.size"
    LIST_URL_FORMAT = "http://{worker_host}:28080/v1/files"
    PAGE_URL_FORMAT = (
        "http://{worker_host}:28080/v1/file/{path_id}/page/{page_index}"
    )

    def __init__(self, worker_hosts, dora_root, options, concurrency, logger):
        self.alluxio_workers = WorkerNetAddress.create_worker_addresses(
            worker_hosts
        )
        self.dora_root = dora_root
        self.logger = logger or logging.getLogger("AlluxioRest")
        self.session = self.create_session(concurrency)
        self.parse_alluxio_rest_options(options)
        self.hash_provider = ConsistentHashProvider(
            self.alluxio_workers, self.logger
        )

    def create_session(self, concurrency):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency
        )
        session.mount("http://", adapter)
        return session

    def parse_alluxio_rest_options(self, options):
        page_size = None
        if self.ALLUXIO_PAGE_SIZE_KEY in options:
            page_size = options[self.ALLUXIO_PAGE_SIZE_KEY]
            _logger.debug(f"Page size is set to {page_size}")
        else:
            page_size = "1MB"
        self.page_size = humanfriendly.parse_size(page_size)

    def list_dir(self, path):
        path_id = self.get_path_hash(path)
        worker_host = self.get_preferred_worker_host(path)
        rel_path = self.subtract_path(path, self.dora_root)
        params = {"path": rel_path}
        try:
            response = self.session.get(
                self.LIST_URL_FORMAT.format(worker_host=worker_host),
                params=params,
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error when listing path {rel_path}: error {e}")
            return None

    def read_file(self, file_path):
        path_id = self.get_path_hash(file_path)
        worker_host = self.get_preferred_worker_host(file_path)
        page_index = 0

        def page_generator():
            nonlocal page_index
            while True:
                page_content = self.read_page(worker_host, path_id, page_index)
                if not page_content:
                    return
                yield page_content
                if len(page_content) < self.page_size:  # last page
                    return
                page_index += 1

        content = b"".join(page_generator())
        return content

    def read_page(self, worker_host, path_id, page_index):
        try:
            response = self.session.get(
                self.PAGE_URL_FORMAT.format(
                    worker_host=worker_host,
                    path_id=path_id,
                    page_index=page_index,
                ),
            )
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            self.logger.error(
                f"Error when requesting file {path_id} page {page_index}: error {e}"
            )
            return None

    def get_path_hash(self, uri):
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

    def get_preferred_worker_host(self, full_ufs_path):
        # Java side uses full ufs path as key instead of path hash
        workers = self.hash_provider.get_multiple_workers(full_ufs_path, 1)
        if len(workers) != 1:
            raise ValueError(
                "Expected exactly one worker from hash ring, but found {} workers.".format(
                    len(workers)
                )
            )
        return workers[0].host

    def subtract_path(self, path, parent_path):
        if "://" in path and "://" in parent_path:
            # Remove the parent_path from path
            relative_path = path[len(parent_path) :]
        else:
            # Get the relative path for local paths
            relative_path = os.path.relpath(path, start=parent_path)
            relative_path = "/" + relative_path
        return relative_path
