import hashlib
import io
import logging
import os

import humanfriendly
import requests
from requests.adapters import HTTPAdapter

from .worker_ring import ConsistentHashProvider
from .worker_ring import EtcdClient
from .worker_ring import WorkerNetAddress

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AlluxioFileSystem:
    ALLUXIO_PAGE_SIZE_KEY = "alluxio.worker.page.store.page.size"
    LIST_URL_FORMAT = "http://{worker_host}:28080/v1/files"
    PAGE_URL_FORMAT = (
        "http://{worker_host}:28080/v1/file/{path_id}/page/{page_index}"
    )

    def __init__(self, etcd_host, dora_root, options, logger, concurrency=64):
        self.dora_root = dora_root
        self.logger = logger or logging.getLogger("AlluxioRest")
        self.session = self._create_session(concurrency)
        self._parse_alluxio_rest_options(options)
        self.hash_provider = ConsistentHashProvider(etcd_host, self.logger)
        self.hash_provider.init_worker_ring()

    def list_dir(self, path):
        """
        Lists the directory.

        Args:
            path (str): The full ufs path to list from.

        Returns:
            list of dict: A list containing dictionaries, where each dictionary has:
                - mType (string): directory or file
                - nName (string): name of the directory/file.

        Example:
            [
                {
                    "mType": "file",
                    "nName": "my_file_name"
                },
                {
                    "mType": "directory",
                    "nName": "my_dir_name"
                },

            ]
        """
        path_id = self._get_path_hash(path)
        worker_host = self._get_preferred_worker_host(path)
        rel_path = self._subtract_path(path, self.dora_root)
        params = {"path": rel_path}
        try:
            response = self.session.get(
                self.LIST_URL_FORMAT.format(worker_host=worker_host),
                params=params,
            )
            response.raise_for_status()
            items = json.loads(response.content)
            print(items)
            return items
        except Exception as e:
            raise type(e)(
                f"Error when listing full path {path} Alluxio path {rel_path}: error {e}"
            ) from e

    def read_file(self, file_path):
        """
        Reads a file.

        Args:
            file_path (str): The full ufs file path to read data from.

        Returns:
            file content (str): The full file content.
        """
        worker_host = self._get_preferred_worker_host(file_path)
        path_id = self._get_path_hash(file_path)
        try:
            return b"".join(self._page_generator(worker_host, path_id))
        except Exception as e:
            raise type(e)(
                f"Error when reading file {file_path}: error {e}"
            ) from e

    def _page_generator(self, worker_host, path_id):
        page_index = 0
        while True:
            page_content = self._read_page(worker_host, path_id, page_index)
            if not page_content:
                break
            yield page_content
            if len(page_content) < self.page_size:  # last page
                break
            page_index += 1

    def _create_session(self, concurrency):
        session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency
        )
        session.mount("http://", adapter)
        return session

    def _parse_alluxio_rest_options(self, options):
        page_size = None
        if self.ALLUXIO_PAGE_SIZE_KEY in options:
            page_size = options[self.ALLUXIO_PAGE_SIZE_KEY]
            _logger.debug(f"Page size is set to {page_size}")
        else:
            page_size = "1MB"
        self.page_size = humanfriendly.parse_size(page_size, binary=True)

    def _read_page(self, worker_host, path_id, page_index):
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
        except Exception as e:
            raise type(e)(
                f"Error when requesting file {path_id} page {page_index} from {worker_host}: error {e}"
            ) from e

    def _get_path_hash(self, uri):
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

    def _get_preferred_worker_host(self, full_ufs_path):
        workers = self.hash_provider.get_multiple_workers(full_ufs_path, 1)
        if len(workers) != 1:
            raise ValueError(
                "Expected exactly one worker from hash ring, but found {} workers {}.".format(
                    len(workers), workers
                )
            )
        return workers[0].host

    def _subtract_path(self, path, parent_path):
        if "://" in path and "://" in parent_path:
            # Remove the parent_path from path
            relative_path = path[len(parent_path) :]
        else:
            # Get the relative path for local paths
            relative_path = os.path.relpath(path, start=parent_path)
            relative_path = "/" + relative_path
        return relative_path
