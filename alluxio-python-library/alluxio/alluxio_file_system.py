import hashlib
import io
import json
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
    """
    Access Alluxio file system

    Examples
    --------
    >>> alluxio = AlluxioFileSystem(dora_root="s3://mybucket/mypath", etcd_host=localhost)
    >>> print(alluxio.list_dir("s3://mybucket/mypath/dir"))
    [
        {
            "mType": "file",
            "mName": "myfile",
            "mLength": 77542
        }

    ]
    >>> print(alluxio.read_file("s3://mybucket/mypath/dir/myfile"))
    my_file_content
    """

    ALLUXIO_PAGE_SIZE_KEY = "alluxio.worker.page.store.page.size"
    LIST_URL_FORMAT = "http://{worker_host}:28080/v1/files"
    PAGE_URL_FORMAT = (
        "http://{worker_host}:28080/v1/file/{path_id}/page/{page_index}"
    )

    def __init__(
        self,
        dora_root,
        etcd_host=None,
        worker_hosts=None,
        options=None,
        logger=None,
        concurrency=64,
    ):
        """
        Inits Alluxio file system.

        Args:
            dora_root (str):
                The dora root ufs.
            etcd_host (str, optional):
                The hostname of ETCD to get worker addresses from
                Either etcd_host or worker_hosts should be provided, not both.
            worker_hosts (str, optional):
                The worker hostnames in host1,host2,host3 format. Either etcd_host or worker_hosts should be provided, not both.
            options (dict, optional):
                A dictionary of Alluxio property key and values.
                Note that Alluxio Python API only support a limited set of Alluxio properties.
            logger (Logger, optional):
                A logger instance for logging messages.
            concurrency (int, optional):
                The maximum number of concurrent operations. Default to 64.
        """
        if dora_root is None:
            raise ValueError("Must supply 'dora_root'")
        if etcd_host is None and worker_hosts is None:
            raise ValueError(
                "Must supply either 'etcd_host' or 'worker_hosts'"
            )
        if etcd_host and worker_hosts:
            raise ValueError(
                "Supply either 'etcd_host' or 'worker_hosts', not both"
            )
        self.dora_root = dora_root
        self.logger = logger or logging.getLogger("AlluxioRest")
        self.session = self._create_session(concurrency)
        # parse options
        page_size = "1MB"
        if options:
            if self.ALLUXIO_PAGE_SIZE_KEY in options:
                page_size = options[self.ALLUXIO_PAGE_SIZE_KEY]
                self.logger.debug(f"Page size is set to {page_size}")
        self.page_size = humanfriendly.parse_size(page_size, binary=True)
        # parse worker info to form hash ring
        worker_addresses = None
        if etcd_host:
            worker_addresses = EtcdClient(etcd_host).get_worker_addresses()
        else:
            worker_addresses = WorkerNetAddress.from_worker_hosts(worker_hosts)
        self.hash_provider = ConsistentHashProvider(
            worker_addresses, self.logger
        )

    def list_dir(self, path):
        """
        Lists the directory.

        Args:
            path (str): The full ufs path to list from

        Returns:
            list of dict: A list containing dictionaries, where each dictionary has:
                - mType (string): directory or file
                - mName (string): name of the directory/file
                - mLength (integer): length of the file or 0 for directory

        Example:
            [
                {
                    "mType": "file",
                    "mName": "my_file_name",
                    "mLength": 77542
                },
                {
                    "mType": "directory",
                    "mName": "my_dir_name",
                    "mLength": 0
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
            return json.loads(response.content)
        except Exception as e:
            raise type(e)(
                f"Error when listing full path {path} Alluxio path {rel_path}: error {e}"
            ) from e

    def read_file(self, file_path):
        """
        Reads a file.

        Args:
            file_path (str): The full ufs file path to read data from

        Returns:
            file content (str): The full file content
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
