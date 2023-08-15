import json
import math
from typing import List
from typing import Set

import etcd3
import mmh3
from sortedcontainers import SortedDict

class WorkerNetAddress:
    DEFAULT_HOST = "localhost"
    DEFAULT_CONTAINER_HOST = ""
    DEFAULT_RPC_PORT = 29999
    DEFAULT_DATA_PORT = 29997
    DEFAULT_WEB_PORT = 30000
    DEFAULT_DOMAIN_SOCKET_PATH = ""

    def __init__(
        self,
        host,
        container_host=DEFAULT_CONTAINER_HOST,
        rpc_port=DEFAULT_RPC_PORT,
        data_port=DEFAULT_DATA_PORT,
        web_port=DEFAULT_WEB_PORT,
        domain_socket_path=DEFAULT_DOMAIN_SOCKET_PATH,
    ):
        self.host = host
        self.container_host = container_host
        self.rpc_port = rpc_port
        self.data_port = data_port  # Default is Netty data port 29997
        self.web_port = web_port
        self.domain_socket_path = domain_socket_path

    def __str__(self):
        return (
            "WorkerNetAddress{{host={}, containerHost={}, rpcPort={}, dataPort={}, webPort={}, domainSocketPath={}}}"
        ).format(
            self.host,
            self.container_host,
            self.rpc_port,
            self.data_port,
            self.web_port,
            self.domain_socket_path,
        )

# Note that EtcdClient should not be passed through python multiprocessing
class EtcdClient:
    PREFIX = "/DHT/DefaultAlluxioCluster/AUTHORIZED/"

    def __init__(self, host="localhost", port=2379):
        self.etcd = etcd3.client(host=host, port=port)

    def get_worker_addresses(self):
        workers = []

        for worker_info, metadata in self.etcd.get_prefix(self.PREFIX):
            # metadata constains key/version info, is ignore for now
            worker_info_string = worker_info.decode("utf-8")
            worker_info_json = json.loads(worker_info_string)
            worker_net_address = worker_info_json.get("WorkerNetAddress", "")

            worker_address = WorkerNetAddress(
                host=worker_net_address.get("Host", WorkerNetAddress.DEFAULT_HOST),
                container_host=worker_net_address.get("ContainerHost", WorkerNetAddress.DEFAULT_CONTAINER_HOST),
                rpc_port=worker_net_address.get("RpcPort", WorkerNetAddress.DEFAULT_RPC_PORT),
                data_port=worker_net_address.get("DataPort", WorkerNetAddress.DEFAULT_DATA_PORT),
                web_port=worker_net_address.get("WebPort", WorkerNetAddress.DEFAULT_WEB_PORT),
                domain_socket_path=worker_net_address.get(
                    "DomainSocketPath", WorkerNetAddress.DEFAULT_DOMAIN_SOCKET_PATH
                ),
            )
            workers.append(worker_address)

        return workers


class ConsistentHashProvider:
    def __init__(
        self,
        etcd_host,
        logger,
        num_virtual_nodes=2000,
        max_attempts=100,
    ):
        self.etcd_host = etcd_host
        self.num_virtual_nodes = num_virtual_nodes
        self.max_attempts = max_attempts
        self.logger = logger or logging.getLogger("ConsistentHashProvider")
        self._init_worker_ring()

    def get_multiple_workers(
        self, key: str, count: int
    ) -> List[WorkerNetAddress]:
        workers: Set[WorkerNetAddress] = set()
        attempts = 0
        while len(workers) < count and attempts < self.max_attempts:
            attempts += 1
            workers.add(self.get_worker(key, attempts))
        return list(workers)

    def get_worker(self, key: str, index: int) -> WorkerNetAddress:
        return self._get_ceiling_value(self._hash(key, index))

    def _init_worker_ring(self):
        worker_addresses = EtcdClient(self.etcd_host).get_worker_addresses()
        hash_ring = SortedDict()
        weight = math.ceil(self.num_virtual_nodes / len(worker_addresses))
        for worker_address in worker_addresses:
            worker_string = worker_address.__str__()
            for i in range(weight):
                hash_key = self._hash(worker_string, i)
                hash_ring[hash_key] = worker_address
        self.hash_ring = hash_ring

    def _get_ceiling_value(self, hash_key: int):
        key_index = self.hash_ring.bisect_right(hash_key)
        if key_index < len(self.hash_ring):
            ceiling_key = self.hash_ring.keys()[key_index]
            ceiling_value = self.hash_ring[ceiling_key]
            return ceiling_value
        else:
            return self.hash_ring.peekitem(0)[1]

    def _hash(self, key: str, index: int) -> int:
        return mmh3.hash(f"{key}{index}".encode("utf-8"))
