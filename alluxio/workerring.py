import hashlib
import math
from typing import List
from typing import Set

import mmh3
from sortedcontainers import SortedDict


class WorkerNetAddress:
    def __init__(
        self,
        host,
        container_host="",
        rpc_port=29999,
        data_port=29997,
        web_port=30000,
        domain_socket_path="",
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

    @classmethod
    def create_worker_addresses(cls, host_string):
        hosts = [item.strip() for item in host_string.split(",")]
        worker_addresses = []
        for host in hosts:
            worker_address = cls(host)
            worker_addresses.append(worker_address)
        return worker_addresses


# TODO(lu) support hash ring refresh with ETCD updates
class ConsistentHashProvider:
    def __init__(
        self,
        worker_addresses: List[WorkerNetAddress],
        logger,
        num_virtual_nodes=2000,
        max_attempts=100,
    ):
        assert worker_addresses, "worker list is empty"
        self.max_attempts = max_attempts
        self.worker_addresses = worker_addresses
        self.logger = logger or logging.getLogger("ConsistentHashProvider")
        self.init_worker_ring(num_virtual_nodes)

    def init_worker_ring(self, num_virtual_nodes: int):
        hash_ring = SortedDict()
        weight = math.ceil(num_virtual_nodes / len(self.worker_addresses))
        for worker_address in self.worker_addresses:
            for i in range(weight):
                hash_key = self.hash(worker_address.__str__(), i)
                hash_ring[hash_key] = worker_address
        self.hash_ring = hash_ring

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
        return self.get_ceiling_value(self.hash(key, index))

    def get_ceiling_value(self, hash_key: int):
        key_index = self.hash_ring.bisect_right(hash_key)
        if key_index < len(self.hash_ring):
            ceiling_key = self.hash_ring.keys()[key_index]
            ceiling_value = self.hash_ring[ceiling_key]
            return ceiling_value
        else:
            return self.hash_ring.peekitem(0)[1]

    def hash(self, key: str, index: int) -> int:
        return mmh3.hash(f"{key}{index}")
