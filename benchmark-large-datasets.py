"""
A script to benchmark data loading of large datasets to compare Alluxio + FUSE
and Alluxio REST APIs.

Example usage:
Note that replace endpoints localhost to your worker_ip
POSIX API
- python3 benchmark-large-datasets.py -a posix -p /mnt/alluxio/fuse/yelp-review/yelp_academic_dataset_review.json
REST API
- python3 benchmark-large-datasets.py -a rest -p s3://ref-arch/yelp-review/yelp_academic_dataset_review.json -d s3://ref-arch/ --endpoints localhost:28080 --pagesize 20MB
S3 API
- python3 benchmark-large-datasets.py -a s3 -p s3://ref-arch/yelp-review/yelp_academic_dataset_review.json -d s3://ref-arch/ --endpoints localhost:29998
"""
import argparse
import logging
import time
from enum import Enum
from logging.config import fileConfig

from alluxio.rest import AlluxioRest
from alluxio.s3 import AlluxioS3

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)

class APIType(Enum):
    POSIX = "posix"
    REST = "rest"
    S3 = "s3"

def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark loading Large datasets like the Yelp Review dataset"
    )

    parser.add_argument(
        "-a",
        "--api",
        help="The API to use. default is posix",
        choices=[e.value for e in APIType],
        default=APIType.POSIX.value
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Local POSIX PATH if API type is POSIX, full ufs path if REST/S3 API (e.g.s3://ref-arch/yelp-review/yelp_academic_dataset_review.json)",
        default="./data/yelp-review/yelp_academic_dataset_review.json",
    )
    parser.add_argument(
        "-d",
        "--doraroot",
        help="Alluxio REST/S3 API require Dora root ufs address to do path transformation",
        default="s3://ref-arch/",
    )
    parser.add_argument(
        "--endpoints",
        help="Alluxio worker REST/S3 endpoints in list of host:port,host2:port2 format (e.g. localhost:28080 for REST API, localhost:29998 for S3 API)",
        default="localhost:28080",
    )
    parser.add_argument(
        "--pagesize",
        help="REST API: Alluxio page size (e.g. 1MB, 4MB, 1024KB)",
        default="1MB",
    )

    return parser.parse_args()


class BenchmarkLargeDatasetRunner:
    _logger = logging.getLogger("BenchmarkLargeDatasetRunner")

    def __init__(
        self,
        api,
        path,
        dora_root,
        endpoints,
        page_size,
    ):
        self.api=api
        self.path=path
        self.dora_root=dora_root
        self.endpoints=endpoints
        self.page_size=page_size

    def benchmark_data_loading(self):
        start_time = time.perf_counter()

        if self.api == APIType.REST.value:
            self._logger.debug(
                f"Using alluxio REST API reading file with workers {self.endpoints}, dora root {self.dora_root}, page size {self.page_size}, ufs path {self.path}"
            )
            alluxio_rest = AlluxioRest(
                self.endpoints,
                self.dora_root,
                self.page_size,
                1,# Only using one thread
                self._logger,
            )
            alluxio_rest.read_file(self.path)
        elif self.api == APIType.POSIX.value:
            self._logger.debug(
                f"Using POSIX API reading file with path {self.path}"
            )
            with open(self.path, "r") as f:
                f.read()
        else:
            self._logger.debug(
                f"Using alluxio S3 API reading file with workers {self.endpoints}, dora root {self.dora_root}, ufs path {self.path}"
            )
            alluxio_s3 = AlluxioS3(
                self.endpoints,
                self.dora_root,
                self._logger,
            )
            alluxio_s3.read_file(self.path)

        end_time = time.perf_counter()
        self._logger.info(f"Data loading in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    args = get_args()

    benchmark_large_dataset_runner = BenchmarkLargeDatasetRunner(
        api=args.api,
        path=args.path,
        dora_root=args.doraroot,
        endpoints=args.endpoints,
        page_size=args.pagesize,
    )
    benchmark_large_dataset_runner.benchmark_data_loading()
