"""
A script to benchmark data loading of large datasets to compare Alluxio + FUSE
and Alluxio REST APIs.

Example usage:
Note that replace endpoints localhost to your worker_ip
POSIX API
- python3 benchmark-large-datasets.py -a posix -p /mnt/alluxio/fuse/yelp-review/yelp_academic_dataset_review.json
REST API
- python3 benchmark-large-datasets.py -a rest -p s3://ref-arch/yelp-review/yelp_academic_dataset_review.json -d s3://ref-arch/ --alluxioworkers localhost
- Configure a different page size, add -o alluxio.worker.page.store.page.size=20MB
S3 API
- python3 benchmark-large-datasets.py -a s3 -p s3://ref-arch/yelp-review/yelp_academic_dataset_review.json -d s3://ref-arch/ --alluxioworkers localhost
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
_logger = logging.getLogger("BenchmarkLargeDatasetLoading")


class APIType(Enum):
    POSIX = "posix"
    REST = "rest"
    S3 = "s3"


def parse_options(options_str):
    options_dict = {}
    if options_str:
        key_value_pairs = options_str.split(",")
        for pair in key_value_pairs:
            key, value = pair.split("=")
            options_dict[key.strip()] = value.strip()
    return options_dict


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark loading Large datasets like the "
        "Yelp Review dataset"
    )

    parser.add_argument(
        "-a",
        "--api",
        help="The API to use. default is posix",
        choices=[e.value for e in APIType],
        default=APIType.POSIX.value,
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Local POSIX PATH if API type is POSIX, "
        "full ufs path if REST/S3 API "
        "(e.g.s3://ref-arch/yelp-review/yelp_academic_dataset_review.json)",
        default="./data/yelp-review/yelp_academic_dataset_review.json",
    )
    parser.add_argument(
        "-d",
        "--doraroot",
        help="Alluxio REST/S3 API require Dora root ufs address to do path "
        "transformation",
        default="s3://ref-arch/",
    )
    parser.add_argument(
        "-aw",
        "--alluxioworkers",
        help="Alluxio REST/S3 API require worker hostnames in format of host1,host2,host3",
        default="localhost",
    )
    parser.add_argument(
        "-o",
        "--options",
        help="Additional Alluxio property key value pars in format of key1=value1,key2=value2",
        default="",
    )

    return parser.parse_args()


class BenchmarkLargeDatasetRunner:
    ALLUXIO_PAGE_SIZE_KEY = "alluxio.worker.page.store.page.size"

    def __init__(
        self,
        api,
        path,
        dora_root,
        alluxio_workers,
        options,
    ):
        self.api = api
        self.path = path
        self.dora_root = dora_root
        self.alluxio_workers = alluxio_workers
        self.options = options

    def benchmark_data_loading(self):
        start_time = time.perf_counter()

        if self.api == APIType.REST.value:
            _logger.debug(
                f"Using alluxio REST API reading file with workers {self.endpoints}, "
                f"dora root {self.dora_root}, "
                f"page size {self.page_size}, "
                f"ufs path {self.path}"
            )
            alluxio_rest = AlluxioRest(
                alluxio_workers=self.alluxio_workers,
                dora_root=self.dora_root,
                options=self.options,
                concurrency=1,
                logger=_logger,
            )
            alluxio_rest.read_file(self.path)
        elif self.api == APIType.POSIX.value:
            _logger.debug(
                f"Using POSIX API reading file with path {self.path}"
            )
            with open(self.path, "r") as f:
                f.read()
        else:
            _logger.debug(
                f"Using alluxio S3 API reading file with workers {self.endpoints}, "
                f"dora root {self.dora_root}, ufs path {self.path}"
            )
            alluxio_s3 = AlluxioS3(
                self.endpoints,
                self.dora_root,
                _logger,
            )
            alluxio_s3.read_file(self.path)

        end_time = time.perf_counter()
        _logger.info(f"Data loading in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    args = get_args()
    options_dict = parse_options(args.options)

    benchmark_large_dataset_runner = BenchmarkLargeDatasetRunner(
        api=args.api,
        path=args.path,
        dora_root=args.doraroot,
        alluxio_workers=args.alluxioworkers,
        options=options_dict,
    )
    benchmark_large_dataset_runner.benchmark_data_loading()
