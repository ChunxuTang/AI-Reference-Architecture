"""
A script to benchmark data loading of large datasets to compare Alluxio + FUSE
and Alluxio REST APIs.

Example usage:
1. To run the script to test a FUSE dataset:
python3 benchmark-large-datasets.py -p /mnt/alluxio/fuse/yelp-review/yelp_academic_dataset_review.json
2. To run the script to test Alluxio GET page API
python3 benchmark-large-datasets.py -a -aw localhost:28080
"""
import argparse
import logging
import time
from logging.config import fileConfig

from alluxio import AlluxioRest

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark loading Large datasets like the Yelp Review dataset"
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path of dataset",
        default="./data/yelp-review/yelp_academic_dataset_review.json",
    )
    parser.add_argument(
        "-a",
        "--alluxio",
        help="Whether to load data from Alluxio, default will load data from the given path directly",
        action="store_true",
    )
    parser.add_argument(
        "-ap",
        "--alluxiopath",
        help="Full UFS path of the dataset in Alluxio",
        default="s3://ref-arch/yelp-review/yelp_academic_dataset_review.json",
    )
    parser.add_argument(
        "-aw",
        "--alluxioworkers",
        help="Alluxio worker addresses in list of host:port,host2:port2 format",
        default="localhost:28080",
    )
    parser.add_argument(
        "-ps",
        "--alluxiopagesize",
        help="Alluxio page size (e.g. 1MB, 4MB, 1024KB)",
        default="1MB",
    )

    return parser.parse_args()


class BenchmarkLargeDatasetRunner:
    _logger = logging.getLogger("BenchmarkLargeDatasetRunner")

    def __init__(
        self,
        path,
        alluxio,
        alluxio_ufs_path,
        alluxio_workers,
        alluxio_page_size,
    ):
        self.path = path
        self.alluxio = alluxio
        self.alluxio_ufs_path = alluxio_ufs_path
        self.alluxio_workers = alluxio_workers
        self.alluxio_page_size = alluxio_page_size

    def benchmark_data_loading(self):
        start_time = time.perf_counter()

        if self.alluxio:
            self._logger.debug(
                f"Using alluxio dataset with workers {self.alluxio_workers}"
            )
            self._logger.info(f"Loading dataset from {self.alluxio_ufs_path}")
            alluxio_rest = AlluxioRest(
                self.alluxio_workers,
                self.alluxio_page_size,
                1,  # Only using one thread
                self._logger,
            )
            alluxio_rest.read_whole_file(self.alluxio_ufs_path)
        else:
            self._logger.debug("Using alluxio FUSE/local dataset")
            self._logger.info(f"Loading dataset from {self.path}")
            with open(self.path, "r") as f:
                f.read()

        end_time = time.perf_counter()
        self._logger.info(f"Data loading in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    args = get_args()

    benchmark_large_dataset_runner = BenchmarkLargeDatasetRunner(
        path=args.path,
        alluxio=args.alluxio,
        alluxio_ufs_path=args.alluxiopath,
        alluxio_workers=args.alluxioworkers,
        alluxio_page_size=args.alluxiopagesize,
    )
    benchmark_large_dataset_runner.benchmark_data_loading()
