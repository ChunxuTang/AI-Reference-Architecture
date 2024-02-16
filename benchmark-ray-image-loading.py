import argparse
import logging
import time

import core  # Can be replaced by alluxiofs
import fsspec
from alluxio import AlluxioFileSystem

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

_logger = logging.getLogger("BenchmarkImageLoadingRunner")


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Ray image dataset loading"
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Path of the images",
        default="/mnt/alluxio/imagenet-mini",
    )

    parser.add_argument(
        "-l",
        "--load",
        help="Whether necessary to load the dataset into Alluxio",
        action="store_true",
    )

    parser.add_argument(
        "--etcd",
        help="The etcd cluster hosts for Alluxio",
        default="localhost",
    )

    parser.add_argument(
        "--port", help="The port of the etcd cluster", default="2379"
    )

    return parser.parse_args()


class BenchmarkImageLoadingRunner:
    def __init__(self, dataset_path, etcd_hosts, etcd_port):
        self.dataset_path = dataset_path
        self.alluxio_fs = AlluxioFileSystem(
            etcd_hosts=etcd_hosts, etcd_port=etcd_port
        )
        fsspec.register_implementation(
            "alluxio", core.AlluxioFileSystem, clobber=True
        )
        options = {
            "alluxio.worker.page.store.page.size": "20MB",
            "alluxio.user.consistent.hash.virtual.node.count.per.worker": "5",
        }
        self.alluxio_fsspec = fsspec.filesystem(
            "alluxio",
            etcd_hosts=etcd_hosts,
            etcd_port=etcd_port,
            options=options,
        )

    def load_dataset(self):
        start_time = time.perf_counter()

        _logger.debug(f"Loading dataset into Alluxio from {self.dataset_path}")
        load_status = self.alluxio_fs.load(self.dataset_path)
        _logger.debug(
            f"Loading dataset into Alluxio from {self.dataset_path} completes"
        )

        end_time = time.perf_counter()
        _logger.info(
            f"Data loading into Alluxio in {end_time - start_time:0.4f} seconds"
        )
        _logger.info(f"Dataset loading status: {load_status}")

    def benchmark_data_loading(self):
        import ray

        start_time = time.perf_counter()

        path = "alluxio:" + self.dataset_path
        _logger.debug(f"Loading dataset into Ray from {path}...")
        ds = ray.data.read_images(path, filesystem=self.alluxio_fsspec)
        _logger.debug(f"Loading dataset into Ray from {path} completed")

        end_time = time.perf_counter()
        _logger.info(
            f"Data loading into Ray in {end_time - start_time:0.4f} seconds"
        )
        _logger.debug(f"Dataset schema: {ds.schema()}")


if __name__ == "__main__":
    args = get_args()

    benchmark_image_loading_runner = BenchmarkImageLoadingRunner(
        dataset_path=args.path, etcd_hosts=args.etcd, etcd_port=args.port
    )

    if args.load:
        benchmark_image_loading_runner.load_dataset()

    benchmark_image_loading_runner.benchmark_data_loading()
