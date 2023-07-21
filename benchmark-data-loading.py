"""
A script to benchmark data loading of the ImageNet dataset in the PyTorch
data loader. Note that it still requires some CPU data processing to convert
the images to PyTorch tensors.

Example usage:
POSIX API
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a posix -p /mnt/alluxio/fuse/imagenet-mini/val
REST API
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a rest -p s3://ref-arch/imagenet-mini/val -d s3://ref-arch/ --endpoints localhost:28080 --pagesize 20MB
S3 API
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a s3 -p s3://ref-arch/imagenet-mini/val -d s3://ref-arch/ --endpoints localhost:29998
"""
import argparse
import logging
import time
import warnings
from enum import Enum
from logging.config import fileConfig

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from alluxio.rest import AlluxioRest
from alluxio.rest import AlluxioRestDataset
from alluxio.s3 import AlluxioS3
from alluxio.s3 import AlluxioS3Dataset

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)
# Explicitly disable the PIL.TiffImagePlugin logger as it also uses
# the StreamHandler which will overrun the console output.
logging.getLogger("PIL.TiffImagePlugin").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)


class APIType(Enum):
    POSIX = "posix"
    REST = "rest"
    S3 = "s3"


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch Data Loading on ImageNet Dataset"
    )

    parser.add_argument(
        "-n", "--name", help="Experiment name", default="data loading"
    )
    parser.add_argument(
        "-e", "--epoch", help="Number of epochs", default=5, type=int
    )
    parser.add_argument(
        "-b", "--batch", help="Batch size", default=64, type=int
    )
    parser.add_argument(
        "-w", "--worker", help="Number of workers", default=4, type=int
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
        help="Local POSIX PATH if API type is POSIX, full ufs path if REST/S3 API (e.g.s3://ref-arch/imagenet-mini/val)",
        default="./data/imagenet-mini/val",
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


class BenchmarkRunner:
    _logger = logging.getLogger("BenchmarkRunner")

    def __init__(
        self,
        name,
        num_epochs,
        batch_size,
        num_workers,
        api,
        path,
        dora_root,
        endpoints,
        page_size,
    ):
        self.name = name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api = api
        self.path = path
        self.dora_root = dora_root
        self.endpoints = endpoints
        self.page_size = page_size

    def benchmark_data_loading(self):
        self._check_device()

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        dataset = None
        if self.api == APIType.REST.value:
            self._logger.debug(
                f"Using alluxio REST API dataset with workers {self.endpoints}, page size {self.page_size}, ufs path {self.path}"
            )
            alluxio_rest = AlluxioRest(
                self.endpoints,
                self.dora_root,
                self.page_size,
                self.num_workers,
                self._logger,
            )
            dataset = AlluxioRestDataset(
                alluxio_rest=alluxio_rest,
                dataset_path=self.path,
                transform=transform,
                _logger=self._logger,
            )
        elif self.api == APIType.POSIX.value:
            self._logger.debug(
                f"Using POSIX API ImageFolder dataset with path {self.path}"
            )
            dataset = ImageFolder(root=self.path, transform=transform)
        else:
            self._logger.debug("Using alluxio S3 API dataset")
            alluxio_s3 = AlluxioS3(
                self.endpoints,
                self.dora_root,
                self._logger,
            )
            dataset = AlluxioS3Dataset(
                alluxio_s3=alluxio_s3,
                dataset_path=self.path,
                transform=transform,
                _logger=self._logger,
            )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        start_time = time.perf_counter()
        for epoch in range(self.num_epochs):
            epoch_start = time.perf_counter()
            for _, _ in loader:
                pass
            epoch_end = time.perf_counter()
            self._logger.debug(
                f"Epoch {epoch}: {epoch_end - epoch_start:0.4f} seconds"
            )
        end_time = time.perf_counter()
        self._logger.debug(
            f"Data loading in {end_time - start_time:0.4f} seconds"
        )
        self._summarize(end_time - start_time)

    def _check_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._logger.debug(f"Using {device}")

    def _summarize(self, elapsed_time):
        self._logger.info(
            f"[Summary] experiment: {self.name} | path: {self.path}"
        )
        self._logger.info(
            f"num_epochs: {self.num_epochs} | batch_size: {self.batch_size} | "
            f"num_workers: {self.num_workers} | time: {elapsed_time:0.4f}"
        )


if __name__ == "__main__":
    args = get_args()

    benchmark_runner = BenchmarkRunner(
        name=args.name,
        num_epochs=args.epoch,
        batch_size=args.batch,
        num_workers=args.worker,
        api=args.api,
        path=args.path,
        dora_root=args.doraroot,
        endpoints=args.endpoints,
        page_size=args.pagesize,
    )
    benchmark_runner.benchmark_data_loading()
