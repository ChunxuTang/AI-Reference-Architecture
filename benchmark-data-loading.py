"""
A script to benchmark data loading of the ImageNet dataset in the PyTorch
data loader. Note that it still requires some CPU data processing to convert
the images to PyTorch tensors.

Example usage:
POSIX API
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a posix -p /mnt/alluxio/fuse/imagenet-mini/val
REST API
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a alluxio -p s3://ref-arch/imagenet-mini/val --etcd localhost
- Configure ETCD user,password add -o alluxio.etcd.username=alluxio,alluxio.etcd.password=alluxio
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a alluxio -p s3://ref-arch/imagenet-mini/val --alluxioworkers host1,host2
- Configure a different page size, add -o alluxio.worker.page.store.page.size=20MB
S3 API
- python3 benchmark-data-loading.py -e 5 -b 128 -w 16 -a alluxios3 -p s3://ref-arch/imagenet-mini/val -d s3://ref-arch/ --alluxioworkers localhost
"""
import argparse
import logging
import time
import warnings
from enum import Enum
from logging.config import fileConfig

import torch
import torchvision.transforms as transforms
from alluxio import AlluxioFileSystem
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from datasets.alluxio import AlluxioDataset
from datasets.alluxios3 import AlluxioS3
from datasets.alluxios3 import AlluxioS3Dataset

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)
_logger = logging.getLogger("BenchmarkDataLoading")
# Explicitly disable the PIL.TiffImagePlugin logger as it also uses
# the StreamHandler which will overrun the console output.
logging.getLogger("PIL.TiffImagePlugin").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)


class APIType(Enum):
    POSIX = "posix"
    ALLUXIO = "alluxio"
    ALLUXIOS3 = "alluxios3"


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
        help="Local POSIX PATH if API type is POSIX, full ufs path if "
        "ALLUXIO/ALLUXIOS3 API (e.g.s3://ref-arch/imagenet-mini/val)",
        default="./data/imagenet-mini/val",
    )
    parser.add_argument(
        "-d",
        "--doraroot",
        help="AlluxioS3 API require Dora root ufs address to do path transformation",
        default="s3://ref-arch/",
    )
    parser.add_argument(
        "--etcd",
        help="Alluxio API require ETCD hostname or Alluxio worker adddresses --alluxioworkers host1,host2,host3",
        default="localhost",
    )
    parser.add_argument(
        "-aw",
        "--alluxioworkers",
        help="Alluxio S3 API require worker hostnames in format of host1,host2,host3",
        default="localhost",
    )
    parser.add_argument(
        "-o",
        "--options",
        help="Additional Alluxio property key value pars in format of key1=value1,key2=value2",
        default="",
    )

    return parser.parse_args()


def parse_options(options_str):
    options_dict = {}
    if options_str:
        key_value_pairs = options_str.split(",")
        for pair in key_value_pairs:
            key, value = pair.split("=")
            options_dict[key.strip()] = value.strip()
    return options_dict


class BenchmarkRunner:
    def __init__(
        self,
        name,
        num_epochs,
        batch_size,
        num_workers,
        api,
        path,
        dora_root,
        etcd_host,
        alluxio_workers,
        options,
    ):
        self.name = name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api = api
        self.path = path
        self.dora_root = dora_root
        self.etcd_host = etcd_host
        self.alluxio_workers = alluxio_workers
        self.options = options

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
        if self.api == APIType.ALLUXIO.value:
            alluxio_file_system = None
            if self.etcd_host is None:
                _logger.debug(
                    f"Using alluxio dataset with ETCD host {self.etcd_host} and ufs path {self.path} "
                )
                alluxio_file_system = AlluxioFileSystem(
                    etcd_host=self.etcd_host,
                    options=self.options,
                    concurrency=self.num_workers,
                    logger=_logger,
                )
            else:
                _logger.debug(
                    f"Using alluxio dataset with worker address {self.alluxio_workers} and ufs path {self.path} "
                )
                alluxio_file_system = AlluxioFileSystem(
                    worker_hosts=self.alluxio_workers,
                    options=self.options,
                    concurrency=self.num_workers,
                    logger=_logger,
                )
            dataset = AlluxioDataset(
                alluxio_file_system=alluxio_file_system,
                dataset_path=self.path,
                transform=transform,
                logger=_logger,
            )
        elif self.api == APIType.POSIX.value:
            _logger.debug(
                f"Using POSIX API ImageFolder dataset with path {self.path}"
            )
            dataset = ImageFolder(root=self.path, transform=transform)
        else:
            _logger.debug("Using alluxio S3 API dataset")
            alluxio_s3 = AlluxioS3(
                self.alluxio_workers,
                self.dora_root,
                _logger,
            )
            dataset = AlluxioS3Dataset(
                alluxio_s3=alluxio_s3,
                dataset_path=self.path,
                transform=transform,
                logger=_logger,
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
            _logger.debug(
                f"Epoch {epoch}: {epoch_end - epoch_start:0.4f} seconds"
            )
        end_time = time.perf_counter()
        _logger.debug(f"Data loading in {end_time - start_time:0.4f} seconds")
        self._summarize(end_time - start_time)

    def _check_device(self):
        try:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            _logger.debug(f"Using {device}")
        except AttributeError:
            device = "cpu"
            _logger.warning(
                "Failed to access 'torch.backends.mps'. Defaulting to 'cpu'."
            )

    def _summarize(self, elapsed_time):
        _logger.info(f"[Summary] experiment: {self.name} | path: {self.path}")
        _logger.info(
            f"num_epochs: {self.num_epochs} | batch_size: {self.batch_size} | "
            f"num_workers: {self.num_workers} | time: {elapsed_time:0.4f}"
        )


if __name__ == "__main__":
    args = get_args()
    options_dict = parse_options(args.options)

    benchmark_runner = BenchmarkRunner(
        name=args.name,
        num_epochs=args.epoch,
        batch_size=args.batch,
        num_workers=args.worker,
        api=args.api,
        path=args.path,
        dora_root=args.doraroot,
        etcd_host=args.etcd,
        alluxio_workers=args.alluxioworkers,
        options=options_dict,
    )
    benchmark_runner.benchmark_data_loading()
