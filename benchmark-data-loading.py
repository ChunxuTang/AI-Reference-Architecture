"""
A script to benchmark data loading of the ImageNet dataset in the PyTorch
data loader. Note that it still requires some CPU data processing to convert
the images to PyTorch tensors.

Example usage:
python3 benchmark-data-loading.py -p /mnt/alluxio/fuse/imagenet-mini/train -e 5 -b 128 -w 16
"""
import argparse
import logging
import time
import warnings
from logging.config import fileConfig

import torch
import torchvision.transforms as transforms
from alluxio import AlluxioDataset
from alluxio import AlluxioRest
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)
# Explicitly disable the PIL.TiffImagePlugin logger as it also uses
# the StreamHandler which will overrun the console output.
logging.getLogger("PIL.TiffImagePlugin").disabled = True
warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch Data Loading on ImageNet Dataset"
    )

    parser.add_argument(
        "-n", "--name", help="Experiment name", default="data loading"
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path of dataset",
        default="./data/imagenet-mini/val",
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
        "--alluxio",
        help="Whether to load data from Alluxio, default will load data from the given path directly",
        action="store_true",
    )
    parser.add_argument(
        "-ap",
        "--alluxiopath",
        help="Full UFS path of the dataset in Alluxio",
        default="s3://ref-arch/imagenet-mini/val",
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


class BenchmarkRunner:
    _logger = logging.getLogger("BenchmarkRunner")

    def __init__(
        self,
        path,
        name,
        num_epochs,
        batch_size,
        num_workers,
        alluxio,
        alluxio_ufs_path,
        alluxio_workers,
        alluxio_page_size,
    ):
        self.path = path
        self.name = name
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.alluxio = alluxio
        self.alluxio_ufs_path = alluxio_ufs_path
        self.alluxio_workers = alluxio_workers
        self.alluxio_page_size = alluxio_page_size

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
        if self.alluxio:
            self._logger.debug(
                f"Using alluxio dataset with workers {self.alluxio_workers}"
            )
            alluxio_rest = AlluxioRest(
                self.alluxio_workers,
                self.alluxio_page_size,
                self.num_workers,
                self._logger,
            )
            dataset = AlluxioDataset(
                local_path=self.path,
                alluxio_ufs_path=self.alluxio_ufs_path,
                alluxio_rest=alluxio_rest,
                transform=transform,
                _logger=self._logger,
            )
        else:
            self._logger.debug("Using ImageFolder dataset")
            dataset = ImageFolder(root=self.path, transform=transform)

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
        if self.alluxio:
            self._logger.info(
                f"[Summary] experiment: {self.name} | path: {self.alluxio_ufs_path}"
            )
        else:
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
        path=args.path,
        name=args.name,
        num_epochs=args.epoch,
        batch_size=args.batch,
        num_workers=args.worker,
        alluxio=args.alluxio,
        alluxio_ufs_path=args.alluxiopath,
        alluxio_workers=args.alluxioworkers,
        alluxio_page_size=args.alluxiopagesize,
    )
    benchmark_runner.benchmark_data_loading()
