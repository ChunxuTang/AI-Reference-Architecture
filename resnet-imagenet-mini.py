"""
A script to train a ResNet on a subset of the ImageNet dataset.

Example usage:
python3 resnet-image-net-mini.py -i ./data/imagenet-mini/train -p
"""
import argparse
import logging
import time
from logging.config import fileConfig

import torch
import torchvision
import torchvision.transforms as transforms

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)
# Explicitly disable the PIL.TiffImagePlugin logger as it also uses
# the StreamHandler which will overrun the console output.
logging.getLogger("PIL.TiffImagePlugin").disabled = True


def get_args():
    parser = argparse.ArgumentParser(description="Resnet Imagenet Mini")

    parser.add_argument(
        "-p",
        "--profile",
        help="Profiling the Resnet Imagenet training",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "-i",
        "--input_path",
        help="Input dataset path",
        default="/mnt/alluxio/fuse/imagenet-mini/train",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        help="Output model path",
        default="./resnet-imagenet-model.pth",
    )

    parser.add_argument(
        "-e", "--epoch", help="Number of epochs", default=3, type=int
    )
    parser.add_argument(
        "-b", "--batch", help="Batch size", default=128, type=int
    )
    parser.add_argument(
        "-w", "--worker", help="Number of workers", default=16, type=int
    )

    return parser.parse_args()


class ResnetTrainer:
    _logger = logging.getLogger("ResnetTrainer")

    def __init__(
        self,
        input_path="/mnt/alluxio/fuse/imagenet-mini/train",
        output_path="./resnet-imagenet-model.pth",
        profiler_log_path="./log/resnet",
        num_epochs=3,
        batch_size=128,
        num_workers=16,
        learning_rate=0.001,
        profiler_enabled=False,
    ):
        self._logger.info(f"Start time: {time.perf_counter()}")

        self.input_path = input_path
        self.output_path = output_path
        self.profiler_log_path = profiler_log_path
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.profiler_enabled = profiler_enabled

        self.train_loader = None
        self.model = None

        self.device = self._check_device()

    def run_trainer(self):
        self.train_loader = self._create_data_loader()
        self.model = self._load_model()
        self._train()
        self._save_model()

    def _create_data_loader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=45),
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=self.input_path, transform=transform
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return train_loader

    def _load_model(self):
        model = torchvision.models.resnet18(pretrained=True)
        self._logger.info(
            f"Current time after loading the model: {time.perf_counter()}"
        )

        # Parallelize training across multiple GPUs
        model = torch.nn.DataParallel(model)

        # Set the model to run on the device
        model = model.to(self.device)
        self._logger.info(
            f"Current time after loading the model to device: {time.perf_counter()}"
        )

        return model

    def _train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        start_time = time.perf_counter()

        profiler = None
        if self.profiler_enabled:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=0, warmup=0, active=1, repeat=1
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.profiler_log_path
                ),
            )
            profiler.start()

        for epoch in range(self.num_epochs):
            batch_start = time.perf_counter()
            for inputs, labels in self.train_loader:
                # Move input and label tensors to the device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                batch_end = time.perf_counter()
                self._logger.debug(
                    f"Loaded input and labels to the device in {batch_end - batch_start:0.4f} seconds"
                )

                # Zero out the optimization
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                batch_start = time.perf_counter()

            self._logger.info(
                f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss.item():.4f} at the timestamp {time.perf_counter()}"
            )

            if self.profiler_enabled:
                profiler.step()

        self._logger.info(f"Finished Training, Loss: {loss.item():.4f}")

        end_time = time.perf_counter()
        self._logger.info(
            f"Training time in {end_time - start_time:0.4f} seconds"
        )

        if self.profiler_enabled:
            profiler.stop()

    def _save_model(self):
        torch.save(self.model.state_dict(), self.output_path)
        self._logger.info(f"Saved PyTorch Model State to {self.output_path}")

    def _check_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self._logger.debug(f"Using {device}")

        return device


if __name__ == "__main__":
    args = get_args()

    resnetTrainer = ResnetTrainer(
        input_path=args.input_path,
        output_path=args.output_path,
        num_epochs=args.epoch,
        batch_size=args.batch,
        num_workers=args.worker,
        profiler_enabled=args.profile,
    )
    resnetTrainer.run_trainer()
