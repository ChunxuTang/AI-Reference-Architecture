import hashlib
import io
import json
import logging
import os

import humanfriendly
import requests
from alluxio import AlluxioFileSystem
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class AlluxioDataset(Dataset):
    def __init__(self, alluxio_file_system, dataset_path, transform, logger):
        self.alluxio = alluxio_file_system
        self.transform = transform
        self.logger = logger or logging.getLogger("AlluxioRestDataset")
        self.data = []

        classes = [
            item["mName"]
            for item in json.loads(self.alluxio.list_dir(dataset_path))
            if item["mType"] == "directory"
        ]

        index_to_class = {i: j for i, j in enumerate(classes)}

        self.class_to_index = {
            value: key for key, value in index_to_class.items()
        }

        for class_name in classes:
            class_path = dataset_path.rstrip("/") + "/" + class_name
            image_names = [
                item["mName"]
                for item in json.loads(self.alluxio.list_dir(class_path))
                if item["mType"] == "file"
            ]
            for image_name in image_names:
                self.data.append(
                    [
                        class_path + "/" + image_name,
                        class_name,
                    ]
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, class_name = self.data[index]
        image_content = self.alluxio.read_file(image_path)
        try:
            image = Image.open(io.BytesIO(image_content)).convert("RGB")
        except Exception as e:
            self.logger.error(
                f"Error when decoding image: {image_path}, error: {e}"
            )
            return None

        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_to_index[class_name]
        return image, class_id
