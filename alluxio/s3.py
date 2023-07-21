import hashlib
import io
import os

import boto3
import humanfriendly
import requests
from botocore.exceptions import NoCredentialsError
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset


class AlluxioS3Dataset(Dataset):
    def __init__(self, alluxio_s3, dataset_path, transform, _logger):
        self.alluxio_s3 = alluxio_s3
        self.transform = transform
        self._logger = _logger
        self.data = []

        list_result = self.alluxio_s3.list_objects(dataset_path)

        all_paths = [path_info.get("Key") for path_info in list_result]
        # Initialize an empty set to store unique classes and a list to store images
        classes = set()

        for path in all_paths:
            sub_paths = path.split("/")
            if len(sub_paths) > 1:
                classes.add(sub_paths[1])
                if len(sub_paths) == 3 and sub_paths[2].endswith("JPEG"):
                    self.data.append(
                        [
                            dataset_path.rstrip("/")
                            + "/"
                            + "/".join(sub_paths[1:]),
                            sub_paths[1],
                        ]
                    )

        # Convert the set to a list
        classes = list(classes)

        index_to_class = {i: j for i, j in enumerate(classes)}
        self.class_to_index = {
            value: key for key, value in index_to_class.items()
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, class_name = self.data[index]
        image_content = self.alluxio_s3.read_file(image_path)
        try:
            image = Image.open(io.BytesIO(image_content)).convert("RGB")
        except Exception as e:
            self._logger.error(
                f"Error when decoding image: {image_path}, error: {e}"
            )
            return None

        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_to_index[class_name]
        return image, class_id


class AlluxioS3:
    def __init__(self, endpoints, dora_root, _logger):
        self.workers = [item.strip() for item in endpoints.split(",")]
        self.dora_root = dora_root
        self._logger = _logger

    def list_objects(self, full_path):
        objects = []
        s3 = self.get_s3_client()
        bucket, path = self.get_bucket_path(full_path)

        response = s3.list_objects_v2(Bucket=bucket, Prefix=path)

        if "Contents" in response:
            objects.extend(response["Contents"])

        while response["IsTruncated"]:
            continuation_token = response["NextContinuationToken"]

            response = s3.list_objects_v2(
                Bucket=bucket,
                Prefix=path,
                ContinuationToken=continuation_token,
            )

            if "Contents" in response:
                objects.extend(response["Contents"])

        return objects

    def read_file(self, full_path):
        # TODO can S3 client be shared
        bucket, path = self.get_bucket_path(full_path)
        try:
            data = self.get_s3_client().get_object(Bucket=bucket, Key=path)
            return data["Body"].read()
        except NoCredentialsError:
            self._logger.error("No AWS credentials were found.")
            return None

    # Alluxio S3 API views top-level dir under dora root as bucket
    # and remaining path as path
    # e.g. path: s3://ref-arch/imagenet-mini/val with dora root: s3://ref-arch
    # here the bucket is imagenet-mini and value is val
    def get_bucket_path(self, full_path):
        alluxio_path = self.subtract_path(full_path, self.dora_root)
        parts = alluxio_path.split("/", 1)
        # TODO(lu) test the combinations
        if len(parts) == 0:
            self._logger.error(
                "Alluxio S3 API can only execute under a directory under dora root. This directory will be used as S3 bucket name"
            )
            return None
        elif len(parts) == 1:
            return parts[0], None
        else:
            return parts[0], parts[1]

    def get_s3_client(self):
        return boto3.client(
            service_name="s3",
            aws_access_key_id="alluxio",  # alluxio user name
            aws_secret_access_key="SK...",  # dummy value
            endpoint_url="http://" + self.get_worker_address()
            # region = 'us-east-1'
        )

    def get_worker_address(self):
        return self.workers[0]

    def subtract_path(self, path, parent_path):
        if "://" in path and "://" in parent_path:
            # Remove the parent_path from path
            relative_path = path[len(parent_path) :]
        else:
            # Get the relative path for local paths
            relative_path = os.path.relpath(path, start=parent_path)
        return relative_path
