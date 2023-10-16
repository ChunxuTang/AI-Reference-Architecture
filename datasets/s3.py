import io

import boto3
from PIL import Image
from torch.utils.data import Dataset


class S3ImageDataset(Dataset):
    def __init__(self, s3_client, s3_bucket, s3_prefix, transform):
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.transform = transform
        paginator = s3_client.get_paginator("list_objects_v2")
        s3_image_paths = []
        for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".JPEG"):
                    s3_image_paths.append(obj["Key"])

        classes = set()
        self.data = []

        for s3_path in s3_image_paths:
            sub_paths = s3_path.split("/")
            if len(sub_paths) >= 2:
                class_name = sub_paths[-2]
                classes.add(class_name)
                self.data.append((s3_path, class_name))

        classes = list(classes)

        self.class_to_index = {
            class_name: class_id for class_id, class_name in enumerate(classes)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s3_path, class_name = self.data[idx]
        s3_image_object = self.s3_client.get_object(
            Bucket=self.s3_bucket, Key=s3_path
        )
        image_data = s3_image_object["Body"].read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = self.transform(image)
        class_id = self.class_to_index[class_name]
        return image, class_id
