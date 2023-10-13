import os

import boto3
from botocore.client import Config


def upload(s3_client, dataset, dataset_prefix):
    # Walk through the local directory and upload each file
    for category in os.listdir(dataset):
        for category_root, _, files in os.walk(
            os.path.join(dataset, category)
        ):
            for file_name in files:
                local_file_path = os.path.join(category_root, file_name)

                # Determine the S3 object key (the path in the bucket)
                s3_object_key = os.path.relpath(local_file_path, dataset)

                # Upload the file to S3
                s3b.upload_file(
                    local_file_path, dataset_prefix + s3_object_key
                )

                print(f"Uploaded: {s3_object_key}")


if __name__ == "__main__":
    s3 = boto3.resource(
        "s3",
        endpoint_url="http://10.0.6.242:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        verify=False,
    )
    bucket_name = "ai-ref"
    s3b = s3.Bucket(bucket_name)
    upload(
        s3b,
        "/AI-Reference-Architecture/data/imagenet-mini/train",
        "imagenet-mini/train",
    )
    upload(
        s3b,
        "/AI-Reference-Architecture/data/imagenet-mini/val",
        "imagenet-mini/val",
    )
