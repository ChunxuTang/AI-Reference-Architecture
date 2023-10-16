import logging
import time
from logging.config import fileConfig

import requests
from minio import Minio
from minio.error import S3Error

log_conf_path = "../conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)
_logger = logging.getLogger("MinIORead")


def main():
    # Create a client with the MinIO server playground, its access key and secret key.
    client = Minio(
        "10.0.6.145:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )
    bucket_name = "ai-ref"
    file_name = "yelp_academic_dataset_review.json"
    local_file_path = (
        "/AI-Reference-Architecture/data/yelp-review/" + file_name
    )

    # Prepare the dataset
    found = client.bucket_exists(bucket_name)
    if not found:
        client.make_bucket(bucket_name)
    else:
        _logger.info("Bucket {} already exists".format(bucket_name))

    client.fput_object(bucket_name, file_name, local_file_path)
    _logger.info(
        "{} is successfully uploaded as object {} to bucket {}.".format(
            local_file_path, file_name, bucket_name
        )
    )

    # Read the dataset
    start_time = time.perf_counter()
    response = None
    try:
        response = client.get_object(bucket_name, file_name)
        file_content = response.read()
    finally:
        if response:
            response.close()
            response.release_conn()
    end_time = time.perf_counter()
    _logger.info(f"Data loading in {end_time - start_time:0.4f} seconds")


if __name__ == "__main__":
    try:
        main()
    except S3Error as exc:
        print("error occurred.", exc)
