import hashlib
import io
import os

import humanfriendly
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from torch.utils.data import Dataset

import boto3
from botocore.exceptions import NoCredentialsError


class AlluxioS3Dataset(Dataset):
    def __init__(
        self, bucket_name, s3_path, transform, _logger
    ):
        self.bucket_name=bucket_name  #imagenet-mini
        self.transform = transform
        self._logger = _logger
        self.data = []
        self.s3 = AlluxioS3(bucket_name)
        
        list_result = self.s3.list_objects(s3_path)
        
        print(len(list_result))
        all_paths = [o.get('Key') for o in list_result]
        
        # Initialize an empty set to store unique classes and a list to store images
        classes = set()

        print(len(all_paths))
        for path in all_paths:
            sub_paths = path.split('/')
            if sub_paths[0] == s3_path.strip('/') and len(sub_paths) > 1:
                classes.add(sub_paths[1])
                if len(sub_paths) == 3 and sub_paths[2].endswith("JPEG") :
                    self.data.append(
                        [
                            '/'.join(sub_paths),
                            sub_paths[1],
                        ]
                    )
        print(len(self.data))

        # Convert the set to a list
        classes = list(classes)

        #print("Classes:", classes)
        #print("Images:", self.data)
        
        index_to_class = {i: j for i, j in enumerate(classes)}
        self.class_to_index = {
            value: key for key, value in index_to_class.items()
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, class_name = self.data[index]
        image_content = self.s3.read_file(image_path)
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
    def __init__(self, endpoint, bucket_name):
        self.workers = [item.strip() for item in endpoint.split(",")]
        self.bucket_name=bucket_name
        self.endpoint
    
    def list_objects(self, prefix):
        s3 = self.get_s3_client()
        objects = []

        response = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)

        if 'Contents' in response:
            objects.extend(response['Contents'])

        while response['IsTruncated']:
            continuation_token = response['NextContinuationToken']

            response = s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix, ContinuationToken=continuation_token)

            if 'Contents' in response:
                objects.extend(response['Contents'])

        return objects

    def read_file(self, file_key):
        try:
            data = self.get_s3_client().get_object(Bucket=self.bucket_name, Key=file_key)
            return data['Body'].read()
        except NoCredentialsError:
            self._logger.error("No AWS credentials were found.")
            return None

    def get_s3_client(self):
        return boto3.client(
            service_name='s3',
            aws_access_key_id='alluxio',  # alluxio user name
            aws_secret_access_key='SK...',  # dummy value
            endpoint_url="http://" + get_worker_address()
            #region = 'us-east-1'
        )
        
    def get_worker_address(self):
        return self.workers[0]

