# Alluxio Python Library

## Installation

Install from source
```
cd alluxio-python-library
python setup.py sdist bdist_wheel
pip install dist/alluxio_python_library-0.1-py3-none-any.whl
```

## Usage

```
from alluxio import AlluxioFileSystem

alluxio_file_system = AlluxioFileSystem(
            etcd_host=self.etcd_host, # Connect to ETCD to get Alluxio worker info
            dora_root=self.dora_root, # Transform ufs path to dora path
            options=self.options, # Alluxio property key value pars in format of key1=value1,key2=value2
            concurrency=self.num_workers, # Concurrent requests allowed to Alluxio filesystem
            logger=_logger,
            )

alluxio.list_dir(full_ufs_dataset_path)
alluxio.read_file(full_ufs_file_path)
# See datasets/alluxio.py AlluxioDataset for more example usage
```
