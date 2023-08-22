from setuptools import setup, find_packages

setup(
    name='alluxio-python-library',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'humanfriendly',
        'requests',
        'etcd3',
        'mmh3',
        'sortedcontainers',
    ]
)
