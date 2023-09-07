from setuptools import find_packages
from setuptools import setup

setup(
<<<<<<< HEAD
    name="alluxio-python-library",
    version="0.1",
    packages=find_packages(),
||||||| e2a2cec
    name='alluxio-python-library',
    version='0.1',
    packages=find_packages(),
=======
    name="alluxio-python-library",
    version="0.1",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    zip_safe=False,
>>>>>>> 916b538fe8fab40a3e55094b839a80a6019fbf3b
    install_requires=[
<<<<<<< HEAD
        "humanfriendly",
        "requests",
        "etcd3",
        "mmh3",
        "sortedcontainers",
    ],
||||||| e2a2cec
        'humanfriendly',
        'requests',
        'etcd3',
        'mmh3',
        'sortedcontainers',
    ]
=======
        "humanfriendly",
        "requests",
        "etcd3",
        "mmh3",
        "sortedcontainers",
    ],
    extras_require={"tests": ["pytest"]},
    python_requires=">=3.8",
>>>>>>> 916b538fe8fab40a3e55094b839a80a6019fbf3b
)
