import argparse
import os
import random

from alluxio import AlluxioFileSystem


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate Alluxio read_range with local file."
    )
    parser.add_argument(
        "--alluxio_file_path",
        default="s3://ai-ref-arch/yelp-review/yelp_academic_dataset_review.json",
        required=False,
        help="The Alluxio file path to read",
    )
    parser.add_argument(
        "--local_file_path",
        default="/Users/alluxio/Downloads/yelp_academic_dataset_review.json",
        required=False,
        help="The local file path to validate against",
    )
    parser.add_argument(
        "--etcd_host",
        type=str,
        default="localhost",
        required=False,
        help="The host address for etcd",
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=100,
        required=False,
        help="The total number of read range test to run",
    )
    return parser.parse_args()


def validate_read_range(
    alluxio_fs, alluxio_file_path, local_file_path, offset, length
):
    alluxio_data = alluxio_fs.read_range(alluxio_file_path, offset, length)

    with open(local_file_path, "rb") as local_file:
        local_file.seek(offset)
        local_data = local_file.read(length)

    assert (
        alluxio_data == local_data
    ), "Data mismatch between Alluxio and local file"


def main(args):
    alluxio_fs = AlluxioFileSystem(etcd_host=args.etcd_host)
    file_size = os.path.getsize(args.local_file_path)

    max_length = 13 * 1024 * 1024

    for _ in range(args.num_tests):
        offset = random.randint(0, file_size - 1)
        length = min(random.randint(1, file_size - offset), max_length)
        validate_read_range(
            alluxio_fs,
            args.alluxio_file_path,
            args.local_file_path,
            offset,
            length,
        )

    print(
        f"Data matches between Alluxio file and local source file for {args.num_tests} times"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
