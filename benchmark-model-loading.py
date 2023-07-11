"""
A script to benchmark model loading of an ML model, typically an LLM, via HuggingFace
transformers. The default model is Google's flan-t5-xl, which is ~11GB in storage
with 3B parameters.

Example usage:
python3 benchmark-model-loadingpy -mp ./models/flan-t5-xl -tp ./models/flan-t5-xl
"""

import argparse
import logging
import time
from logging.config import fileConfig

from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer

log_conf_path = "./conf/logging.conf"
fileConfig(log_conf_path, disable_existing_loggers=True)


def get_args():
    parser = argparse.ArgumentParser(
        description="Benchmark model loading with HuggingFace transformers"
    )

    parser.add_argument(
        "-mp",
        "--model_path",
        help="Path of input model",
        default="/mnt/alluxio/fuse/models/flan-t5-xl",
    )

    parser.add_argument(
        "-tp",
        "--tokenizer_path",
        help="Path of tokenizer model",
        default="/mnt/alluxio/fuse/models/flan-t5-xl",
    )

    return parser.parse_args()


class BenchmarkModelLoadingRunner:
    _logger = logging.getLogger("BenchmarkModelLoadingRunner")

    def __init__(self, model_path, tokenizer_path):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.tokenizer = None
        self.model = None

    def benchmark_model_loading(self):
        start_time = time.perf_counter()

        self._logger.debug(f"Loading tokenizer from {self.tokenizer_path}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.tokenizer_path)
        self._logger.debug(f"Loading model from {self.model_path}...")
        self.model = T5ForConditionalGeneration.from_pretrained(
            self.model_path, device_map="auto"
        )

        end_time = time.perf_counter()
        self._logger.info(
            f"Model loading in {end_time - start_time:0.4f} seconds"
        )

    def test_model_inference(
        self,
        input_text="Q: Can Geoffrey Hinton have a conversation with George Washington? "
        + "Give the rationale before answering.",
    ):
        input_ids = self.tokenizer(
            input_text, return_tensors="pt"
        ).input_ids.to("cuda")
        outputs = self.model.generate(input_ids, max_length=100)

        self._logger.info("Test model inference with the following question:")
        self._logger.info(input_text)
        self._logger.info(self.tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    args = get_args()

    benchmark_model_loading_runner = BenchmarkModelLoadingRunner(
        tokenizer_path=args.tokenizer_path, model_path=args.model_path
    )

    benchmark_model_loading_runner.benchmark_model_loading()
    benchmark_model_loading_runner.test_model_inference()
