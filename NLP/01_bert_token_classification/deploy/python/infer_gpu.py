# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import argparse
from ernie_predictor import ErniePredictor


def parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--task_name",
        default='seq_cls',
        type=str,
        help="The name of the task to perform predict, selected in: seq_cls and token_cls"
    )
    parser.add_argument(
        "--model_name_or_path",
        default="ernie-3.0-medium-zh",
        type=str,
        help="The directory or name of model.", )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.", )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for predict.", )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--set_dynamic_shape",
        action='store_true',
        help="Whether to automatically set dynamic shape.", )
    parser.add_argument(
        "--shape_info_file",
        default="shape_info.txt",
        type=str,
        help="The collected dynamic shape info file.", )
    parser.add_argument(
        "--precision_mode",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8"],
        help="Inference precision.", )
    parser.add_argument(
        "--text",
        default="北京欢迎你",
        type=str,
        help="text for token cls", )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.task_name = args.task_name.lower()
    args.device = 'gpu'
    predictor = ErniePredictor(args)

    outputs = predictor.predict([args.text])


if __name__ == "__main__":
    main()
