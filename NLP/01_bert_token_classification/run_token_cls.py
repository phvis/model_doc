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

import argparse
import os
import random
import time
from functools import partial

# from datasets import load_dataset
from paddlenlp.datasets import load_dataset

import paddle
from paddle.io import DataLoader, BatchSampler

from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import AutoModelForTokenClassification, AutoTokenizer
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.data import DataCollatorForTokenClassification
from paddlenlp.utils.log import logger

parser = argparse.ArgumentParser()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument(
        "--ckp",
        default=None,
        type=str, )
    parser.add_argument(
        "--task_name",
        default="msra_ner",
        type=str,
        # choices=["msra_ner"],
        help="The named entity recognition datasets.")
    parser.add_argument(
        "--output_dir",
        default="best_msra_ner_model",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written."
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.")
    parser.add_argument(
        "--do_train", action='store_true', help="Whether to train.")
    parser.add_argument(
        "--do_eval", action='store_true', help="Whether to predict.")
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        choices=["cpu", "gpu", "xpu"],
        help="The device to select to train the model, is must be cpu/gpu/xpu.")
    args = parser.parse_args()
    return args


@paddle.no_grad()
def evaluate(model, loss_fct, metric, data_loader, label_num, mode="valid"):
    model.eval()
    metric.reset()
    avg_loss, precision, recall, f1_score = 0, 0, 0, 0
    for batch in data_loader:
        logits = model(batch['input_ids'], batch['token_type_ids'])
        loss = loss_fct(logits, batch['labels'])
        avg_loss = paddle.mean(loss)
        preds = logits.argmax(axis=2)
        num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
            batch['seq_len'], preds, batch['labels'])
        metric.update(num_infer_chunks.numpy(),
                      num_label_chunks.numpy(), num_correct_chunks.numpy())
        precision, recall, f1_score = metric.accumulate()
    print("%s: eval loss: %f, precision: %f, recall: %f, f1: %f" %
          (mode, avg_loss, precision, recall, f1_score))
    model.train()
    return f1_score


def preprocess_function(example, tokenizer, label_vocab, max_seq_length=128):

    labels = example['labels']
    tokens = example['tokens']
    no_entity_id = label_vocab['O']

    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_length)

    # 保证label与input_ids长度一致
    # -2 for [CLS] and [SEP]
    if len(tokenized_input['input_ids']) - 2 < len(labels):
        labels = labels[:len(tokenized_input['input_ids']) - 2]
    tokenized_input['labels'] = [no_entity_id] + labels + [no_entity_id]
    tokenized_input['labels'] += [no_entity_id] * (
        len(tokenized_input['input_ids']) - len(tokenized_input['labels']))

    return tokenized_input


def run(args):
    paddle.set_device(args.device)
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    train_ds, test_ds = load_dataset(args.task_name, split=["train", "test"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # label_list = train_ds.features['ner_tags'].feature.names
    # label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    label_vocab = {
        label: label_id
        for label_id, label in enumerate(train_ds.label_list)
    }
    label_list = list(label_vocab.values())

    label_num = len(label_list)

    trans_func = partial(
        preprocess_function,
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        max_seq_length=128)
    train_ds = train_ds.map(trans_func)
    test_ds = test_ds.map(trans_func)

    collate_fn = DataCollatorForTokenClassification(
        tokenizer=tokenizer, label_pad_token_id=-1)

    # Define the model netword and its loss
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path, num_classes=label_num)
    if args.ckp is not None:
        model.set_dict(paddle.load(args.ckp))

    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    test_batch_sampler = BatchSampler(
        test_ds, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(
        dataset=test_ds,
        batch_sampler=test_batch_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        return_list=True,
        shuffle=False)

    if args.do_train:
        train_batch_sampler = BatchSampler(
            train_ds, batch_size=args.batch_size, shuffle=True)
        train_data_loader = DataLoader(
            dataset=train_ds,
            batch_sampler=train_batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            return_list=True)

        num_training_steps = args.max_steps if args.max_steps > 0 else len(
            train_data_loader) * args.num_train_epochs

        lr_scheduler = LinearDecayWithWarmup(
            args.learning_rate, num_training_steps, args.warmup_steps)

        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            epsilon=args.adam_epsilon,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            apply_decay_param_fun=lambda x: x in decay_params)

        loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)

        metric = ChunkEvaluator(label_list=train_ds.label_list)

        global_step = 0
        best_f1 = 0.0
        last_step = args.num_train_epochs * len(train_data_loader)
        tic_train = time.time()
        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_data_loader):
                global_step += 1
                logits = model(batch['input_ids'], batch['token_type_ids'])
                loss = loss_fct(logits, batch['labels'])
                avg_loss = paddle.mean(loss)
                if global_step % args.logging_steps == 0:
                    print(
                        "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                        % (global_step, epoch, step, avg_loss,
                           args.logging_steps / (time.time() - tic_train)))
                    tic_train = time.time()
                avg_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()
                if global_step % args.save_steps == 0 or global_step == num_training_steps:
                    if paddle.distributed.get_rank() == 0:
                        f1 = evaluate(model, loss_fct, metric,
                                      test_data_loader, label_num, "test")
                        if f1 > best_f1:
                            best_f1 = f1
                            output_dir = args.output_dir
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            # Need better way to get inner model of DataParallel
                            model_to_save = model._layers if isinstance(
                                model, paddle.DataParallel) else model
                            model_to_save.save_pretrained(output_dir)
                            tokenizer.save_pretrained(output_dir)
                if global_step >= num_training_steps:
                    print("best_f1: ", best_f1)
                    return
        print("best_f1: ", best_f1)

    if args.do_eval:
        eval_batch_sampler = BatchSampler(
            test_ds, batch_size=args.batch_size, shuffle=False)
        eval_data_loader = DataLoader(
            dataset=test_ds,
            batch_sampler=eval_batch_sampler,
            collate_fn=collate_fn,
            num_workers=0,
            return_list=True)

        # # Define the model netword and its loss
        # model = AutoModelForTokenClassification.from_pretrained(
        #     args.model_name_or_path, num_classes=label_num)
        loss_fct = paddle.nn.loss.CrossEntropyLoss(ignore_index=-1)

        metric = ChunkEvaluator(label_list=test_ds.label_list)

        model.eval()
        metric.reset()
        for step, batch in enumerate(eval_data_loader):
            logits = model(batch["input_ids"], batch["token_type_ids"])
            loss = loss_fct(logits, batch["labels"])
            avg_loss = paddle.mean(loss)
            preds = logits.argmax(axis=2)
            num_infer_chunks, num_label_chunks, num_correct_chunks = metric.compute(
                batch["length"], preds, batch["labels"])
            metric.update(num_infer_chunks.numpy(),
                          num_label_chunks.numpy(), num_correct_chunks.numpy())
            precision, recall, f1_score = metric.accumulate()
        print("eval loss: %f, precision: %f, recall: %f, f1: %f" %
              (avg_loss, precision, recall, f1_score))


def print_arguments(args):
    """print arguments"""
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).items()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    run(args)
