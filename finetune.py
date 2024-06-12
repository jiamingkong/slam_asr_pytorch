# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from os.path import exists, join, isdir
import sys
from typing import Dict
import numpy as np
import logging

import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    set_seed,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Dataset
import evaluate
from modeling.arguments import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    GenerationArguments,
)
from modeling.data_collator import DataCollatorForSlamASR
from modeling.asr import SLAM_ASR
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from datasets import load_from_disk

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


# 加载模型，tokenizer
def get_accelerate_model(args, checkpoint_dir):

    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    # print(f"loading base model {args.model_name_or_path}...")
    model = SLAM_ASR(
        "facebook/hubert-base-ls960",
        "TinyLlama/TinyLlama-1.1B-Chat-v0.4",
        train_mode="adapter",
    )
    # Tokenizer
    tokenizer = model.language_tokenizer

    return model, tokenizer


# 打印模型中可训练参数的数量
def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            print(f"    - [{name}] (params.size: {param.numel()}) TRAINABLE")
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || " f"all params: {all_param} || ")


# 调整分词器和嵌入的大小
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    non_special_tokens=None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(
        special_tokens_dict
    ) + tokenizer.add_tokens(non_special_tokens)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
    print(f"Resized tokenizer and embedding to {len(tokenizer)} tokens.")


# 加载本地数据集
def local_dataset(dataset_name):
    if dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


# 创建数据模块，包括训练集、验证集和预测集，以及数据整理器。
def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }
    """
    temp_dataset_file = "temp_dataset/librispeech_asr_360"

    def format_dataset(dataset):
        def map_to_array(batch):
            # Adapted to librispeech dataset
            # speech, _ = sf.read(batch["file"])
            batch["speech"] = batch["audio"]["array"]
            return batch

        print(f"dataset: {dataset}")
        dataset = dataset.map(
            map_to_array,
            num_proc=8,
            remove_columns=["file", "speaker_id", "chapter_id", "id", "audio"],
        )
        
        print(f"dataset after mapping: {dataset}")

        def check_duration(sample):
            # 音频的采样率为16kHz
            sample_rate = 16000
            # 计算音频的长度（秒）
            duration = len(sample["speech"]) / sample_rate
            # 如果音频的长度大于15秒，返回False
            if duration > 15:
                return False
            # 否则，返回True
            return True

        dataset = dataset.filter(check_duration, num_proc=10)
        dataset.save_to_disk(temp_dataset_file)

        return dataset

    if args.dataset == "librispeech_asr":
        TRAIN_TAG = "train.360"
    elif args.dataset == "hf-internal-testing/librispeech_asr_dummy":
        TRAIN_TAG = "validation"
    else:
        TRAIN_TAG = "train"
    # Load dataset.

    from datasets import DatasetDict

    if os.path.exists(temp_dataset_file):
        print("load directly")
        dataset = load_dataset(temp_dataset_file)
    else:

        dataset = load_dataset(args.dataset, args.split, trust_remote_code=True)
        dataset = DatasetDict(
            {
                "train": dataset[TRAIN_TAG],
                "validation": dataset["validation"],
                "test": dataset["test"],
            }
        )
        dataset = format_dataset(dataset)
    # rename TRAIN_TAG to "train"
    # dataset["train"] = dataset.pop(TRAIN_TAG)

    # print(
    #     f"Splitting train dataset in train and validation according to `eval_dataset_size = {args.eval_dataset_size}`"
    # )
    # dataset = dataset["train"].train_test_split(
    #     test_size=args.eval_dataset_size, shuffle=True, seed=42
    # )

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if "validation" in dataset:
            eval_dataset = dataset["validation"]
        else:
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_bylength = "temp_dataset/eval_bylength"
            if os.path.exists(eval_bylength):
                eval_dataset = load_from_disk(eval_bylength)
            else:
                eval_dataset = eval_dataset.map(
                    lambda x: {"length": len(x["text"])}, num_proc=8
                )
                eval_dataset.save_to_disk(eval_bylength)
    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_bylength = "temp_dataset/train_bylength"
            if os.path.exists(train_bylength):
                train_dataset = load_from_disk(train_bylength)
            else:
                train_dataset = train_dataset.map(
                    lambda x: {"length": len(x["text"])}, num_proc=8
                )
                train_dataset.save_to_disk(train_bylength)

    data_collator = DataCollatorForSlamASR(
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


# 获取最新的检查点。它首先检查给定的目录是否存在，然后在该目录中查找最新的检查点。
def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():

    # 1. 解析命令行参数，这些参数定义了模型参数、数据参数、训练参数和生成参数。
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        generation_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    # 2. 如果在输出目录中找到了先前的训练检查点，那么就从该检查点恢复训练。
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    # 3. 加载模型和分词器。
    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print("loaded model")
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # 4. 加载训练器
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(
            test_dataset=data_module["predict_dataset"], metric_key_prefix="predict"
        )
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as fout:
            for i, example in enumerate(data_module["predict_dataset"]):
                example["prediction_with_input"] = predictions[i].strip()
                example["prediction"] = (
                    predictions[i].replace(example["input"], "").strip()
                )
                fout.write(json.dumps(example) + "\n")
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
