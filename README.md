# (Unofficial) PyTorch implementation of the paper SLAM-ASR



This is an unofficial torch implementation of the paper [[2402.08846\] An Embarrassingly Simple Approach for LLM with Strong ASR Capacity (arxiv.org)](https://arxiv.org/abs/2402.08846). In this git, we showcased the adapter-only training of the SLAM-ASR model and released a toy adapter weight between hubert + tinyllama. 

## How does SLAM-ASR work

As described in the original paper, SLAM-ASR model simply adapts the hidden representations encoded by readily trained audio encoders to a LLM, suffixed by additional prompts for speech recognition task. The ASCII chart below showed a simplified compute graph.

```
               |------|
               |hubert|
               |------|
                   |
                   |
               (adapter, the only trainable part here)
                   |
                   v
LLAMA("<user>: [______], transcribe it, <assistant>") --> "The weather is good"
```

The adapter is the only weight that we need to train to get a top performance ASR model.

## Quick Start

Use `python proof_of_concept.py` to try the toy adapter weight that bridges between hubert and tinyllama. You should see an "OK-ish" performance.

## Training

The command in `script.sh` will start a 4-gpu training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes 1 --mixed_precision fp16 finetune.py \
    --dataset librispeech_asr \
    --split clean \
    --output_dir ./output/slam_asr_lr_1e-4 \
    --logging_steps 10 \
    --save_strategy epoch \
    --data_seed 42 \
    --save_total_limit 3 \
    --eval_dataset_size 10 \
    --max_eval_samples 1000 \
    --per_device_eval_batch_size 1 \
    --max_new_tokens 32 \
    --dataloader_num_workers 8 \
    --group_by_length=True \
    --logging_strategy steps \
    --remove_unused_columns False \
    --do_train \
    --warmup_ratio 0.1 \
    --lr_scheduler_type linear \
    --source_max_len 16 \
    --target_max_len 512 \
    --per_device_train_batch_size 2 \
    --max_steps 0 \
    --num_train_epochs 50 \
    --learning_rate 1e-4 \
    --adam_beta2 0.999 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --seed 0 \
    --trust_remote_code \
    --report_to tensorboard \
    --gradient_accumulation_steps 8
```

## Testing

modify `proof_of_concept.py` to suit your testing need.



## TODO

- [ ] Release a better trained weight
- [ ] Try multi-lingual

## References

This codebase is modified from [jzhang38/TinyLlama: The TinyLlama project is an open endeavor to pretrain a 1.1B Llama model on 3 trillion tokens. (github.com)](https://github.com/jzhang38/TinyLlama) for the training script. 