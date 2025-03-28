#!/bin/bash
torchrun --nproc_per_node=4 finetune.py --batch_size 256 --model_path ./qbert-mlm-pretrain/checkpoint-23475 --from_scratch
wait
torchrun --nproc_per_node=4 finetune.py --batch_size 256 --model_path ./qbert-mlm-pretrain/checkpoint-23475 --from_scratch --weighted_qk_concat
wait
torchrun --nproc_per_node=4 finetune.py --batch_size 256 --is_bert --model_path ./bert-mlm-pretrain/checkpoint-23475 --from_scratch