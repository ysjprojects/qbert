import logging
from transformers import (
    BertConfig,
    BertForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from qbert.configuration_qbert import QBertConfig
from qbert.modeling_qbert import QBertForMaskedLM
import argparse
from accelerate import Accelerator, DistributedType
import time
import os
import torch
# 1. Load the off-the-shelf BERT tokenizer
tokenizer_base = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer_mini = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-256_A-4")
tokenizer_medium = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")

tokenizers = [tokenizer_base, tokenizer_mini, tokenizer_medium]

# 2. Define a BERT configuration from scratch
config_base = QBertConfig(
    vocab_size=tokenizer_base.vocab_size,    # Make sure vocab size matches the tokenizer
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    quaternion_mode="all",
)

config_mini = QBertConfig(
    vocab_size=tokenizer_mini.vocab_size,    # Make sure vocab size matches the tokenizer
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=512,
    quaternion_mode="all",
)

config_medium = QBertConfig(
    vocab_size=tokenizer_medium.vocab_size,    # Make sure vocab size matches the tokenizer
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
    quaternion_mode="all",
)

bert_config_base = BertConfig(
    vocab_size=tokenizer_base.vocab_size,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
)

bert_config_mini = BertConfig(
    vocab_size=tokenizer_mini.vocab_size,
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=1024,
    max_position_embeddings=512,
)

bert_config_medium = BertConfig(
    vocab_size=tokenizer_medium.vocab_size,
    hidden_size=512,
    num_hidden_layers=8,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=512,
)

configs = [config_base, config_mini, config_medium]
bert_configs = [bert_config_base, bert_config_mini, bert_config_medium]
def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)


if __name__ == "__main__":
    # 3. Instantiate a BERT model for masked language modeling (MLM)
    os.environ["WANDB_DISABLED"] = "true"
    
    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=int, default=2) #0: base, 1: mini, 2: medium
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--learning_rate", type=float, default=1.0e-4)
    args.add_argument("--block_size", type=int, default=128)
    args.add_argument("--is_bert", action="store_true")
    args = args.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize the process group
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    tokenizer = tokenizers[args.mode]
    if args.is_bert:
        config = bert_configs[args.mode]
    else:
        config = configs[args.mode]

    if args.is_bert:
        model = BertForMaskedLM(config)
    else:
        model = QBertForMaskedLM(config)

    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_params:,}")

    
    model.to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)


    dataset = load_dataset("Skylion007/openwebtext", split="train")



    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=["text"], num_proc=os.cpu_count()//2)

    lm_dataset = tokenized_dataset.map(group_texts, fn_kwargs={"block_size": args.block_size}, batched=True, num_proc=os.cpu_count()//2)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    output_dir = "./bert-mlm-pretrain" if args.is_bert else "./qbert-mlm-pretrain"

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=5,
        logging_steps=500,
        logging_dir="./logs_bert-mlm-pretrain" if args.is_bert else "./logs_qbert-mlm-pretrain",
        dataloader_drop_last=True,
        local_rank=local_rank,
        ddp_backend="nccl" if torch.cuda.is_available() else "gloo",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    if rank == 0:
        # Save model (only on rank 0)
        model_to_save = model.module  # Unwrap DDP
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    
    # Clean up
    dist.destroy_process_group()
