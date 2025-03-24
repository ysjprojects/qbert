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


from qbert.configuration_qbert import QBertConfig
from qbert.modeling_qbert import QBertForMaskedLM
import argparse
from accelerate import Accelerator, DistributedType
import time

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

configs = [config_base, config_mini, config_medium]

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

    args = argparse.ArgumentParser()
    args.add_argument("--mode", type=int, default=2) #0: base, 1: mini, 2: medium
    args.add_argument("--epochs", type=int, default=3)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--learning_rate", type=float, default=1.0e-4)
    args.add_argument("--block_size", type=int, default=128)
    args = args.parse_args()

    tokenizer = tokenizers[args.mode]
    config = configs[args.mode]
    

    model = QBertForMaskedLM(config)

    dataset = load_dataset("Skylion007/openwebtext", split="train")



    tokenized_dataset = dataset.map(tokenize_function, fn_kwargs={"tokenizer": tokenizer}, batched=True, remove_columns=["text"])


    lm_dataset = tokenized_dataset.map(group_texts, fn_kwargs={"block_size": args.block_size}, batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir="./qbert-mlm-pretrain",
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=10_000,
        save_total_limit=5,
        logging_steps=500,
        strategy="ddp",
    )

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
    )

    if accelerator.is_main_process:
        start_time = time.time()

    trainer.train()

    if accelerator.is_main_process:
        end_time = time.time()

        total_training_time = end_time - start_time
        print(f"Total training time: {total_training_time} seconds")

        # Estimate time per epoch.
        epoch_time = total_training_time / args.epochs
        print(f"Estimated time per epoch: {epoch_time} seconds")

        # Estimate total time for all epochs
        total_estimated_time = epoch_time * args.epochs
        print(f"Total estimated time for {args.epochs} epochs: {total_estimated_time} seconds")

    model.save_pretrained("./qbert-mlm-pretrain")
    tokenizer.save_pretrained("./qbert-mlm-pretrain")
