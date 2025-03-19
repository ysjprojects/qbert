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

# 1. Load the off-the-shelf BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Define a BERT configuration from scratch
config = QBertConfig(
    vocab_size=tokenizer.vocab_size,    # Make sure vocab size matches the tokenizer
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=512,
    quaternion_mode="all",
)

# 3. Instantiate a BERT model for masked language modeling (MLM)
model = QBertForMaskedLM(config)

dataset = load_dataset("Skylion007/openwebtext", split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

block_size = 128
def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./qbert-mlm-pretrain",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=10_000,
    save_total_limit=5,
    logging_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./qbert-mlm-pretrain")
tokenizer.save_pretrained("./qbert-mlm-pretrain")
