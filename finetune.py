import os
import time
import torch
import torch.distributed as dist
import argparse
import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    BertForSequenceClassification,
    BertConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Import QBert models
from qbert.modeling_qbert import QBertForSequenceClassification, QBertConfig

def load_pretrained_model(pretrained_model_path, num_labels=3, is_bert=False, from_scratch=False, qbert_mode="all", weighted_qk_concat=False):
    """
    Load a pretrained model and convert it to a sequence classification model.
    
    Args:
        pretrained_model_path: Path to the pretrained model
        num_labels: Number of classes for classification
        is_bert: Whether to use BERT (True) or QBert (False)
        
    Returns:
        SequenceClassification model with weights transferred from the pretrained model
    """
    print(f"Loading pretrained model from {pretrained_model_path}")

    tokenizer_medium = AutoTokenizer.from_pretrained("google/bert_uncased_L-8_H-512_A-8")

    bert_config = BertConfig(
        vocab_size=tokenizer_medium.vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        num_labels=num_labels,
    )

    qbert_config = QBertConfig(
        vocab_size=tokenizer_medium.vocab_size,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=512,
        quaternion_mode=qbert_mode,
        num_labels=num_labels,
        weighted_qk_concat=weighted_qk_concat,
    )

    
    if is_bert:
        if not from_scratch:
            # Create a new BertForSequenceClassification model
            model = BertForSequenceClassification.from_pretrained(
                pretrained_model_path,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )
        else:
            # Create a new BertForSequenceClassification model from scratch
            model = BertForSequenceClassification(config=bert_config)
    else:
        if not from_scratch:
            # Create a new QBertForSequenceClassification model
            model = QBertForSequenceClassification.from_pretrained(
                pretrained_model_path,
                num_labels=num_labels,
                problem_type="single_label_classification",
            )
        else:
            # Create a new QBertForSequenceClassification model from scratch
            model = QBertForSequenceClassification(config=qbert_config)
    
    print(f"Created sequence classification model with {num_labels} labels")
    return model

def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.
    
    Args:
        eval_pred: Tuple of predictions and labels
        
    Returns:
        Dictionary of metrics
    """
    print("Computing metrics")
    predictions, labels = eval_pred
    print(predictions)
    print(labels)
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary' if len(np.unique(labels)) <= 2 else 'macro'
    )
    accuracy = accuracy_score(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }



def prepare_dataset(dataset_name, tokenizer, max_length=128):
    """
    Load and prepare dataset for finetuning.
    """
    print(f"Loading dataset: {dataset_name}")
    
    dataset = load_dataset(dataset_name)

    sentence_key = "text"
    label_key = "label"

    def tokenize_function(examples):
        return tokenizer(
            examples[sentence_key],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != label_key],
        num_proc=os.cpu_count()//2
    )
    
    tokenized_datasets = tokenized_datasets.rename_column(label_key, "labels")
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["valid"]
    test_dataset = tokenized_datasets["test"]
    
    print(f"Dataset loaded and tokenized")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(eval_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    return train_dataset, eval_dataset, test_dataset

if __name__ == "__main__":
    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pretrained model")
    parser.add_argument("--dataset", type=str, default="sjyuxyz/financial-sentiment-analysis",
                        help="Dataset for finetuning")
    parser.add_argument("--num_labels", type=int, default=3,
                        help="Number of labels for classification")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--is_bert", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    parser.add_argument("--qbert_mode", type=str, default="all")
    parser.add_argument("--weighted_qk_concat", action="store_true")
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()
    print(f"Received model_path: '{args.model_path}'") 

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize the process group if using torchrun
    use_ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if use_ddp:
        print("Initializing DDP")
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    model = load_pretrained_model(args.model_path, args.num_labels, args.is_bert, args.from_scratch, args.qbert_mode, args.weighted_qk_concat)
    model.to(device)
    
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {all_params:,}")
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    train_dataset, eval_dataset, test_dataset = prepare_dataset(args.dataset, tokenizer, args.max_length)
    print(train_dataset.column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model_type = "bert" if args.is_bert else ("qbert-weighted" if args.weighted_qk_concat else "qbert")
    output_dir = ("test_" if args.testing else "") + f"{model_type}_sequence_classification_ft"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs if not args.testing else 1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        save_total_limit=5,
        load_best_model_at_end=True,
        #metric_for_best_model="f1",
        #greater_is_better=True,
        report_to=None,  # Disable wandb
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        label_names=["labels"],
        # DDP specific parameters
        local_rank=local_rank if use_ddp else -1,
        dataloader_drop_last=True if use_ddp else False,
        ddp_backend="nccl" if torch.cuda.is_available() and use_ddp else None,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if not args.testing else eval_dataset,
        eval_dataset=eval_dataset if not args.testing else test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    start_time = time.time()
    
    if rank == 0:
        print(f"Starting finetuning for {args.epochs} epochs")
    
    trainer.train()

    training_time = time.time() - start_time
    training_time_str = str(datetime.timedelta(seconds=int(training_time)))
    
    if rank == 0:
        all_results = {}
        
        print("Evaluating on validation set")
        eval_results = trainer.evaluate()
        print(f"Validation results: {eval_results}")
        all_results["validation"] = eval_results
        
        print("Evaluating on test set")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
        all_results["test"] = test_results
        
        print(f"Saving model to {output_dir}")
        model_to_save = model.module if use_ddp else model
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        import json
        import datetime
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        all_results["config"] = {
            "model_type": model_type,
            "dataset": args.dataset,
            "num_labels": args.num_labels,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "trainable_params": trainable_params,
            "total_params": all_params
        }
        
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"results_{timestamp}.json")
        
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        summary = {
            "model": model_type,
            "dataset": args.dataset,
            "timestamp": timestamp,
            "train_params": f"lr={args.learning_rate}, bs={args.batch_size}, epochs={args.epochs}",
            "val_accuracy": eval_results.get("eval_accuracy", 0),
            "val_f1": eval_results.get("eval_f1", 0),
            "test_accuracy": test_results.get("eval_accuracy", 0),
            "test_f1": test_results.get("eval_f1", 0),
            "training_time": training_time_str,
            "training_time_seconds": training_time
        }
        
        summary_file = os.path.join(results_dir, "summary.csv")
        summary_exists = os.path.exists(summary_file)
        
        import csv
        with open(summary_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if not summary_exists:
                writer.writeheader()
            writer.writerow(summary)
        
        print(f"Summary added to {summary_file}")
    
    if use_ddp:
        dist.destroy_process_group()
