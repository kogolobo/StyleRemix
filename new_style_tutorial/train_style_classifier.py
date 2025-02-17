import os
import pprint
import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    set_seed
)
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets

LABEL_TO_ID = {
    "less": 0,
    "more": 1
}

ID_TO_LABEL = {
    0: "less",
    1: "more"
}

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large")
    parser.add_argument("--style", type=str, default="sarcasm")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    args = parser.parse_args()
    pprint.pprint(vars(args))

    set_seed(args.seed)
    dataset = load_dataset("json", data_files=f"{args.style}_classifier_examples.jsonl")['train'].remove_columns(['book_id', 'title', 'date'])
    dataset_less = dataset.map(
        lambda x: {
            "text": x[f'{args.style}_less'],
            "label": LABEL_TO_ID['less'],
        },
        remove_columns=[f'{args.style}_less', f'{args.style}_more', 'original'],
        num_proc=args.num_proc
    )
    dataset_more = dataset.map(
        lambda x: {
            "text": x[f'{args.style}_more'],
            "label": LABEL_TO_ID['more'],
        },
        remove_columns=[f'{args.style}_less', f'{args.style}_more', 'original'],
        num_proc=args.num_proc
    )
    dataset = concatenate_datasets([dataset_less, dataset_more]).shuffle(seed=args.seed)
    dataset = dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
    print(f"Train data size: {len(dataset['train'])}")
    print(f"Test data size: {len(dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding='do_not_pad', truncation=True)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(LABEL_TO_ID))
    model.to(device)

    out_path = f'./classifiers/{args.style}_classifier'
    training_args = TrainingArguments(
        run_name=f"{args.style}_classifier",
        optim='adamw_hf',
        bf16=True,
        learning_rate=args.lr,
        output_dir=out_path,
        save_strategy='epoch',
        eval_strategy='epoch',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(out_path, 'logs'),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        save_total_limit=1,
        greater_is_better=True
    )

    compute_metrics = lambda output: {'accuracy': (np.argmax(output.predictions, axis=1) == output.label_ids).mean()}
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")

if __name__ == "__main__":
    main()