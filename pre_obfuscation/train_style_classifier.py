import os
import pprint
from typing import Dict, List, Tuple
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding, 
    set_seed
)
from argparse import ArgumentParser
from datasets import load_dataset, concatenate_datasets, Dataset

LABEL_TO_ID = {
    "less": 0,
    "more": 1
}

ID_TO_LABEL = {
    0: "less",
    1: "more"
}

def parse_style(style: str) -> List[str]:    
    return style.split(',')

def load_one_style_dataset(args, style: str) -> Tuple[Dataset, Dict[str, int]]:
    data_filepath = os.path.join(args.data_dir, f"{style}_classifier_examples.jsonl")
    dataset = load_dataset("json", data_files=data_filepath)['train']
    label_to_id = {f'{style}_more': 1, f'{style}_less': 0}

    dataset_less = dataset.map(
        lambda x: {
            "text": x[f'{style}_less'],
            "label": LABEL_TO_ID['less'],
        },
        remove_columns=[f'{style}_less', f'{style}_more', 'original'],
        num_proc=args.num_proc
    )
    dataset_more = dataset.map(
        lambda x: {
            "text": x[f'{style}_more'],
            "label": LABEL_TO_ID['more'],
        },
        remove_columns=[f'{style}_less', f'{style}_more', 'original'],
        num_proc=args.num_proc
    )
    dataset = concatenate_datasets([dataset_less, dataset_more]).shuffle(seed=args.seed)
    dataset = dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
    return dataset, label_to_id

def load_multi_style_dataset(args, styles: List[str]) -> Tuple[Dataset, Dict[str, int]]:
    datasets = []
    label_to_id = {f'{style}_more': idx for idx, style in enumerate(styles)}
    for style in styles:
        print(f"Loading {style} dataset")
        data_filepath = os.path.join(args.data_dir, f"{style}_classifier_examples.jsonl")
        dataset = load_dataset("json", data_files=data_filepath)['train']
        dataset = dataset.map(
            lambda x: {
                "text": x[f'{style}_more'],
                "label": label_to_id[f'{style}_more'],
            },
            remove_columns=dataset.column_names,
            num_proc=args.num_proc
        )
        datasets.append(dataset)
    dataset = concatenate_datasets(datasets).shuffle(seed=args.seed)
    dataset = dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
    return dataset, label_to_id
        
def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="FacebookAI/roberta-large")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--save_dir", type=str, default="en_classifiers/")
    parser.add_argument("--data_dir", type=str, default="en_data/")
    parser.add_argument("--style", type=parse_style, default="sarcasm")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    args = parser.parse_args()
    pprint.pprint(vars(args))

    set_seed(args.seed)
    if len(args.style) == 1:
        dataset, style_label_to_id = load_one_style_dataset(args, args.style[0])
    else:
        dataset, style_label_to_id = load_multi_style_dataset(args, args.style)
    print(f"Loaded dataset for styles: {args.style}")
    print(f"Train data size: {len(dataset['train'])}")
    print(f"Test data size: {len(dataset['test'])}")
    

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding='do_not_pad', truncation=True, max_length=args.max_seq_length)
    dataset = dataset.map(tokenize_function, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=len(style_label_to_id),
        label2id=style_label_to_id,
        id2label={id: label for label, id in style_label_to_id.items()},
        device_map='auto'
    )

    classifier_name = args.style[0] if len(args.style) == 1 else 'type'
    out_path = os.path.join(args.save_dir, 'classifiers', f'{classifier_name}_classifier')
    training_args = TrainingArguments(
        run_name=f"{classifier_name}_classifier",
        optim='adamw_hf',
        bf16=True,
        learning_rate=args.lr,
        output_dir=out_path,
        save_strategy='epoch',
        save_total_limit=1,
        eval_strategy='epoch',
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        logging_dir=os.path.join(out_path, 'logs'),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
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
    trainer.save_model(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Model saved to {out_path}")

if __name__ == "__main__":
    main()