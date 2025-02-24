import math
import os
import pprint
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tokenizers.processors import TemplateProcessing
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from argparse import ArgumentParser
from datasets import load_dataset
from peft import LoraConfig, TaskType

def train_adapter(dataset, direction, args):
    dataset = dataset.map(
        lambda x: {
            "rewrite": x[f'{args.style}_{direction}'],
        },
        remove_columns=[f'{args.style}_less', f'{args.style}_more'],
        num_proc=args.num_proc
    )
    dataset = dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
    print(f"Train data size: {len(dataset['train'])}")
    print(f"Test data size: {len(dataset['test'])}")

    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        bias = "none"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=f"{bos}:0 $A:0 {eos}:0",
        pair=f"{bos}:0 $A:0 {eos}:0 {bos}:1 $B:1 {eos}:1",
        special_tokens=[
            (f"{bos}", tokenizer.bos_token_id), 
            (f"{eos}", tokenizer.eos_token_id)
        ],
    )
    # Check this is working - the last token should be eos
    assert tokenizer("Hey what's up with you I'm gonna go").input_ids[-1] == tokenizer.eos_token_id
    assert tokenizer("Hey what's up with you I'm gonna go", max_length=5, truncation=True).input_ids[-1] == tokenizer.eos_token_id

    if not tokenizer.pad_token: # Set pad token if it doesn't exist
        tokenizer.add_special_tokens({'pad_token': '<padding_token>'})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    out_path = os.path.join(args.save_dir, 'adapters', f'{args.style}_{direction}')
    steps = len(dataset['train'])/(args.train_batch_size*torch.cuda.device_count())
    save_steps = math.ceil(steps / args.save_ratio)
    training_args = SFTConfig(
        run_name=f"{args.style}_{direction}_adapter",
        bf16=True,
        learning_rate=args.lr,
        output_dir=out_path,
        logging_dir=os.path.join(out_path, 'logs'),
        evaluation_strategy='steps',
        eval_steps=save_steps,
        save_steps=save_steps,
        logging_steps=save_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        load_best_model_at_end=True,
        save_total_limit=1,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        packing=False,
        max_seq_length=args.max_seq_length
    )

    response_template = " ### Rewrite:"
    formatting_prompts_func = lambda example: f"### Original: {example['original']}\n{response_template} {example['rewrite']}"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)    
    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        formatting_func=formatting_prompts_func,
        peft_config=peft_config,
        tokenizer=tokenizer,
        data_collator=collator
    )

    trainer.train()
    trainer.model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"Adapter saved to {out_path}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--save_dir", type=str, default="pre_obfuscation")
    parser.add_argument("--style", type=str, default="sarcasm")
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.01)
    
    parser.add_argument("--train_batch_size", type=int, default=6)
    parser.add_argument("--eval_batch_size", type=int, default=6)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--save_ratio", type=int, default=4)
    args = parser.parse_args()
    pprint.pprint(vars(args))

    set_seed(args.seed)
    dataset = load_dataset("json", data_files=f"pre_obfuscation/{args.style}_classifier_examples.jsonl")['train'].remove_columns(['book_id', 'title', 'date'])
    
    for direction in ["less", "more"]:
        train_adapter(dataset, direction, args)    

if __name__ == "__main__":
    main()