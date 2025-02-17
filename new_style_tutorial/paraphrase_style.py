import os
import pprint
import tomllib
import openai
import tqdm
import pandas as pd

from dataclasses import dataclass
from functools import partial
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Tuple
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer


@dataclass
class ParaphrasePrompt:
    less: str
    more: str

class Paraphraser(ABC):
    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        self.prompts = prompts
        self.model_name = model_name
        self.style = style

    def paraphrase(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        less, more = self.paraphrase_prompts(df['original'])
        df[f'{self.style}_less'] = less
        df[f'{self.style}_more'] = more
        return df
    
    def paraphrase_prompts(self, texts: List[str]) -> Tuple[List[str], List[str]]:
        user_prompts = [f"Paragraph: {text} \n Rewrite:" for text in texts]
        less = self.make_completions(self.prompts.less, user_prompts)
        more = self.make_completions(self.prompts.more, user_prompts)
        return less, more

    @abstractmethod
    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        pass
        
class OpenAIParaphraser(Paraphraser):
    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        super().__init__(style, model_name, prompts)
        self.client = openai.Client()

    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        completions = []
        for text in tqdm.tqdm(texts, desc=f"Paraphrasing"):
            completions.append(self.make_completion(prompt, text))
        return completions

    def make_completion(self, prompt: str, text: str) -> str:
        messages = [
            {'role': 'developer', 'content': prompt},
            {'role': 'user', 'content': text}
        ]
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return completion.choices[0].message['content']
    
class LocalParaphraser(Paraphraser):
    def __init__(self, style: str, model_name: str, prompts: ParaphrasePrompt):
        super().__init__(style, model_name, prompts)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map='auto')
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id

    def make_completions(self, prompt: str, texts: List[str]) -> List[str]:
        messages = [
            [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': text}
            ]
            for text in texts
        ]
        gen_pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        generator = partial(gen_pipeline, max_new_tokens=200, num_return_sequences=1, return_full_text=False)
        completions = []
        for output in tqdm.tqdm(generator(messages, batch_size=1), desc=f"Paraphrasing"):
            completions.append(output[0]['generated_text'])
            
        return completions

def main():
    parser = ArgumentParser()
    parser.add_argument("--use_openai", action="store_true")
    parser.add_argument("--model_name", type=str, default="openai-gpt")
    parser.add_argument("--prompts", type=str, default="sarcasm.toml")
    parser.add_argument("--adapter_base_texts", type=str, default="disc_base_texts.jsonl")
    parser.add_argument("--classifier_base_texts", type=str, default="disc_for_classifiers_base_texts.jsonl")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    
    with open(args.prompts, 'rb') as f:
        prompts = tomllib.load(f)
    prompts = ParaphrasePrompt(**prompts)
    style, _ = os.path.splitext(os.path.basename(args.prompts))
    paraphraser_cls = OpenAIParaphraser if args.use_openai else LocalParaphraser
    paraphraser = paraphraser_cls(style, args.model_name, prompts)
    
    adapter_base_texts = pd.read_json(args.adapter_base_texts, lines=True)
    adapter_paraphrased = paraphraser.paraphrase(adapter_base_texts)
    adapter_paraphrased.to_json(f"{style}_adapter_examples.jsonl", lines=True, orient='records')
    
    if args.classifier_base_texts is not None:
        classifier_base_texts = pd.read_json(args.classifier_base_texts, lines=True)
        classifier_paraphrased = paraphraser.paraphrase(classifier_base_texts)
        classifier_paraphrased.to_json(f"{style}_classifier_examples.jsonl", lines=True, orient='records')

if __name__ == "__main__":
    main()
    
        