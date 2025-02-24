from dataclasses import dataclass
import numpy as np
import textstat
import torch
import tqdm
import string

from collections import defaultdict
from functools import partial
from typing import List, Dict, Optional
from nltk import sent_tokenize, pos_tag
from abc import ABC, abstractmethod
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

class BaseEvaluator(ABC):
    @abstractmethod
    def __call__(self, texts: List[str]) -> Dict[str, float]:
        pass
    
    def cleanup(self):
        pass

@dataclass
class EvaluationRunner:
    evaluators: List[BaseEvaluator]
    
    def __init__(self, classifier_config: Dict[str, str]):
        self.evaluators = [
            LengthEvaluator(),
            FunctionWordsEvaluator(),
            GradeLevelEvaluator()
        ]
        for axis in classifier_config.keys():
            if classifier_config[axis]['label'] == '*':
                self.evaluators.append(MultiClassEvaluator(**classifier_config[axis]))
            else:
                self.evaluators.append(ClassifierEvaluator(**classifier_config[axis]))
    
    def __call__(self, texts: List[str]) -> Dict[str, float]:
        results = {}
        for evaluator in self.evaluators:
            results.update(evaluator(texts))
        return results

    def cleanup(self):
        for evaluator in self.evaluators:
            evaluator.cleanup()
        torch.cuda.empty_cache()

class LengthEvaluator(BaseEvaluator):
    def __call__(self, texts: List[str]) -> Dict[str, float]:
        sent_lengths = []
        for text in texts:
            for sent in sent_tokenize(text):
                sent_lengths.append(len(sent.split()))
        return {"length_more": np.mean(sent_lengths)}
    
class FunctionWordsEvaluator(BaseEvaluator):
    LEXICAL_WORD_TAGS = ['NNP', 'NN' 'ADJ', 'ADV', 'FW', 'N', 'NP', 'NUM', 'VB', 'VBD', 'VBG', 'VBN', 'RB'] 
    
    def __init__(self):
        self.translator = str.maketrans('', '', string.punctuation.replace("'", ""))
    
    def __call__(self, texts: List[str]) -> Dict[str, float]:
        total_function_words = 0
        total_words = 0
        for text in texts:
            words = text.translate(self.translator).split()
            pos_tags = pos_tag(words)
            function_words = [word for word, pos in pos_tags if pos not in self.LEXICAL_WORD_TAGS]
            total_function_words += len(function_words)
            total_words += len(words)
        return {"function_more": total_function_words / total_words}
    
class GradeLevelEvaluator(BaseEvaluator):
    def __call__(self, texts: List[str]) -> Dict[str, float]:
        averages = []
        for text in texts:
            fk_text = textstat.flesch_kincaid_grade(text)
            lw_text = textstat.linsear_write_formula(text)
            gf_text = textstat.gunning_fog(text)
            averages.append(np.nanmean([fk_text, lw_text, gf_text]))
        return {"grade_more": np.nanmean(averages)}
    
class ClassifierEvaluator(BaseEvaluator):
    def __init__(self, model_name_or_path: str, label: str, out_key: Optional[str] = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.generator = pipeline('text-classification', model=model, tokenizer=tokenizer, device=device)
        self.generation_kwargs = {
            'truncation': True,
            'batch_size': 8,
            'top_k': model.config.num_labels
        }
        self.label = label
        self.out_key = out_key

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        out_key = self.out_key or self.label
        pipe = partial(self.generator, **self.generation_kwargs)
        scores = []
        for result in tqdm.tqdm(pipe(texts), desc="Classifying"):
            dict_scores = list(filter(lambda x: x['label'] == self.label, result))
            assert len(dict_scores) == 1
            scores.append(dict_scores[0]['score'])
        return {out_key: np.mean(scores)}

    def cleanup(self):
        del self.generator.model
        del self.generator.tokenizer
        del self.generator
    
class MultiClassEvaluator(ClassifierEvaluator):
    def __init__(self, model_name_or_path: str, label: str, out_key: Optional[str] = None):
        super().__init__(model_name_or_path, label, out_key)
        self.generation_kwargs.update({'top_k': 1})

    def __call__(self, texts: List[str]) -> Dict[str, float]:
        scores = defaultdict(list)
        pipe = partial(self.generator, **self.generation_kwargs)
        for result in tqdm.tqdm(pipe(texts), desc="Classifying"):
            scores[result[0]['label']].append(result[0]['score'])

        all_labels = self.generator.model.config.id2label.values()
        return {label: np.mean(scores.get(label, [0])) for label in all_labels}