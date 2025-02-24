from argparse import ArgumentParser
from functools import partial
import pprint
import pandas as pd
import json
from tqdm.auto import tqdm

from obfuscation.evaluation import EvaluationRunner
from obfuscation.direction_selection import choose_directions_genre_mean, combine_directions
from obfuscation.remix import RemixRunner

def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", type=str, default="data.jsonl")
    parser.add_argument("--config_file", type=str, default="obfuscation/styleremix_config.json")
    parser.add_argument("--text_key", type=str, default="fullText")
    parser.add_argument("--author_key", type=str, default="authorIDs")
    parser.add_argument("--document_key", type=str, default="documentID")
    parser.add_argument("--output_key", type=str, default="remixedText")
    parser.add_argument("--output_file", type=str, default="output.jsonl")
    parser.add_argument("--top_n_styles_to_change", type=int, default=3)
    args = parser.parse_args()
    pprint.pprint(vars(args))
    
    data = pd.read_json(args.input_file, lines=True).drop(columns=[args.output_key], errors='ignore')
    author_data = data.explode(args.author_key, ignore_index=True)
    author_texts = author_data.groupby(args.author_key)[args.text_key].apply(list)
    with open(args.config_file, 'r') as f:
        styleremix_config = json.load(f)
    
    evaluation_runner = EvaluationRunner(styleremix_config['classifiers'])
    all_author_scores = {}
    for author_id, texts in author_texts.items():
        author_scores = evaluation_runner(texts)
        all_author_scores[author_id] = author_scores
    evaluation_runner.cleanup()
    
    author_scores = pd.DataFrame.from_dict(all_author_scores, orient='index')
    normalized_scores = (author_scores / author_scores.max(axis=0)).fillna(0)
    score_mean = normalized_scores.mean(axis=0)
    score_std = normalized_scores.std(axis=0)
    z_scores = ((normalized_scores - score_mean) / score_std).fillna(0)
    
    types = ['type_persuasive', 'type_narrative', 'type_expository', 'type_descriptive']
    available_types = [type_name for type_name in types if score_std[type_name] > 0]
    
    author_directions = z_scores.apply(
        lambda row: choose_directions_genre_mean(row, args.top_n_styles_to_change, available_types, types), axis=1
    )
    author_directions.name = 'StyleRemix_directions'
    author_data = author_data.merge(author_directions, left_on=args.author_key, right_index=True)
    combine_directions_partial = partial(combine_directions, all_types=types)
    agg_rules = {
        args.author_key: list,
        **{col : 'first' for col in author_data.columns if col not in [args.document_key, args.author_key, author_directions.name]},
        author_directions.name: combine_directions_partial
    }
    author_data = author_data.groupby(args.document_key).agg(agg_rules).reset_index()
    
    remix_runner = RemixRunner(styleremix_config['adapters'])
    tqdm.pandas(desc="Processing texts", total=len(author_data))
    author_data[args.output_key] = author_data.progress_apply(
        lambda row: remix_runner.remix(row[args.text_key], row['StyleRemix_directions']), axis=1
    )
    author_data.to_json(args.output_file, lines=True, orient='records')
    
    
    
if __name__ == '__main__':
    main()
