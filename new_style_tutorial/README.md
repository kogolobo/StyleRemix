## Adding a new Style Axis to StyleRemix

Prerequisites: To proceed with this guide, you will need to verbally describe the elements of the desired style.

### Procedure
The general procedure for onboarding a new style to StyleRemix consists of the following steps:

1. Write clear, descriptive prompts that contain instructions and examples of the new style axis. 
    - Please put the prompts in the `<style>.toml` file in this directory, replacing `<style>` with the desired style name. Please choose a style name not already used in StyleRemix.
    - Inside the `<style>.toml`, please specify keys `more` and `less`, containing the prompts to paraphrase text respectively in the direction of the desired style and away from it.
2. Run `python paraphrase_style.py --prompts <style>.toml`. This will paraphrase the StyleRemix base texts (present in this folder) toward and away from your desired style, using the provided prompts. Currently, only paraphrase via OpenAI API is supported.
    - This will produce two files: `<style>_classifier_examples.jsonl` and `<style>_adapter_examples.jsonl`
3. Run `python train_style_classifier.py --style <style>`. This will train the classifier based on examples in `<style>_classifier_examples.jsonl`.
    - Please note the path where the classifier is saved.
    - Please add the classifier path to the `config.jsonl` dictionary in this folder, in the `classifiers` section.
4. Run `python train_style_adapters.py --style <style>`. This will train two LoRA adapters for the style: an adapter that paraphrases toward the style and away from it, repsectively.
    - Please note the path where the adapters are saved.
    - Please add the adapter paths to the `config.jsonl` dictionary in this folder, in the `adapters` section.
5. Run `python quickstart.py --config config.jsonl --directions "{'<style>': weight}"` to run StyleRemix with the nex axis.
    - The `directions` argument will be parsed as dictionary, you may specify multiple remix directions. For example: `"{'length': 0.7, 'sarcasm': -0.7, 'formality': 1.5}"` will increase length and formality but decrease sarcasm of the text.
    - Please make sure that all specified directions are present in the `config.jsonl`.
