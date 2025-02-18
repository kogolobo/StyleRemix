## Adding a new Style Axis to StyleRemix

Prerequisites: To proceed with this guide, you will need to verbally describe the elements of the desired style.

### Procedure
The general procedure for onboarding a new style to StyleRemix consists of the following steps:

1. Write clear, descriptive prompts that contain instructions and examples of the new style axis. 
    - Please put the prompts in the `<style>.toml` file in this directory, replacing `<style>` with the desired style name. Please choose a style name not already used in StyleRemix.
    - Inside the `<style>.toml`, please specify keys `more` and `less`, containing the prompts to paraphrase text respectively in the direction of the desired style and away from it.
2. Run `python -m pre_obfuscation.paraphrase_style --prompts pre_obfuscation/<style>.toml`. This will paraphrase the StyleRemix base texts (present in this folder) toward and away from your desired style, using the provided prompts. Currently, only paraphrase via OpenAI API is supported.
    - This will produce two files: `<style>_classifier_examples.jsonl` and `<style>_adapter_examples.jsonl` in the `pre_obfuscation/` directory.
3. Run `python -m pre_obfuscation.train_style_classifier --style <style>`. This will train the classifier based on examples in `pre_obfuscation/<style>_classifier_examples.jsonl`.
    - Please note the path where the classifier is saved.
    - Please add the classifier path to the `obfuscation/styleremix_config.json` dictionary in this folder, in the `classifiers` section.
4. Run `python -m pre_obfuscation.train_style_adapters --style <style>`. This will train two LoRA adapters for the style: an adapter that paraphrases toward the style and away from it, repsectively.
    - Please note the path where the adapters are saved.
    - Please add the adapter paths to the `obfuscation/styleremix_config.json` dictionary in this folder, in the `adapters` section.
5. Run `python -m obfuscation.run_styleremix --config obfuscation/styleremix_config.jsonl --input_file <path_to_input_file> --text_key <text_column> --author_key <author_column> --document_key <document_column>` to run StyleRemix with the new axis.
    - Please make sure that all specified directions are present in the `obfuscation/styleremix_config.json`.
    - `--input_file` Accepts the path to a JSON-Lines format data, each line needs following fields:
        - `--text_key` e.g., `fullText` -- a field containing text to be obfuscated
        - `--author_key` e.g., `authorIDs` -- a **list** containing IDs of authors to whom this text is attributable
        - `--document_key` e.g., `documentID` -- a unique identifier for the given text
