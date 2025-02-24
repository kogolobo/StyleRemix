## StyleRemix

This repository contains a simplified implementation of [StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements](http://www.arxiv.org/abs/2408.15666). This code is aimed at customization and product integration of the method. The original code can be found [here](https://github.com/jfisher52/StyleRemix).

### Background
StyleRemix, is an adaptive and interpretable obfuscation method that perturbs specific, fine-grained style elements of the original input text. StyleRemix uses pre-trained Low Rank Adaptation (LoRA) modules to rewrite inputs along various stylistic axes (e.g., formality, length) while maintaining low computational costs.

In general, the StyleRemix method works in two stages:
1. **Pre-obfuscation**: prepare the data and models needed for the obfuscation
- Prompt an LLM to re-write some base texts along a desired style axis
- Train a classifier to decide the degree to which the style axis is present in a new text
- Train LoRA adapters to remix the new text either toward or away from the style axis.
  
2. **Obfuscation**: apply the models to new data
- Use the classifiers for all available style axes to evaluate an author's style.
- Select the most prominent directions for the given author.
- Combine the LoRA adapters in the opposite directions to make the author's style more average, thus making them harder to distinguish from other authors' texts.

<p align="center">
<img src="fig1_wide.png" width="80%">
</p>

### Setting up the Environment
To set up the environment to run the code, make sure to have conda installed, then run

    conda create --name obf python=3.10

Then, activate the environment

    conda activate obf

Finally, install the required packages (make sure you are in the root directory).

    pip install -r requirements.txt

Please make sure you have `gcc` and `nvcc` compilers available in your environment.

### Running Style Remix
Please run StyleRemix with the command below:
```bash
python -m obfuscation.run_styleremix 
    --config obfuscation/styleremix_config.jsonl \
    --input_file <path_to_input_file> \
    --text_key <text_column> \
    --author_key <author_column> \
    --document_key <document_column>
```
`--input_file` Accepts the path to a JSON-Lines format data, each line needs following fields:
- `--text_key` e.g., `fullText` -- a field containing text to be obfuscated
- `--author_key` e.g., `authorIDs` -- a **list** containing IDs of authors to whom this text is attributable
- `--document_key` e.g., `documentID` -- a unique identifier for the given text

### Adding a New Style Axis
**Prerequisite**: To proceed with this guide, you will need to verbally describe the elements of the desired style, in the form of a text prompt.

The general procedure for onboarding a new style to StyleRemix consists of the following steps:

1. Write clear, descriptive prompts that contain instructions and examples of the new style axis. 
    - Please put the prompts in the `<style>.<ext>` file in this directory. 
      - Replacing `<style>` with the desired style name. Please choose a style name not already used in StyleRemix.
      - Replace `<ext>` with the desired prompt extension, for parsing. Supported extensions: `toml`, `json`, `j2`.
      - Consider using `toml` and `json` files for simple prompts, and `j2` for advanced prompting with custom Jinja2 templating. Example prompts re provided for all.
    - Inside the `<style>.<ext>`, please specify keys `more` and `less`, containing the prompts to paraphrase text respectively in the direction of the desired style and away from it.
2. Run `python -m pre_obfuscation.paraphrase_style --prompts pre_obfuscation/<style>.toml`. This will paraphrase the StyleRemix base texts (present in this folder) toward and away from your desired style, using the provided prompts. Currently, only paraphrase via OpenAI API is supported.
    - This will produce two files: `<style>_classifier_examples.jsonl` and `<style>_adapter_examples.jsonl` in the `pre_obfuscation/` directory.
3. Run `python -m pre_obfuscation.train_style_classifier --style <style>`. This will train the classifier based on examples in `pre_obfuscation/<style>_classifier_examples.jsonl`.
    - Please note the path where the classifier is saved.
    - Please add the classifier path to the `obfuscation/styleremix_config.json` dictionary in this folder, in the `classifiers` section.
4. Run `python -m pre_obfuscation.train_style_adapters --style <style>`. This will train two LoRA adapters for the style: an adapter that paraphrases toward the style and away from it, repsectively.
    - Please note the path where the adapters are saved.
    - Please add the adapter paths to the `obfuscation/styleremix_config.json` dictionary in this folder, in the `adapters` section.
5. Run StyleRemix by invoking `python -m obfuscation.run_styleremix`, as per the [instructions above](#running-style-remix).
    - Please make sure to add the recently trained classifiers and adapters to the `obfuscation/styleremix_config.json`.

## Citation
If you find this repository useful, or you use it in your research, please cite the original paper:
```
@misc{fisher2024styleremixinterpretableauthorshipobfuscation,
      title={StyleRemix: Interpretable Authorship Obfuscation via Distillation and Perturbation of Style Elements}, 
      author={Jillian Fisher and Skyler Hallinan and Ximing Lu and Mitchell Gordon and Zaid Harchaoui and Yejin Choi},
      year={2024},
      eprint={2408.15666},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.15666}, 
}
```