# Task: `TextToTextSimilarityEvaluation` @ `rambla/text_to_text_tasks/text_to_text_similarity_eval.py`

## Goal

The goal is to understand the extent to which a component can be used to evaluate the semantic similarity of two pieces of text. A component may refer to any machine learning model (such as an LLM or an NLI model) or metric (such as ROUGE or BLEU)
that can take as input two pieces of text and output a similarity score, either a classification or regression model.

We use datasets with pairs of texts and a ground truth label. The component's response is then compared against the ground truth.

## Example usage:
```bash
python rambla/bin/run_text_to_text.py
```