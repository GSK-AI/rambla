# Task: `SimilarityBasedLongFormTask` @ `rambla.tasks.longform`

## Goal
Evaluate an LLMs ability to answer questions with long form text.

## Overview: 
- The task prompts the LLM under evaluation to answer a question with context in one sentence.
- The meaning of the long form answers is then compared to a ground truth example answer (from the dataset) using a textual similarity component.
- Accuracy measures the proportion of generated answers semantically equivalent to the ground truth as measured by the textual similarity component.

## Config:
The default config can be found in `rambla.conf.task.similarity_long_form_qa_bioasq.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a dataset filterer config for filtering dataset entries (choose from `rambla.conf.utils`)
- a textual similarity component config (from `rambla.conf.task_text_to_text_component`). This config in turn calls on a model config (from `rambla.conf.model`) which determines the model used to evaluate the semantic similarity of two texts, and a prompt template (default is `rambla.conf.template.descriptive_text_to_text.yaml`) which defines the prompt used to instruct the LLM to score the texts as similar or not.
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`)
- an evaluator config (choose from `rambla.conf.evaluator`)

## Example usage:
```bash
python rambla/run/run_task.py task=similarity_long_form_qa_bioasq model=openai_chat
```