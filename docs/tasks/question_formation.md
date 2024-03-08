# Task: `SimilarityBasedLongFormTask` @ `rambla.tasks.longform`

## Goal
Evaluate an LLMs question generation ability.

## Overview: 
- The tasks prompts an LLM to generate a one sentence question based on context or an answer.
- The meaning of the generated questions is then compared to a ground truth example question using a textual similarity component. For more information on the textual similarity component please see 'docs/adding_new/text_to_text_component.md'
- Performance can be evaluated using classification metrics. Perfect performance is an accuracy of 1 where by all of the generated answers are considered semantically similar and equivalent to the ground truth example question by the textual similarity component. 

## Config:
The default config can be found in `rambla.conf.task.question_formation.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a dataset filterer config for filtering dataset entries (choose from `rambla.conf.utils`)
- a textual similarity component config (from `rambla.conf.task_text_to_text_component`). This config in turn calls on a model config (from `rambla.conf.model`) which determines the model used to evaluate the semantic similarity of two texts, and a prompt template (default is `rambla.conf.template.descriptive_text_to_text.yaml`) which defines the prompt used to instruct the LLM to score the texts as similar or not.
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`)
- an evaluator config (choose from `rambla.conf.evaluator`)

## Example usage:
```bash
python rambla/run/run_task.py task=question_formation model=openai_chat
```