# Task: `RephraseTask` @ `rambla.tasks.rephrase.rephrase`

## Goal
Evaluate the model robustness to rephrasing a question in shortform QA datasets.

## Overview: 
Comes in two settings: with and without context. The difference with the MCQABaseline task is that we rephrase the question field using an LLM .

## Config:
The default config can be found in `rambla.conf.task.rephrase.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a rephrase configuration that allows paraphrasing or negating the question (from `rambla.conf.rephrase`). This config in turn calls on a model config (from `rambla.conf.model`) which determines the model used to paraphrase the questions from the dataset, and a prompt template (from `rambla.conf.template`) which defines the prompt used to instruct the LLM to paraphrase the questions.
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`).

## Example usage:
```bash
python rambla/run/run_task.py task=rephrase model=openai_chat
```