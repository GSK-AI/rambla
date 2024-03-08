# Task: `NegationTask` @ `rambla.tasks.negation.negation`

## Goal
Evaluate the model's ability to answer based on context _only_. This is quantified by negating the provided context, and testing whether the model changes its response accordingly.

## Overview: 
- A negation-model (eg ChatGPT) is prompted to negate the meaning of each “long_answer” field for yes/no instances of PubMedQA-labeled such that the answer to the “question” field given the new “long_answer” is the opposite to the original “final_decision” field.

- The LLM under evaluation is prompted to answer the original PubMedQA “question” using the negation-model generated "long answers" as context.

- The responses from the LLM under evaluation are reversed (“yes” → “no”/”no” → “yes”) and compared to the ground-truth to generate evaluation metrics. 

## Config:
The default config can be found in `rambla.conf.task.negation.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a rephrase configuration that allows paraphrasing or negating the question (from `rambla.conf.rephrase`). This config in turn calls on a model config (from `rambla.conf.model`) which determines the model used to paraphrase the questions from the dataset, and a prompt template (from `rambla.conf.template`) which defines the prompt used to instruct the LLM to paraphrase the questions.
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`)

## Example usage:
```bash
python rambla/run/run_task.py task=negation model=openai_chat
```