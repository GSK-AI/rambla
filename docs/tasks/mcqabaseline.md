# Task: `MCQABaseline` @ `rambla.tasks.baseline.baseline`

## Goal
Evaluate the model against shortform QA datasets. Comes in two settings: with and without context.

## Overview: 
For QA datasets, such as PubMedQA, every row comes with {question, context, answer}.

- Setting 1: Provide the model with just the question and evaluate its prediction against the ground-truth shortform answer from the dataset.

- Setting 2: Provide the model with both the question and the context and evaluate its prediction against the ground-truth shortform answer from the dataset.

Setting 1 tests knowledge, while setting 2 tests “understanding” from context (more relevant for testing reliability specifically). The default is setting 2, and this can be changed by editing the prompt template (see config breakdown in next section) to remove the context field.

## Config:
The default config can be found in `rambla.conf.task.mcqabaseline.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`).

## Example usage:
```bash
python rambla/run/run_task.py task=mcqabaseline model=openai_chat"
```