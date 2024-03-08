# Task: `ParentFewShotExamplesTask` @ `rambla.tasks.few_shot_examples.few_shot_examples`

## Goal
Evaluate the model robustness to few shot examples in short-form QA. Eg. is the model biased to responding "yes" if most of the example questions have "yes" as an answer (majority bias) or is it biased to repeating the most recent answer from the examples provided (recency bias).

## Overview: 
A set of "orders" (ie list of ordered short-form answers to be given in the example questions), such as "Yes, Yes, No" can be provided. The LLM will be prompted with a set of examples that meet the given order of answers followed by a question. Then, it is evaluated to see whether it is biased to responding a specific class out of the possible answers.
Can be used with or without context. The dataset of questions to prompt the LLM with (after the fewshot examples) is by default balanced so that the proportion of "yes" or "no" answers directly reflects on the model's bias.

## Config:
The default "parent task" config can be found in `rambla.conf.task.parent_fewshot_examples_with_context.yaml`. The "parent" class defines a list of configurations where each configuration is an ordered list of shortform answers to be used in the fewshot promt. For each configuration, the "parent" class calls the "child" class, which runs the fewshot experiment with the given order of fewshot examples. The default child task config can be found in `rambla.conf.task.fewshot_examples_with_context.yaml`. It can be configurated by changing the variables and/or its call to the following configs: 
- a dataset config (choose from `rambla.conf.dataset`)
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`)

## Example usage:
```bash
python rambla/run/run_task.py task=parent_fewshot_examples_with_context model=openai_chat
```