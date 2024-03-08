# Task: `DistractingContext` @ `rambla.tasks.distracting_context`

## Goal
Evaluate the model's ability to use the relevant context when it is "buried" within irrelevant context.

## Overview: 
Take a dataset that carries a `context` field and create a new `context` field that is an augmented version of the original `context` WITH shuffled versions of it (ie with context from other questions in the dataset).

## Config:
The default config can be found in `rambla.conf.task.distracting_context.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a prompt formatter config (choose from `rambla.conf.template`)
- a response formatter (choose from `rambla.conf.response_formatter`)

## Example usage:
```bash
python rambla/run/run_task.py task=distracting_context model=openai_chat
```