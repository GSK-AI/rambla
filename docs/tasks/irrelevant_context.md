# Task: `IrrelevantContext` @ `rambla.tasks.irrelevant_context.irrelevant_context`

## Goal
The model is provided with a question and a piece of irrelevant context. The goal is to see whether the model can understand that the context is not relevant to the question and respond with "Unknown".

## Overview: 
We provide the model with a random piece of irrelevant context together with the question. We prompt the model to respond with either Yes, No or Unknown. The desired outcome would be that the model responds with Unknown, given that we ask it to use the context.

## Config:
The default config can be found in `rambla.conf.task.irrelevant_context.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a prompt formatter config (choose from `rambla.conf.template`)
- an evaluator config (choose from `rambla.conf.evaluator`)

## Example usage:
```bash
python rambla/run/run_task.py task=irrelevant_context model=openai_chat
```