# Task: `MCQALongFormTask` @ `rambla.tasks.longform.longform.py`

## Goal
Evaluate an LLM’s ability to generate long-form summaries of text. The summary is considered of high quality if the question can be answered using the summary instead of the full text as context.

## Overview: 
- LLM under evaluation is prompted to generate a conclusion for the context associated with each PubMedQA-labeled instance.

- Each generated conclusion is paired with the original PubMedQA question and passed to a scoring LLM (ChatGPT) which is prompted to answer the question with “yes”, “no” or “unknown”, using only the generated conclusion.

- Answers of the scoring LLM (yes/no/unknown) are compared to PMQA ground truth answer for each summary.

## Config:
The default config can be found in `rambla.conf.task.conclusion_generation.yaml`. It can be configurated by changing the variables and/or its call to the following configs:
- a dataset config (choose from `rambla.conf.dataset`)
- a "longform prompt formatter" config which defines the prompt used to ask the LLM to generate summaries (from `rambla.conf.template`)
- a scoring model config which defines the model used to evaluate the generated summaries by using them to answer the relevant questions (choose from `rambla.conf.model`)
- a response formatter (choose from `rambla.conf.response_formatter`)
- a response prompt formatter config (choose from `rambla.conf.template`).

## Example usage:
```bash
python rambla/run/run_task.py task=conclusion_generation model=openai_chat
```
