# Task: rambla/conf/template

## Goal
Templates are used to prompt LLMs and generate outputs used in the tasks.

## Overview

Templates used for prompting LLMs are task specific. 

## Shortform tasks
- Templates describe the task, expected output and contain fields required for the spesfic task.
- For example pqa_yesno_with_context.yaml file asks the LLM to answer the question with 'yes' or 'no' and has field for the question and context.

## Longform 
- Templates describe the task, expected output and contain fields required for the spesfic task.
- For example the similarity_long_form_summarise_bioasq.yaml file asks the LLM to provide a 1 sentence question, where by the answer provided would be an appropriate answer for the question and has a field for the answer.

## Text to text tasks

- Templates contain fields for text pairs and ask the LLM to understand if they are semantically similar or not and respond with either yes or no as appropriate.
- Alternative templates were created to investigate the impact of prompting on performance.
- Templates with context were created to investigate the impact of providing examples in the prompt. These examples are unique for each dataset and are from the test set when available. 
- Note that '\n' was not used to seperate lines due to other templates not using it very frequently. But we could consider adding this in the future.

## Example usage:
```
Within llm_component.yaml

  - ../template@params.prompt_formatter_config: descriptive_text_to_text
```