# Task: `PromptRobustness` @ `rambla.tasks.prompt_robustness.prompt_robustness`

## Goal
Evaluate the model's robustness against spelling mistakes by randomly replacing alphanumerical characters in the prompt with new characters (matching the case).

## Overview:
- This task takes an existing task ([baseline MCQA](docs/tasks/mcqabaseline.md) by default) and applies mutations designed to simulate spelling mistakes to the prompt
- The task is re-run multiple times with a different number of mutations applied to the prompt on each run
- The performance across each run is compared at the end to assess the impact of spelling mistakes on model performance

## Config:
The default config can be found in `rambla.conf.task.spelling_robustness.yaml`.

The config should define a **mutation schedule**: a list of integers specifying the number of mutations to apply per-prompt on each run. Typically this should be an increasing list, for example:
```yaml
mutation_schedule:
- 1
- 3
- 10
- 20
- 50
```

This will apply 1 mutation per-prompt on the first run, 3 mutations per-prompt on the second run, 10 mutations on the third... etc.

It can be configurated by changing the variables and/or its call to the following configs:
- a subtask config; the task on which to add "spelling mistakes". Its default is the [MCQABaseline](docs/tasks/mcqabaseline.md). (choose from `rambla.conf.task`)
- a mutator config that determines the type of mutations to apply, such as randomly changing characters, or adding random whitespaces (choose from `rambla.conf.prompt_mutator`).

## Example usage:
```bash
python rambla/run/run_task.py task=spelling_robustness model=openai_chat
```