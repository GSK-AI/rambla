## Adding a new text-to-text task

The `TextToTextSimilarityEvaluation` class (found in `rambla/text_to_text_tasks/text_to_text_similarity_eval.py`) is flexible enough to work with different datasets and different `TextToTextSimilarityComponent` components. If you wish to add an additional text-to-text task you will need to follow these steps:
1. Add a python script in the `rambla/text_to_text_tasks` directory that contains:
    - A `pydantic.BaseModel` config class with the required attributes
    - A task class that inherits from `BaseTextToTextTask` (found in `rambla/text_to_text_tasks/base.py`).
2. Add tests.
3. In the `rambla/conf/text_to_text_task` directory, create config files for your task (create a unique file for each dataset you want to use). Note that it must contain a `class_key` and conform to the config class created above.
4. Add the config name and task class name to the `TEXT_TO_TEXT_TASK_MAP` in `rambla/text_to_text_tasks/__init__.py` so that it can be found.
3. Run a text to text task using the new task and a suitable dataset using the `rambla/run/run_text_to_text.py` script with the appropriate command. For example, 

```bash
python rambla/run/run_test_to_text.py text_to_text_task=text_to_text_mrpc text_to_text_component=llm_component
```