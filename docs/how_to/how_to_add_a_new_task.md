## Adding a new task

To add an additional tasks to the repository you will need to follow these steps:
1. Design the task so that it assesses either shortform or longform generation. Add a python script in the [rambla/tasks](/rambla/tasks) directory that contains:
    - A `pydantic.BaseModel` config class with the required attributes
    - A task class that inherits from `BaseTask` (This can be found in: [rambla/tasks/base.py](/rambla/tasks/base.py)). 
2. Add tests.
2. In the [rambla/conf/task](/rambla/conf/task) directory, create config files for your task (create a unique file for each dataset you want to use). Note that it must contain a `class_key` and conform to the config class created above.
4. Add the config name and task class name to the TASK_MAP in [rambla/tasks/__init__.py](/rambla/tasks/__init__.py) so that it can be found.
5. Identify a suitable dataset and if necessary add the dataset to the repository.
6. Run the new task using a suitable dataset using scripts in the [rambla/run](/rambla/run) directory with the appropriate command. Example commands can be found for existing tasks in [docs/tasks](/docs/tasks).
