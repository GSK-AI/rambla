## Adding a new evaluator

To add an additional evaluator to the repository you will need to follow these steps:
1. Add a python script in the `rambla/evaluation` directory that contains: 
    - A `pydantic.BaseModel` config class with the required attributes
    - A class that inherits from `BaseEvalComponent` found in: `rambla/evaluation/base.py`.
2. Add tests.
3. In the `rambla/conf/evaluator` directory, create config files for the new evaluation component. Note it must contain `name` and `params` keys.
4. Add the config name and component's class name to the `EVAL_COMPONENT_MAP` in `rambla/evaluation/__init__.py` so that it can be found.
5. In order to use the new evaluation component, it needs to be referenced in a task config.