## Adding a new text-to-text component

To add an additional text-to-text component to the repository you will need to follow these steps:
1. Add a python script in the `rambla/text_to_text_components` directory that contains: 
    - A `pydantic.BaseModel` config class with the required attributes
    - A component class that inherits from `BaseTextToTextSimilarityComponent`
2. Add tests. 
3. In the `rambla/conf/text_to_text_component` directory, create config files for your component. Note that it must contain a name and conform to the config class created above.
4. Add the config name and component class name to the `COMPONENT_MAP` in `rambla/text_to_text_components/__init__.py` so that it can be found.
3. Run a text to text task using your new component and an suitable dataset using the `rambla/run/run_text_to_text.py` script with the appropriate command. For example, 

```bash
python rambla/run/run_test_to_text.py text_to_text_task=text_to_text_mrpc text_to_text_component=llm_component
```