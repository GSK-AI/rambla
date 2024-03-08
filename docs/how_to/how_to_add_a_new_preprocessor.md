## Adding a new preprocessor

To add an additional preprocessor to the repository you will need to follow these steps:
1. Add a python script in the `rambla/preprocessing/` directory that contains:
    - A config class with the required attributes
    - A preprocessor that inherits from `BasePreprocessor`. 
2. Add tests.
3. In the `rambla/conf/preprocessor` directory, create config files for your preprocessor. Note it must contain `name` and `params`.
4. In order for the processor to be used it needs to be referenced in a task config file.