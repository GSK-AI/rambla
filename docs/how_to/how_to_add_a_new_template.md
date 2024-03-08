## Adding a new template

To add an additional template to the repository you will need to follow these steps:
1. In the `rambla/conf/template` directory, create a `.yaml` file for your template. The prompt template should be listed under a key called `template`.
2. In order to use it you need to create a new task config file and add the template's name in the template variable under `prompt_formatter_config`.
