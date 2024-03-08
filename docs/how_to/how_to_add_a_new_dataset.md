## Adding a new dataset

To add an additional dataset to the repository you will need to follow these steps:
1. In `rambla/datasets/io.py` add the name of your dataset to the appropriate `DATASETS_LIST`, depending on if it is loaded from huggingface or from local storage or if it's an MCQA dataset. If loaded locally then then the `LOCAL_STORAGE_DATASET_DICT` will also need to be updated to provide a path to the file.
2. If required, add functionlity to preprocess the dataset as necessary for the specific task. This can be done in the `prepare_generic_hf_dataset` function for datasets loaded from huggingface, and the `prepare_local_dataset` function for datasets loaded from local storage.
3. In the `rambla/conf/dataset` directory, create config files for your dataset and the associated task you would like to run. Note that the dataset config files must contain `name` and `params` keys. Note that the config files must conform to the suitable `pydantic.BaseModel` class in `rambla.datasets.io.py`.
4. Run the new task against the new dataset using scripts in the `/rambla/run` directory with the appropriate command. For example, 

```bash
python rambla/run/run_task.py task=mcqabaseline model=openai_chat 
```