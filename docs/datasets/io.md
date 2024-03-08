# Data Sources
## Generic Huggingface Datasets

The following are datasets we access through the [huggingface](https://huggingface.co/docs/datasets/en/index) (HF) ecosystem which can be used in this repository. They are loaded from the `prepare_generic_hf_dataset` function from [rambla.datasets.io.py](/rambla/datasets/io.py) with params set as shown below. The params can be changed/accessed from the dataset config files found in [rambla.conf.dataset](/rambla/conf/dataset).

| Name | Path | Subset | Split |
| -------- | ------- | -------- | ------- |
| sick | sick | | train |
| glue_mrcp | glue | mrpc | train |
| pubmed_qa_long_form | pubmed_qa | pqa_labeled | train |

## Local Datasets

bioasq was downloaded from here (note that you need an account): http://participants-area.bioasq.org/datasets/

# Data Uses
## Shortform datasets

The following datasets are used by the short form tasks to evaluate the reliability of an LLM.
- pubmed_qa (HF)

## Longform datasets

The following datasets are used by the similarity long form tasks to evaluate the long form generative performance of LLM models based on semantic similarity with a ground truth, which is determined by a task_text_to_text_component. 

| Name | Task | Bio | Size | Source |
| -------- | ------- | -------- | ------- | ------- |
| bioasq_task_b | Summarisation | Yes | 4719 | Downloaded |
| bioasq_task_b | Q&A | Yes | 4719 | Downloaded |
| pubmed_qa | Q&A | Yes | 1000 | HF |

## Text to text datasets

The following datasets contain sentence pairs labeled for similarity and are used by the text to text tasks to evaluate text to text components at determining semantic similarity. 

| Name | Label | Bio | Size | Source |
| -------- | ------- | -------- | ------- | ------- |
| glue_mrcp | Binary | No | 3668 | HF |
| SICK | Categorical (NLI) and continuous | No | 4439 | HF |
