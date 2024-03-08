<h1 align="center">RAmBLA: A Framework for Evaluating the Reliability of LLMs as Assistants in the Biomedical Domain</h1>

RAmBLA (Reliability Assessment for Biomedical LLM Assistants) is a framework for evaluating LLMs on a set of tasks designed to test for reliability. Specifically, the tasks can be divided into the following three aspects of reliability:
1. __Robustness to non-semantic variations__: LLMs should be robust to prompt variations that do
not alter prompt meaning, and they should not display biases during few-shot prompting.
2. __High recall__: When operating on documents, LLMs should recall all relevant information, relying
on either parametric knowledge or context exclusively, as instructed.
3. __Hallucinations__: If they have insufficient knowledge or context information to answer a question,
LLMs should refuse to answer.

Further details can be found in our [paper](LINK PLACEHOLDER).


## Table of Contents
- [Installation](#installation)
- [Running Evaluations](#running-evaluations)
- [Tasks](#tasks)
- [Semantic/Textual similarity component evaluation](#semantictextual-similarity-component-evaluation)
- [Unit-Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [More Information](#more-information)
- [Contributing](#contributing)
- [License](#licence)
- [Contact Info](#contact-info)
- [Citing](#citing)

## Installation 

`RAmBLA` uses Python version 3.10.10. To install follow these steps:

1. Clone the repository:

```bash
git clone (URL placeholder)
```

2. Create a conda environment and **install the package** using the [Makefile](Makefile) with the following command:

```bash
make init
```

3. Set **environment variables** by creating a `.env` file according to [.env_example](.env_example). This includes the following environment variables:

| Variable | Description |
| ------   | ----        |
| `OPENAI_<var-name>` | Set of variables required to access OpenAI API |
| `DATASET_STORAGE_PATH` | Path where datasets should be stored |
| `MLFLOW_PROJECT_NAME` | Sets the name of the project to run evaluations under for logging purposes |
| `BACKOFF_MAX_TRIES`/`BACKOFF_INTERVAL` | Controls retry parameters when using API-based models |

4. Download the `bioasq` dataset under `DATASET_STORAGE_PATH`. See [this](docs/datasets/io.md) for instructions.

## Running Evaluations

### Individual Tasks

The main entry point for evaluating LLMs against an individual task is in [rambla/run/run_task.py](rambla/run/run_task.py). An example command is:
```bash
python rambla/run/run_task.py task=mcqabaseline model=openai_chat
```

__NOTE:__ We have a few model configs under [rambla/conf/model/](rambla/conf/model). For the case of [rambla/conf/model/llama2_7b_chat_local.yaml](rambla/conf/model/llama2_7b_chat_local.yaml) the `params.model_name` parameter needs to be updated to point to the path the model is stored.

All tasks in this repo are configured using [Hydra](https://hydra.cc/docs/intro/)

### Full Evaluation Suite

To run the full evaluation suite on a model use the script [bin/run_all_tasks.py](bin/run_all_tasks.py). For example:

```bash
python bin/run_all_tasks.py --models=openai_chat,mistral_7b_instruct_hf
```

This will run the full evaluation suite on ChatGPT and the Mistral 7b model

__NOTE__: Running the full evaluation suite can be very slow. We recommend running individual tasks over the full suite.

## Tasks

For detailed information of each task, including how to configure them and example run commands, please refer to the [docs](docs/).

### Supported Tasks

#### Robustness
- [MCQABaseline](docs/tasks/mcqabaseline.md)
- [Paraphrasing Task](docs/tasks/paraphrase.md)
- [Few-shot Prompt Bias](docs/tasks/few_shot_bias.md)
- [Spelling Mistakes Robustness](docs/tasks/spelling_robustness.md)
#### Recall
- [Recall from Negated Context](docs/tasks/recall_from_negated_context.md)
- [Recall from Distracting Context](docs/tasks/recall_from_distracting_context.md)
#### Hallucinations
- [MCQABaseline Longform](docs/tasks/mcqabaseline_longform.md)
- [Conclusion generation](docs/tasks/conclusion_generation.md)
- [Question Formation](docs/tasks/question_formation.md)
- [I don't know Task](docs/tasks/irrelevant_context.md)

### Supported LLMs
- [Openai Chat model](rambla/models/openai.py)
- [Openai Completion model](rambla/models/openai.py)
- [HuggingFace Text Generation Model](rambla/models/huggingface.py)
- [HuggingFace Natural-Language Inference (NLI) Model](rambla/models/huggingface.py)

### Supported datasets
- [PubmedQA](https://huggingface.co/datasets/pubmed_qa)
- [sick](https://huggingface.co/datasets/sick)
- [Glue Mrpc](https://huggingface.co/datasets/glue/viewer/mrpc)

## Semantic/Textual similarity component evaluation
This task was designed to evaluate different components (/models) at their ability to measure semantic similarity. These components take as input two pieces of text and output a score (binary or continuous) that reflects the similarity between the two input tests. The best performing component (chat GPT-4) was then chosen as default for the evaluation tasks where a semantic similarity metric was required.

### Supported tasks
We currently support one task, which consists in passing two long-form texts to a component and receiving a metric for how similar the two texts are. It can support different components against different datasets and capture a range of different metrics.

- [Textual Similarity Task](docs/textual_similarity/textual_similarity_task.md)

### Supported Components
#### LLM Component
We prompt GPT with the two sentences and ask whether they are semantically equivalent. Returns Yes or No.

#### Embedding-based Component
We first embed the two sentences using an embeddings model and then compute inner product between the two embeddings. Returns a score between 0 and 1 (if the embeddings are normalised).

#### NLI models (Natural Language Inference) (See `NLIModel` in [rambla/models/huggingface.py](rambla/models/huggingface.py))
We provide the two texts as input to the NLI model and the output are scores for the following classes: {entailment, neutral, contradiction}.

- Unidirectional model: “Does sentence A follow from sentence B?”

  - Classification: Argmax of the scores (returns predicted class)

  - Regression: Exponential softmax of the entailment score (returns a score between 0 and 1)

- Bidirectional model: “Does sentence A follow from sentence B AND does sentence B follow from sentence A?”

  - Classification: 

      - Strict - Bidirectional entailment required for similarity classification (this was our initial preferred method given results from the SICK dataset - please see below)

      - Relaxed - Bidirectional entailment or entailment and neutral required for similarity classification

    - Regression:

      - Average - Bidirectional mean exponential softmax of the entailment score (returns a score between 0 and 1)

### Supported datasets

- [MRPC (Microsoft Research Paraphrase Corpus)](https://www.microsoft.com/en-us/download/details.aspx?id=52398): Pairs of sentences which are either a paraphrase of each other or not - this could be extrapolated to imply similarity.
- [SICK (Sentences Involving Compositional Knowledge)](https://huggingface.co/datasets/sick): Pairs of sentences annotated for two crucial semantic tasks: relatedness in meaning (with a 5-point rating scale) and entailment relation between the two elements (with three possible gold labels: entailment, contradiction, and neutral).

## Unit-Tests
For testing, `RAmBLA` uses [pytest](https://docs.pytest.org/en/8.0.x/)

All unit-tests are located under [tests](/tests). To run the full test suite, run:

```bash
pytest tests/
```

## Integration tests
**NOTE: These need to be run manually!**

We have two sets of integration tests:
### Integration tests for [rambla/run/run_task.py](rambla/run/run_task.py)
Example usage:

- This will run a minimal version of the mcqabaseline task against openai_chat
  - `python integration_tests/run_task.py -m openai_chat -t mcqabaseline`

- This will run a minimal version of the mcqabaseline task against all available models
  - `python integration_tests/run_task.py -t mcqabaseline`

- This will run a minimal version of all available tasks against openai_chat
  - `python integration_tests/run_task.py -m openai_chat`

- This will run a minimal version of all available tasks against all available models
  - `python integration_tests/run_task.py`


### Integration tests for [rambla/run/run_text_to_text.py](rambla/run/run_text_to_text.py)
- This will run a minimal version of all available tasks against all available components.
  - `python integration_tests/run_text_to_text.py`


## More Information

For further details about working with `RAmBLA` see the extended documentation located under [docs](docs/)


## Contributing

We welcome contributions, feedback and suggestions to `RAmBLA`. If you would like to make a contribution, please follow our guidelines.

Please check for existing GitHub issues related to the change and create a new issue if one does not exist so we can first open discussions on the proposed change.

### Setting up local development environment

1. Clone and install the repo according to the [installation instructions](#installation)

2. Create a new branch:

```bash
git checkout -b <my-branch-name>
```

Ideally use the prefix `feat/` for feature-based branches, and `hotfix/` for bug fixes. 

### Making Changes

When you make changes to the code please ensure your changes adhere to our code style.

We use the following:

- [Numpy docstring style](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
- [black](https://github.com/psf/black) and [flake8](https://github.com/PyCQA/flake8) to ensure consistent code-style
- [isort](https://pycqa.github.io/isort/) to ensure imports are organised consistently

We use a [pre-commit](.pre-commit-config.yaml) to ensure all code adheres to these standards. If you install the package according to our [installation instructions](#installation) then this will be run automatically on every commit. To run manually use:

```bash
pre-commit run --all
```

### Testing

All code submissions should include unit-tests written using the [pytest](https://docs.pytest.org/en/8.0.x/) framework and these should be located in the relevant directory under [tests](tests/).

Please ensure all tests pass, before submitting a change, by following the [unit testing](#unit-tests) and [integration testing](#integration-tests) instructions.

### Submission

After following the above guidelines, please create a pull request into the `master` branch. Please ensure your pull-request contains:

* Title
* Brief description of changes made

## Licence

Copyright 2023 of GlaxoSmithKline Research & Development Limited. All rights reserved.

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Contact Info

`RAmBLA` was originally created by the **Responsible AI** team at [GSK.ai](https://gsk.ai/)

To get in touch please find our contact details:

* Rafael Poyiadzi: rafael.x.poyiadzi@gsk.com
* Ed Morrell: ed.r.morrell@gsk.com
* Gabriela van Bergen Gonzalez-Bueno: gabriela.v.vanbergengonzalez-bueno@gsk.com
* Lea Goetz: lea.x.goetz@gsk.com


## Citing

If you find this code useful in your research, please cite the associated paper:

```
@inproceedings{
bolton2024rambla,
title={{RAMBLA}: A {FRAMEWORK} {FOR} {EVALUATING} {THE} {RELIABILITY} {OF} {LLMS} {AS} {ASSISTANTS} {IN} {THE} {BIOMEDICAL} {DOMAIN}},
author={William James Bolton and Rafael Poyiadzi and Edward Morrell and Gabriela van Bergen Gonzalez Bueno and Lea Goetz},
booktitle={ICLR 2024 Workshop on Reliable and Responsible Foundation Models},
year={2024},
url={https://openreview.net/forum?id=lPXMUJlFfP}
}
```
