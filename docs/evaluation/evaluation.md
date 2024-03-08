## Goal

Evaluates a models outputs against labels.

## Overview

Evaluators return the performance and results for a model's categorical or continuous predictions against labels within the dataset. The MCQAEvalComponent can be used in tasks where the model returns categorical predictions, while the ContinuousEvalComponent is used when continuous values are outputted. When an LLM is being evaluated the ResponseQualityEvalComponent can be used to compare the length in tokens of an LLMs output vs a ground truth in similarity long form tasks.

Example: `MCQAEvalComponent` @ `rambla/evaluation/shortform.py`

Example: `ContinuousEvalComponent` @ `rambla/evaluation/continuous.py`

Example: `ResponseQualityEvalComponent` @ `rambla/evaluation/longform.py`
