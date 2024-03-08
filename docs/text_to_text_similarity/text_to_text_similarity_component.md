## Goal

Get a score for the similarity between two pieces of text. Components take two sentences as an input and produce a result based on the semantic similarity of two pieces of text (for some components the output is categorical, for others the output is continuous).

## Overview

Four types of components have been created:

- __LLM Component__: The LMM is prompted to answer if two pieces of text from a dataset are semantically similar. Responses from the LLM indicate the degree of similarity between the two pieces of provided text. Returns Yes (1) or No (0).
Component: `LLMTextToTextSimilarity` @ `rambla/text_to_text_components/llm_similarity_component.py`

- __Embedding-based Component__: We embed the two sentences using an embedding model and take the inner product to return a score (If the embeddings are normalised then this score will range from 0 to 1).
Component: `EmbeddingBasedTextToTextComponent` @ `rambla/text_to_text_components/embeddings_component.py`

- __NLI Model Component__: We pass the two sentences into the NLI model and from the output determine the final prediction. These models are pre-trained to classify an ordered pair of sentences as one of: {entailment, neutral, contradiction}. We can run the models in a unidirectional or bidirectional way and return a catagorical or continuous output. Unidirectional: “Does sentence A follow from sentence B?”, classification: argmax of the scores (returns predicted class), regression: exponential softmax of the entailment score (returns a score between 0 and 1). Bidirectional: “Does sentence A follow from sentence B AND does sentence B follow from sentence A?”, classification: strict - bidirectional entailment required for similarity classification, relaxed - bidirectional entailment or entailment and neutral required for similarity classification, regression: mean exponential softmax of the entailment score (returns a score between 0 and 1).
Components: `NLIUnidirectional` and `NLIBidirectional` @ `rambla/text_to_text_components/nli_wrappers.py`

- __NLP Component__: Uses BLEU or ROUGE metrics to evaluate semantic similarity. Returns a score between 0 and 1.
Component: `NgramTextToTextSimilarity` @ `rambla/text_to_text_components/nlp_component.py`

Text to text tasks evaluate the components through utilising datasets that contain sentence pairs labeled for similarity. By comparing the components responses against the ground-truth we can evaluate the component.
