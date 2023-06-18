# Probing-for-NLE

## TODO:

- An explanation model
    - [ ] Some T5 based model
        - Use T5 based models to understand the format of rationalization on datasets 
        - Possible implementations: WT5, Leakage-adjusted ...
    - [ ] GPT
        - [x] Need to setup a secret key before use 
        - [ ] Build a pipeline to enable GPT to reply to a dataset of queries.

- An explanation dataset
    - [ ] Implement the COS-E dataset
        - [x] run the [scipt](data/cos-e/code/parse-commonsenseQA.py) to combine the original COS dataset and the human annotated explanations from COS-E into a csv file.
        - [ ] write a dataset class
            - [ ] retrieving a batch of Q+A+E 
            - [ ] tokenization and embedding

    - [ ] Implement e-SNLI dataset

- SHAP score calculation
    - [ ] Implement SHAP calculation

- Probe
    - [ ] Train a simple probing model on the generated explanations for the dataset.

---

## Probing Plan for COS-E dataset:
- Purpose:
    - To examine the ease of recovery of SHAP score
- Inputs: 
    - Question $Q$ 
    - $M$'s Explanation $E_m$ 
    - $M$'s Answer $a_m$ (ignored for now)
    - All Answer Choices $A$
- Output: Highest SHAP score
    - explainer = shap.Explainer(model_pipeline, model_tokenizer) 
    - max_shap = max(explainer($Q$))
- Pipeline:
    - tokenizer + feature embedding:
        - Roberta (Byte pair level)
        - Transformer (word level)
    - Probe (token level):
        - Method 1: Linear Regression
            - $Linear_1: \mathbb{R}^{n\times d} \rightarrow \mathbb{R}^n$ 
            - $Linear_2: \mathbb{R}^{n} \rightarrow \mathbb{R}^1$ 
            - Loss Function: MSE( $Linear_2(Linear_1$($E_m$)) - max_shap)
        - Method 2: Simple RNN
            - Loss Function: MSE(RNN($E_m$) - max_shap)
- Issues:
    1. For extractive question answering, there's a context for which SHAP score can be based on. But for commonsense reasoning, extra knowledge is required. Hence, merely providing $Q$ for SHAP score calculation might be a problem.
        - Solution 1:
            - Ideas:
                - Calculate SHAP for the explanations as well
            - Steps:
                - (Optional) Use the original model's final feature embeddings 
                - Calculate SHAP score for both the $E_m$ and the all choices $A$
                - Probe for Highest SHAP of $Q$ with respect to $E_m$ given $E_m$ and $Q$
                - Probe for max SHAP score's difference between $Q \rightarrow a_m$ and $Q \rightarrow E_m$ given $Q$, $E_m$ and $a_m$ 
        - Solution 2 (ref. leakage-adjusted simulatability):
            - Ideas:
                - Use another model $S$ to simulate the answer conditioned on the explanation, where a SHAP score for $E_m$ can be produced 
                - Probe for the SHAP using GPT-4's embeddings
                - May need to emphasize that the explanation is subjected to the question
                - May also try models other than GPT-4, could be as simple as LSTM
            - Steps:
                - Send $Q$ and $E$ to GPT-4
                - Calculate GPT-4's SHAP score for $A$ given $Q$ and $E$ 
                - Given $Q$ and $E_m$, probe for word level SHAP scores for each $a_i \in A$

---
## Questions:
1. Will probing actually be an reflection of the explaner model or the embedding model?
2. What will be the effect of answer leakages in the explanations?  