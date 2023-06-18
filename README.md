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

- Probing Plan for COS-E dataset:
    - Inputs: 
        - Question $Q$ 
        - Model's Explanation $E_m$ 
        - Model's Answer $a_m$ (ignored for now)
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
                
                
