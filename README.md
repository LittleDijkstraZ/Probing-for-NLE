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
    - [ ] Implement e-SNLI dataset

- SHAP score calculation
    - [ ] Implement SHAP calculation

- Probe
    - [ ] Train a simple probing model on the generated explanations for the dataset.
    