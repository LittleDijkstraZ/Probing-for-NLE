import pandas as pd
def get_CommonsenseQA():
    return pd.read_csv("../data/cos-e/parsed-cose-train.csv")