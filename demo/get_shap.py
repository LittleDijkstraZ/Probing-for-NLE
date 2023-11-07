import sys
sys.path.append('..')

import warnings
# warnings.resetwarnings()
warnings.filterwarnings("ignore")
import pandas as pd
from transformers import ZeroShotClassificationPipeline
import torch
from tqdm.auto import tqdm
import shap
import numpy as np

from transformers import pipeline, ZeroShotClassificationPipeline
original_pipe = pipeline("zero-shot-classification",model="sileod/deberta-v3-base-tasksource-nli")

class ZeroShotModelPipeline(ZeroShotClassificationPipeline):
    # Overwrite the __call__ method
    labels = None
    def __call__(self, *args):
        out = super().__call__(args[0], self.labels)[0]
        # return [[{"label":x[0], "score": x[1]}  for x in zip(out["labels"], out["scores"])] for out in outs]
        return [[{"label":x[0], "score": x[1]}  for x in zip(out["labels"], out["scores"])]]

model = original_pipe.model
tokenizer = original_pipe.tokenizer

model = ZeroShotModelPipeline(
    model=model, tokenizer=tokenizer, 
    device=torch.device("cuda:0"), return_all_scores=True,
)
text = ["one day I will see the world"]
candidate_labels = ['travel', 'cooking', 'fighting', 'x', 'y']
# text = ["one day I will see the world"]
# candidate_labels = ['travel', 'cooking', 'fighting']

# classifier(text, candidate_labels, topk=20)
model.labels = candidate_labels
model.model.config.label2id.update({v:k for k,v in enumerate(model.labels)})
model.model.config.id2label.update({k:v for k,v in enumerate(model.labels)})
model(text)

if __name__ == '__main__':

    df = pd.read_csv('../COS-E_7191_7184_nle.csv')
    # s_range = range(0, len(df)//4 * 1)
    # s_range = range(len(df)//4 * 1, len(df)//4 * 2)
    # s_range = range(len(df)//4 * 2, len(df)//4 * 3)
    s_range = range(len(df)//4 * 3, len(df))

    # s_range = range(0, 10)
    # s_range = range(10, 20)
    # s_range = range(20, 30)



    top_k=3
    df = df.loc[s_range].copy()
    for i, sample in tqdm(df.iterrows(), total=len(df)):
        candidate_labels = [sample[f'choice_{i}'] for i in range(5)]
        model.labels = candidate_labels
        model.model.config.label2id.update({v:k for k,v in enumerate(model.labels)})
        model.model.config.id2label.update({k:v for k,v in enumerate(model.labels)})
        explainer = shap.Explainer(model)
        shap_values = explainer([sample.question])

        instance = shap_values[0,:,sample[f"choice_{int(sample.label)}"]]
        pos_vals = instance.values[instance.values>0]
        max_n = np.argsort(pos_vals)[-top_k:]
        max_sum = pos_vals[max_n].sum()
        df.loc[i, f'max_shap_value'] = max_sum
        # print(len(sample.question.split(" ")))
        # print(max_sum)

    df.to_csv(f'../COS-E_{str(s_range)}_shap.csv', index=False)
# candidate_labels