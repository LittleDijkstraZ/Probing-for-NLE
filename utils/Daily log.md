
## 0823
- Realized with horror that the accuracy of fewshot classification is terrible.
- Realized that fewshot with NLE generated at the same time may save some bad generations.
- Need to figure out a better way for fewshot answer generation. Unless we switch to OpenLlama, which seemed to be better.

## 0822
- Found a way to stably producing fewshot classification via GPT-j
- Generated the NLE for GPT-j, took 1:30 mins to finish 1600 samples

## 0820
- Mimiced the tutorial sample code and found that GPT-j with a slight modification will produce the correct shap score.
- Decide to regenerate the classification and NLE for GPT-j from the beginninng again.
- Realizing generating the 2 things at the same is not stable on GPT-j.
- Need to generate classification first and then generate NLE based on that

## 0819
- Tried to implement that shap method for OpenLlama, only realizing that it encountered bugs.
- Initially thought the bug was due to the mismatch between the shap tutorial and the current shap library, as well as an mismatch between shap and hugging face.
- Turned out that the tutorial sample code actually worked.

## 0818
- Realized that GPT-j or OpenLlama are only text generation models, not text classification models. So zero-shot classification is not possible.
- Found a way of calculating shap score on text generation