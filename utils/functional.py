

def create_choices(sample_df, add_prefix=True):
    if add_prefix:
        return [str([f"{cx+1}. "+sample[f"choice_{cx}"] for cx in range(5)]) for _,sample in sample_df.iterrows()]
    else:
        return [str([sample[f"choice_{cx}"] for cx in range(5)]) for _,sample in sample_df.iterrows()]

def make_choice(sample_df):
    return [sample[f"choice_{sample['label']}"] for _,sample in sample_df.iterrows()]

def generate_fewshot_prompt(few_shot_samples, input_premise, input_choices, input_label, label_idx):
    # query.append(f"Q {i}: {x}\nChoices: {s}\nThe correct choice is '{y}'\nExplanation:")
    query = f"Question: {input_premise}\nChoices: {input_choices}\nAnswer: {label_idx}. {input_label}\nExplanation:"
    prompt = few_shot_samples + query
    return prompt

def generate_fewshot_prompt_QA(few_shot_samples, input_premise, input_choices, input_label, label_idx):
    # query.append(f"Q {i}: {x}\nChoices: {s}\nThe correct choice is '{y}'\nExplanation:")
    query = f"Question: {input_premise}\nChoices: {input_choices}\nAnswer:"
    prompt = few_shot_samples + query
    return prompt