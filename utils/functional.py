

def create_choices(sample_df, add_prefix=True):
    if add_prefix:
        return [str([f"{cx+1}. "+sample[f"choice_{cx}"] for cx in range(5)]) for _,sample in sample_df.iterrows()]
    else:
        return [str([sample[f"choice_{cx}"] for cx in range(5)]) for _,sample in sample_df.iterrows()]
    
def create_choices_2(sample_df, add_prefix=True):
    if add_prefix:
        return [", ".join(["'" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "'" for cx in range(5)]) for _,sample in sample_df.iterrows()]
    else:
        return [", ".join(["'" + (sample[f"choice_{cx}"]).strip(".")+ "'" for cx in range(5)]) for _,sample in sample_df.iterrows()]

def create_choices_3(sample_df, add_prefix=True):
    if add_prefix:
        return [" \n "+" \n ".join([f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") for cx in range(5)]) for _,sample in sample_df.iterrows()]
    else:
        return [" \n "+" \n ".join([ (sample[f"choice_{cx}"]).strip(".") for cx in range(5)]) for _,sample in sample_df.iterrows()]
    
def create_choices_4(sample_df, add_prefix=True):
    if add_prefix:
        return [" ".join([f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + ";" for cx in range(5)]) for _,sample in sample_df.iterrows()]
    else:
        return [" ".join([(sample[f"choice_{cx}"]).strip(".") + ";" for cx in range(5)]) for _,sample in sample_df.iterrows()]

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

def generate_fewshot_prompt_AE(few_shot_samples, input_premise, input_choices, input_label, label_idx):
    # query.append(f"Q {i}: {x}\nChoices: {s}\nThe correct choice is '{y}'\nExplanation:")
    query = f"Question: {input_premise}\nChoices: {input_choices}\nAnswer: {input_label}\nExplanation:"
    prompt = few_shot_samples + query
    return prompt

def generate_fewshot_prompt_QA_new(few_shot_samples, input_premise, input_choices, input_label, label_idx):
    # query.append(f"Q {i}: {x}\nChoices: {s}\nThe correct choice is '{y}'\nExplanation:")
    query = f"Question: {input_premise}  Make a choice and provide an explanation.\nChoices: {input_choices}\nThe best choice is"
    prompt = few_shot_samples + query
    return prompt

# __all__ = ['create_choices', 'create_choices_2']
