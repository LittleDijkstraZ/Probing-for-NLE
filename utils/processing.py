import abc
import re
import numpy as np
import string

def find_subarray(arr, subarr):
    for i in range(len(arr)):
        if ''.join(arr[i:i+len(subarr)]) == ''.join(subarr):
            return i, i+len(subarr)-1
    return None

def check_and_erase(source, target):
    full_target = target
    start_pos = None
    end_pos = None
    for idx, x in enumerate(source):
        if x == '':
            continue
        if x == target[:len(x)]:
            if target == full_target:
                start_pos = idx
            target = target.replace(x, "")
            if target == "":
                end_pos = idx
                break
        else:
            if start_pos != None:
                target = full_target
    if end_pos != None: end_pos += 1
    return start_pos, end_pos

class ProcessingForLM(abc.ABC):
    def __init__(self):
        self.Zeroshot_QA_sample = """Answer the following based on commonsense:\n"""
        self.Zeroshot_QAE_sample = """Based on commonsense:\n"""
        self.answer_regex = r'ice: (.*)\n'
        self.explanation_regex = r'Explanation: ([\w\W]*?)\<'
        self.shape_quesiton_start = 'Question'
        self.shap_choice_start =  ['Choose', ' from', ':']
        self.shap_choice_end =  ['\n', 'Best', ' answer', ' choice', ':']
    
    @abc.abstractmethod
    def create_choices(self, sample_df, add_prefix):
        pass

    @abc.abstractmethod
    def generate_zeroshot_prompt_QA(self, input_premise, input_choices, input_label=None, label_idx=None, few_shot_samples=None, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def generate_zeroshot_prompt_QAE(self, input_premise, input_choices, input_label, label_idx=None, few_shot_samples=None, *args, **kwargs):
        pass

    @staticmethod
    def make_choice(sample_df, key = "label"):
        return [sample[f"choice_{int(sample[key])}"] for _,sample in sample_df.iterrows()]
        
    def get_answer_from_output_text(self, output_text, input_choices_list, idx=None):
        answer = re.findall(self.answer_regex, output_text)
        input_choices_list = [x.strip(string.punctuation) for x in input_choices_list]
        try:
            answer = answer[0]
            answer = answer.replace("</s>", "")
            answer = answer.replace("\n", "").lower().strip(' ').strip(string.punctuation)
            model_answer = input_choices_list.index(answer)
        except:
            model_answer = None
            print(">>>>>>>>>>>>>>model answer not found>>>>>>>>>>>>>>")
            print(output_text)
            print(answer)
            print(input_choices_list)

            if idx: print(idx)
        return model_answer
    
    def get_explanation_from_output_text(self, output_text, idx=None):
        explanation = re.findall(self.explanation_regex, output_text)
        try:
            explanation = explanation[0]
            explanation = explanation.replace("</s>", "")
            explanation = explanation.replace("\n", "").strip(' ')
        except:
            explanation = None
            print(">>>>>>>>>>>>>>model answer not found>>>>>>>>>>>>>>")
            print(output_text)
            print(explanation)

            if idx: print(idx)
        return explanation
        
    def get_max_percent_shap(self,context_shap, top_k=5):
        context_shap[context_shap<0] = 0
        pos_context_shap = context_shap.sum(-1)
        max_index = np.argpartition(pos_context_shap, -top_k)[-top_k:]
        max_percent_shap = pos_context_shap[max_index].sum() / pos_context_shap.sum()
        return max_percent_shap
    
    def get_counter_factual_ratio(self, question_shap, choices_shap, top_k=5):
        question_shap[question_shap<0] = 0
        choices_shap[choices_shap<0] = 0        
        return  choices_shap.sum() / (question_shap.sum() + choices_shap.sum()) 
    
    def get_text_wise_shap(self, context_shap):
        pos_context_shap = context_shap.sum(-1)
        return list(pos_context_shap)
    
    def get_context_shap(self, shap_values, target_choice):
        try:
            text_data = shap_values.data[0]
            question_start = np.arange(len(text_data))[np.where(text_data == self.shape_quesiton_start)].max() + 2 # :, \n, 
            question_end, choices_start = find_subarray(text_data, self.shap_choice_start) # :, \n, 

            question = list(text_data[question_start:question_end])
            choices_end = find_subarray(text_data, self.shap_choice_start)[0]
            choices = list(text_data[choices_start:choices_end])

            output_source = list(shap_values._s._aliases['output_names']) # outputs being shapped on obtained from here
            os, oe = check_and_erase(output_source, target_choice)
            valid_shap = shap_values.values[0][..., os:oe]

            cs, ce = check_and_erase(choices, target_choice)

            question_shap = valid_shap[question_start:question_end]
            choices_shap = valid_shap[choices_start:choices_end]
            clean_choices_shap = np.vstack([choices_shap[:cs], choices_shap[ce:]])
            clean_choices = choices[:cs]+choices[ce:]
            target = choices[cs:ce]

            context = question + clean_choices
            context_shap = np.vstack([question_shap, clean_choices_shap])
            return {'question': question,
                'question_shap': question_shap,
                'clean_choices': clean_choices,
                'clean_choices_shap': choices_shap,
                'context': context,
                'context_shap': context_shap,
                'target': target}
        except:
            print(">>>>>>>>>>>>>>model answer not found>>>>>>>>>>>>>>")
            print(text_data)
            return None
    

class mpt7b_instruct(ProcessingForLM):
    def __init__(self):
        self.Zeroshot_QA_sample = """Base on commonsense, make a choice:\n"""
        self.Zeroshot_QAE_sample = """Based on commonsense:\n"""
        self.answer_regex = r"is: (.*.*\n*.*)\""
        self.explanation_regex = r'Explanation: ([\w\W]*?)\<'
        self.shape_quesiton_start = 'Question'
        self.shap_choice_start = ['Cho', 'ices', ': ', '', '\n']
        self.shap_choice_end = ['\n', 'The ', 'best ']
        
    def create_choices(self, sample_df, add_prefix=True):
        if add_prefix:
            return [", ".join(["\"" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "\"" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        else:
            return [", ".join(["\"" + (sample[f"choice_{cx}"]).strip(".")+ "\"" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        
    def generate_zeroshot_prompt_QA(self, input_premise, input_choices, input_label=None, label_idx=None, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QA_sample
        query = f"""Question:\n{input_premise} \nChoices: \n{input_choices} \nThe best choice is: """
        prompt = few_shot_samples + query
        return prompt
    
    def generate_zeroshot_prompt_QAE(self, input_premise, input_choices, input_label, label_idx=None, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QAE_sample
        query = f"""Question: \n{input_premise} \nChoices: \n{input_choices} \nThe best choice is: \n\"{input_label}\"\nExplanation: """
        prompt = few_shot_samples + query
        return prompt


class gpt_j(ProcessingForLM):
    def __init__(self):
        self.Zeroshot_QA_sample = """Answer the following based on commonsense:\n"""
        self.Zeroshot_QAE_sample = """Based on commonsense:\n"""
        self.answer_regex = r'ice: (.*)\n'
        self.explanation_regex = r'Explanation: ([\w\W]*?)\<'
        self.shape_quesiton_start = 'Question'
        self.shap_choice_start =  ['Choose', ' from', ':']
        self.shap_choice_end =  ['\n', 'Best', ' answer', ' choice', ':']
    
    def create_choices(self, sample_df, add_prefix=True):
        # if add_prefix:
        #     return [", ".join(["\"" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "\"" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        # else:
        #     return [", ".join(['"' + (sample[f"choice_{cx}"]).strip(".")+ '"' for cx in range(5)]) for _,sample in sample_df.iterrows()]
        if add_prefix:
            return [", ".join(["'" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "'" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        else:
            return [", ".join(["'" + (sample[f"choice_{cx}"]).strip(".")+ "'" for cx in range(5)]) for _,sample in sample_df.iterrows()]

    def generate_zeroshot_prompt_QA(self, input_premise, input_choices, input_label=None, label_idx=None, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QA_sample
        # query = f"""Question: {input_premise} \nChoices: {input_choices} \nAnswer: """
        query = f"Question: {input_premise}\nChoose from: {input_choices}.\nBest answer choice: "
        prompt = few_shot_samples + query
        return prompt
    
    def generate_zeroshot_prompt_QAE(self, input_premise, input_choices, input_label, label_idx=None, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QAE_sample
        query = f"""Question: {input_premise}\nChoices: {input_choices}.\nBest answer choice: '{input_label}'.\nExplanation: """
        prompt = few_shot_samples + query
        return prompt


class llama2_7b(ProcessingForLM):
    def __init__(self):
        # self.Zeroshot_QA_sample = """Answer the following based on commonsense:\n"""
        # self.Zeroshot_QAE_sample = """Based on commonsense:\n"""
        # self.answer_regex = r'choice: (.*)</'
        # self.explanation_regex = r'Explanation: ([\w\W]*?)\<'
        # self.shape_quesiton_start = 'Question'
        # self.shap_choice_start =  ['Choose', ' from', ':']
        # self.shap_choice_end =  ['\n', 'Best', ' answer', ' choice', ':']
        self.Zeroshot_QA_sample = """Based on commonsense, pick the best choice:\n"""
        self.Zeroshot_QAE_sample = """Based on commonsense, pick the best choice:\n"""
        self.answer_regex = r'ice: (.*)\n?'
        self.explanation_regex = r"Explanation:\s+([\w\W]*)"
        self.shape_quesiton_start = 'Question'
        self.shap_choice_start =  ['Choose', ' from', ':']
        self.shap_choice_end =  ['\n', 'Best', ' answer', ' choice', ':']
    
    def create_choices(self, sample_df, add_prefix=True):
        # if add_prefix:
        #     return [", ".join(["\"" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "\"" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        # else:
        #     return [", ".join(['"' + (sample[f"choice_{cx}"]).strip(".")+ '"' for cx in range(5)]) for _,sample in sample_df.iterrows()]
        if add_prefix:
            return [", ".join(["'" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "'" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        else:
            return [", ".join(["'" + (sample[f"choice_{cx}"]).strip(".")+ "'" for cx in range(5)]) for _,sample in sample_df.iterrows()]

    def generate_zeroshot_prompt_QA(self, input_premise, input_choices, input_label=None, label_idx=None, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QA_sample
        # query = f"""Question: {input_premise} \nChoices: {input_choices} \nAnswer: """
        query = f"Question: {input_premise}\nChoose from: {input_choices}\nBest Choice:"
        prompt = few_shot_samples + query
        return prompt
    
    def generate_zeroshot_prompt_QAE(self, input_premise, input_choices, input_label, label_idx=None, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QAE_sample
        query = f"""Question: {input_premise}\Choose from: {input_choices}.\nBest Choice: '{input_label}'.\nExplanation: """
        prompt = few_shot_samples + query
        return prompt

all = ['mpt7b_instruct', 'gpt_j', 'llama2_7b']