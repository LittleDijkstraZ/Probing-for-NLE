import abc
import re
import numpy as np

def find_subarray(arr, subarr):
    for i in range(len(arr)):
        if ''.join(arr[i:i+len(subarr)]) == ''.join(subarr):
            return i, i+len(subarr)-1
    return None

def check_and_erase(source, target):
    bool_results = []
    true_ends = 0
    for x in source:
        if x == '':
            bool_results.append(False)
            continue
        if x in target:
            bool_results.append(True)
            true_ends = 1
            target = target.replace(x, "")
        else:
            if true_ends == 1:
                break
            bool_results.append(False)
    bool_results = bool_results + [False]*(len(source)-len(bool_results))
    return np.array(bool_results)

class ProcessingForLM(abc.ABC):

    @staticmethod
    def make_choice(sample_df, key = "label"):
        return [sample[f"choice_{int(sample[key])}"] for _,sample in sample_df.iterrows()]
    
    @abc.abstractmethod
    def generate_zeroshot_prompt_QA(self, few_shot_samples, input_premise, input_choices, input_label, label_idx, *args, **kwargs):
        pass

    @abc.abstractmethod
    def get_answer_from_output_text(self, output_text):
        pass



class mpt7b_instruct(ProcessingForLM):
    def __init__(self):
        self.Zeroshot_QA_sample = """Base on commonsense, make a choice:\n"""
        self.Zeroshot_QAE_sample = """Based on commonsense:\n"""
    @staticmethod
    def create_choices(sample_df, add_prefix=True):
        if add_prefix:
            return [", ".join(["\"" + f"{cx+1}. "+(sample[f"choice_{cx}"]).strip(".") + "\"" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        else:
            return [", ".join(["\"" + (sample[f"choice_{cx}"]).strip(".")+ "\"" for cx in range(5)]) for _,sample in sample_df.iterrows()]
        
    def generate_zeroshot_prompt_QA(self, input_premise, input_choices, input_label, label_idx, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QA_sample
        # query = f">>QUESTION<<\n{input_premise}.\n>>CONTEXT<<\nBased on commonsense, copy the best choice after '>>ANSWER<<': {input_choices}\n>>ANSWER<<\n"
        # query = f"\n>>CONTEXT<< {input_premise}\n>>QUESTION<< What is the best choice in: {input_choices}?\n>>ANSWER<< The best choice is"
        # input_choices = input_choices[0].upper() + input_choices[1:]
        # last_comma_index = input_choices.rfind(',')
        # input_choices = input_choices[:last_comma_index] + ', or' + input_choices[last_comma_index + 1:]
        query = f"""Question:\n{input_premise} \nChoices: \n{input_choices} \nThe best choice is: """
        prompt = few_shot_samples + query
        return prompt
    
    def generate_zeroshot_prompt_QAE(self, input_premise, input_choices, input_label, label_idx, few_shot_samples=None, *args, **kwargs):
        if few_shot_samples is None: 
            few_shot_samples = self.Zeroshot_QAE_sample
        # query = f">>QUESTION<<\n{input_premise}.\n>>CONTEXT<<\nBased on commonsense, copy the best choice after '>>ANSWER<<': {input_choices}\n>>ANSWER<<\n"
        # query = f"\n>>CONTEXT<< {input_premise}\n>>QUESTION<< What is the best choice in: {input_choices}?\n>>ANSWER<< The best choice is"
        # input_choices = input_choices[0].upper() + input_choices[1:]
        # last_comma_index = input_choices.rfind(',')
        # input_choices = input_choices[:last_comma_index] + ', or' + input_choices[last_comma_index + 1:]
        query = f"""Question: \n{input_premise} \nChoices: \n{input_choices} \nThe best choice is: \n\"{input_label}\"\nExplanation: """
        prompt = few_shot_samples + query
        return prompt

    def get_answer_from_output_text(self, output_text, input_choices_list, idx=None):
        answer = re.findall(r"is: (.*.*\n*.*)\"", output_text)
        try:
            answer = answer[0]
            answer = answer.replace("\n", "").strip(' ').strip("\"")
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
        explanation = re.findall(r'Explanation: ([\w\W]*?)\<', output_text)
        try:
            explanation = explanation[0]
            explanation = explanation.replace("\n", "").strip(' ')
        except:
            explanation = None
            print(">>>>>>>>>>>>>>model answer not found>>>>>>>>>>>>>>")
            print(output_text)
            print(explanation)

            if idx: print(idx)
        return explanation

    def get_context_shap(self, shap_values, target_choice):
        text_data = shap_values.data[0]
        question_start = np.arange(len(text_data))[np.where(text_data == "Question")].max() + 2 # :, \n, 
        try:
            question_end, choices_start = find_subarray(text_data, ['Cho', 'ices', ': ', '', '\n']) # :, \n, 
        except:
            print(text_data)
            print("Cannot find quesiton_end")
            return None
        question = list(text_data[question_start:question_end])
        choices_end = find_subarray(text_data, ['\n', 'The ', 'best '])[0]
        choices = list(text_data[choices_start:choices_end])
        # print(question, choices)

        output_source = list(shap_values._s._aliases['output_names'])
        output_positions = check_and_erase(output_source, target_choice)
        valid_shap = shap_values.values[0][..., output_positions]

        question_shap = valid_shap[question_start:question_end]
        choices_shap = valid_shap[choices_start:choices_end]
        context = question + choices
        context_shap = np.vstack([question_shap, choices_shap])
        return {'question': question,
                'question_shap': question_shap,
                'choices': choices,
                'choices_shap': choices_shap,
                'context': context,
                'context_shap': context_shap,}
    
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
    
all = ['mpt7b_instruct']