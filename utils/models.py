
import torch

def get_model_general(model_name):
    from transformers import AutoTokenizer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipe = pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0",
    )
    tokenizer = pipe.tokenizer
    model = pipe.model
    return model, tokenizer


def get_gpt_j():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained('nlpcloud/instruct-gpt-j-fp16')
    generator = AutoModelForCausalLM.from_pretrained("nlpcloud/instruct-gpt-j-fp16", torch_dtype=torch.float16)
    return generator, tokenizer

# def get_openllama(model_path = 'openlm-research/open_llama_3b'):
#     # model_path = 'openlm-research/open_llama_7b'
#     from transformers import LlamaTokenizer, LlamaForCausalLM

#     tokenizer = LlamaTokenizer.from_pretrained(model_path)
#     generator = LlamaForCausalLM.from_pretrained(
#         model_path, torch_dtype=torch.float16, device_map='cuda:0',
#     )
#     return generator, tokenizer

def get_openllama_auto(model_path = 'openlm-research/open_llama_3b'):
    # model_path = 'openlm-research/open_llama_7b'
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("openlm-research/open_llama_3b", legacy=False)
    generator = AutoModelForCausalLM.from_pretrained(
        "openlm-research/open_llama_3b", torch_dtype=torch.float16,
    )
    return generator, tokenizer

def get_debertav3nli():
    
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("sileod/deberta-v3-base-tasksource-nli")
    generator = AutoModelForCausalLM.from_pretrained(
        "sileod/deberta-v3-base-tasksource-nli", torch_dtype=torch.float16,
    )
    return generator, tokenizer

def get_gpt2():
    
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    generator = AutoModelForCausalLM.from_pretrained(
        "gpt2", torch_dtype=torch.float16,
    )
    return generator, tokenizer

def get_llama2():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", torch_dtype=torch.float16)
    return model, tokenizer


