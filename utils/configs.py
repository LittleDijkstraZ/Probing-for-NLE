class config_base(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self["nle_generation"] = dict(
            total_trials = 4,
            length_increment = 512,
            max_length = 1024,
        )
        self['generation_configs'] = dict(
            num_beams=5,
            # renormalize_logits=True,
            early_stopping=True,
        )
config = config_base()


gpt_j_config = config_base()
falcon7b_config = config_base()

# we use contrastive search for mpt
mpt7b_instruct_config = config_base()
mpt7b_instruct_config['generation_configs'] = dict(
    # max_new_tokens=8, 
    # no_repeat_ngram_size=8, 
    # pad_token_id=tokenizer.eos_token_id, # this will and should be specified in the main code, after the tokenizer is loaded
    renormalize_logits=True,
    early_stopping=True,

    # penalty_alpha=1.3,
    penalty_alpha=0.3, 
    top_k=12,
    # top_k=6,
    use_cache=True,
)
llama2_7b_config = config_base()
llama2_7b_config['generation_configs'] = dict(
    # max_new_tokens=8, 
    # no_repeat_ngram_size=8, 
    # pad_token_id=tokenizer.eos_token_id, # this will and should be specified in the main code, after the tokenizer is loaded
    renormalize_logits=True,
    early_stopping=True,

    # penalty_alpha=1.3,
    penalty_alpha=0.3, 
    top_k=12,
    # top_k=6,
    use_cache=True,
)