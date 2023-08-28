class config_base(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self["nle_generation"] = dict(
            total_trials = 4,
            length_increment = 512,
            max_length = 1024,
        )
        self['generation_configs'] = dict(
            return_dict_in_generate =True,
            output_scores=True,
        )
config = config_base()


gpt_j_config = config_base()
gpt_j_config['generation_configs'].update(
    max_new_tokens=324,
    num_beams=5,
    # no_repeat_ngram_size=2, 
    early_stopping=False,

)