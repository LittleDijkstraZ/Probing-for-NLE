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
            renormalize_logits=True,
            no_repeat_ngram_size=8, 
            # early_stopping=True,
        )
config = config_base()


gpt_j_config = config_base()
