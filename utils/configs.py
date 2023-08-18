class config_base(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self["nle_generation"] = dict(
            total_trials = 4,
            length_increment = 512,
            max_length = 1024,
        )
config = config_base()
