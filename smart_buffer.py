class SmartReplayBuffer:
    def __init__(self, max_samples = 2000, max_samples_per_class = None):
        """
        max_samples: the absolute maximum number of samples the buffer can hold
        max_samples_per_class: the absolute maximum number of samples the buffer can hold from one class
        """
        self.buffer = [] # List of samples

    def create_model(self, samples):
        print()
