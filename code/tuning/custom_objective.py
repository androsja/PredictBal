# File: code/tuning/custom_objective.py
class CustomObjective:
    def __init__(self, use_validation=True):
        self.use_validation = use_validation
        self.name = 'val_loss' if use_validation else 'loss'
        self.direction = 'min'

    def get_value(self, logs):
        return logs.get(self.name, logs.get('loss'))
