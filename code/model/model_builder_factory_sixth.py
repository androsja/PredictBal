# File: code/model/model_builder_factory_sixth.py
from model.model_builder_sixth import ModelBuilderSixth

class ModelBuilderFactorySixth:
    @staticmethod
    def create(time_step, input_dim):
        return ModelBuilderSixth(time_step, input_dim)
