# File: code/model/model_builder_factory_five.py
from model.model_builder_five import ModelBuilderFive

class ModelBuilderFactoryFive:
    @staticmethod
    def create(time_step, input_dim):
        return ModelBuilderFive(time_step, input_dim)
