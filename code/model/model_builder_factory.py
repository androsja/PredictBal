# File: code/model/model_builder_factory.py
from model_builder import ModelBuilder

class ModelBuilderFactory:
    @staticmethod
    def create(time_step, input_dim, output_dim, is_sixth_column=False):
        return ModelBuilder(time_step, input_dim, output_dim, is_sixth_column)

