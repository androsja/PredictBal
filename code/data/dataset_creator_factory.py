import os
import numpy as np
import pandas as pd
from dataset_creator import DatasetCreator

class DatasetCreatorFactory:
    def __init__(self):
        self.file_path = "historic_data/bal_results_i.txt"

    def create_dataset_creator(self, time_step):
        data = self.read_file_contents()
        df = self.convert_content_to_dataframe(data)
        print("Data for bal results getted")
        return DatasetCreator(df, time_step)

    def read_file_contents(self):
        with open(self.file_path, 'r') as file:
            content = file.readlines()
        return content

    def convert_content_to_dataframe(self, content):
        rows = [list(map(int, line.strip().split(' - '))) for line in content]
        df = pd.DataFrame(rows)
        return df
