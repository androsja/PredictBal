import pandas as pd

class LotteryData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_and_prepare_data(self):
        # Leer el archivo de texto
        self.data = pd.read_csv(self.file_path, sep=' - ', header=None, engine='python')
        
        # Nombrar las columnas
        self.data.columns = ['Num1', 'Num2', 'Num3', 'Num4', 'Num5', 'SpecialNum']
        
        # Convertir los valores a enteros
        self.data = self.data.astype(int)
        
        # Verificar duplicados
        self.data = self.data.drop_duplicates()

    def get_data(self):
        if self.data is None:
            self.load_and_prepare_data()
        return self.data 