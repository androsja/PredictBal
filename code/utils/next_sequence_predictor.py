import numpy as np

class NextSequencePredictor:
    def __init__(self, model, scalers, time_step):
        self.model = model
        self.scalers = scalers  # Ahora es un diccionario de escaladores
        self.time_step = time_step

    def predict_next_sequence(self, input_sequence):
        # Asegurarse de que la secuencia de entrada tiene el tamaño adecuado para el time_step
        if len(input_sequence) != self.time_step:
            raise ValueError(f"La secuencia de entrada debe tener {self.time_step} pasos de tiempo.")
        
        # Escalar la secuencia de entrada usando cada escalador correspondiente
        input_sequence_scaled = np.zeros_like(input_sequence)
        for i in range(input_sequence.shape[1]):
            input_sequence_scaled[:, i] = self.scalers[i].transform(input_sequence[:, i].reshape(-1, 1)).flatten()
        
        # Asegurarse de que la entrada tiene la forma correcta
        input_sequence_scaled = input_sequence_scaled.reshape(1, self.time_step, input_sequence_scaled.shape[1])
        
        # Hacer la predicción
        prediction_scaled = self.model.predict(input_sequence_scaled)
        
        # Invertir la escala de la predicción
        prediction = np.zeros_like(prediction_scaled)
        for i in range(prediction_scaled.shape[1]):
            prediction[:, i] = self.scalers[i].inverse_transform(prediction_scaled[:, i].reshape(-1, 1)).flatten()
        
        return prediction.flatten().tolist()

    def print_prediction(self, input_sequence):
        next_sequence = self.predict_next_sequence(input_sequence)
        print(f"Predicted next sequence: {next_sequence}")
