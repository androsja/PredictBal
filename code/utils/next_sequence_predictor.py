# File: code/utils/next_sequence_predictor.py
import numpy as np

class NextSequencePredictor:
    def __init__(self, model, scaler, time_step):
        self.model = model
        self.scaler = scaler
        self.time_step = time_step

    def predict_next_sequence(self, input_sequence):
        # Asegurarse de que la secuencia de entrada tiene el tamaño adecuado para el time_step
        if len(input_sequence) != self.time_step:
            raise ValueError(f"La secuencia de entrada debe tener {self.time_step} pasos de tiempo.")
        
        # Escalar la secuencia de entrada
        input_sequence_scaled = self.scaler.transform(input_sequence)
        
        # Asegurarse de que la entrada tiene la forma correcta
        input_sequence_scaled = input_sequence_scaled.reshape(1, self.time_step, input_sequence_scaled.shape[1])
        
        # Hacer la predicción
        prediction_scaled = self.model.predict(input_sequence_scaled)
        
        # Invertir la escala de la predicción
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        return prediction.flatten().tolist()

    def print_prediction(self, input_sequence):
        next_sequence = self.predict_next_sequence(input_sequence)
        print(f"Predicted next sequence: {next_sequence}")
