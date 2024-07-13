# File: code/model/model_builder_five.py
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Input
from keras.optimizers import Adam
import tensorflow as tf

class ModelBuilderFive:
    def __init__(self, time_step, input_dim):
        self.time_step = time_step
        self.input_dim = input_dim

    def build_model(self, hp):
        model = Sequential()
        model.add(Input(shape=(self.time_step, self.input_dim)))
        num_layers = hp.Int('num_layers', 10, 15)
        for i in range(num_layers):
            units = hp.Int('units', 32, 128, step=32)
            return_sequences = i < num_layers - 1
            model.add(LSTM(units=units, return_sequences=return_sequences))
            model.add(Dropout(hp.Float('dropout', 0.1, 0.3, step=0.1)))

        model.add(Dense(5, activation='linear'))  # Asegurar que la salida tenga 5 valores

        learning_rate = hp.Float('learning_rate', 1e-3, 1e-2, sampling='LOG')
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse'  # Usar Mean Squared Error
        )

        hp.Int('batch_size', 32, 128, step=32)

        return model
