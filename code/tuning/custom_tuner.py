# File: code/tuning/custom_tuner.py
from keras_tuner import Tuner
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

class CustomTuner(Tuner):
    def __init__(self, oracle, X_val, y_val, plot_saver, **kwargs):
        super(CustomTuner, self).__init__(oracle, **kwargs)
        self.X_val = X_val
        self.y_val = y_val
        self.plot_saver = plot_saver

    def run_trial(self, trial, x, y, epochs, validation_data=None, callbacks=None):
        self.model = self.hypermodel.build(trial.hyperparameters)

        history = self.model.fit(
            x, y,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks or []
        )
        
        y_pred = self.model.predict(self.X_val)
        trial_id = trial.trial_id
        self.plot_saver.save_plot(trial_id, self.y_val, y_pred)
        
        print(f"Trial {trial_id} - Model built and trained.")
        return history
