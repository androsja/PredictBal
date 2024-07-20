# File: code/tuning/custom_tuner.py
from keras_tuner import Tuner
import tensorflow as tf

class CustomTuner(Tuner):
    def __init__(self, oracle, plot_saver, **kwargs):
        super(CustomTuner, self).__init__(oracle, **kwargs)
        self.plot_saver = plot_saver

    def run_trial(self, trial, x, y, epochs, callbacks=None):
        self.model = self.hypermodel.build(trial.hyperparameters)

        history = self.model.fit(
            x, y,
            epochs=epochs,
            callbacks=callbacks or []
        )

        logs = history.history

        # Asegurarse de que 'loss' esté presente y convertirlo a float si es necesario
        logs = {k: (v[-1] if isinstance(v, list) else v) for k, v in logs.items()}

        if 'loss' in logs:
            self.oracle.update_trial(
                trial.trial_id,
                {'loss': logs['loss']}
            )
        else:
            raise ValueError("The logs dictionary does not contain the key 'loss'")

        print(f"Trial {trial.trial_id} - Model built and trained.")
        return logs
