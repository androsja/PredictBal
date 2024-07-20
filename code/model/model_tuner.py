from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import pickle
import tensorflow as tf
from keras.callbacks import EarlyStopping
from custom_tuner import CustomTuner
from model_manager import ModelManager
from model_builder_factory import ModelBuilderFactory
from tensorflow.keras import backend as K
from utils.plot_saver import PlotSaver
from tuning.custom_objective import CustomObjective

class ModelTuner:
    def __init__(self, X_train, y_train, time_step, input_dim, output_dim, drive_dir, project_name, folder_name):
        self.X_train = X_train
        self.y_train = y_train
        self.time_step = time_step
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.drive_dir = drive_dir
        self.project_name = project_name
        self.folder_name = folder_name
        self.model_path = os.path.join(self.drive_dir, self.folder_name, f'model_{self.project_name}.h5')
        self.hyperparameters_path = os.path.join(self.drive_dir, self.folder_name, f'hyperparameters_{self.project_name}.pkl')
        self.tuner = None

    def tune_model(self, oracle, max_epochs=100):
        print("-------------tune_model-------------")
        model_builder = ModelBuilderFactory.create(self.time_step, self.input_dim, self.output_dim)
        objective = CustomObjective(use_validation=False)

        self.tuner = CustomTuner(
            oracle=oracle,
            hypermodel=model_builder.build_model,
            directory=os.path.join(self.drive_dir, self.folder_name),
            project_name=self.project_name,
            overwrite=False,
            plot_saver=PlotSaver(self.drive_dir, self.project_name)
        )
        print("-------------EarlyStopping-------------")
        callbacks = [EarlyStopping(monitor='loss', patience=3)]

        self.tuner.search(
            x=self.X_train, y=self.y_train,
            epochs=max_epochs,
            callbacks=callbacks
        )
        print("-------------get_best_trials-------------")
        best_trial = oracle.get_best_trials(num_trials=1)[0]
        best_hps = best_trial.hyperparameters

        with open(self.hyperparameters_path, 'wb') as f:
            pickle.dump(best_hps.values, f)

        print("-------------hypermodel-------------")
        model = self.tuner.hypermodel.build(best_hps)
        print("-------------EarlyStopping-------------")
        es = EarlyStopping(monitor='loss', patience=3)
        history = model.fit(self.X_train, self.y_train, epochs=max_epochs, batch_size=best_hps.get('batch_size'), callbacks=[es])
        print("-------------print_model_summary-------------")
        self.print_model_summary(model, best_hps)
        print("-------------print_evaluation_metrics-------------")
        self.print_evaluation_metrics(history, self.X_train, self.y_train, model)

        print("-------------save_model_and_hyperparameters-------------")
        ModelManager.save_model_and_hyperparameters(model, best_hps, self.model_path, self.hyperparameters_path)

        K.clear_session()

        return model, best_hps



    def print_model_summary(self, model, best_hps):
        print("Model summary:")
        model.summary()
        print(f"Optimal hyperparameters: Units: {best_hps.get('units')}, Layers: {best_hps.get('num_layers')}, Learning rate: {best_hps.get('learning_rate')}, Batch size: {best_hps.get('batch_size')}")

    def print_evaluation_metrics(self, history, X_train, y_train, model):
        y_pred_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        r2_train = tf.keras.metrics.R2Score()
        r2_train.update_state(y_train.reshape(-1, 1), y_pred_train.reshape(-1, 1))
        r2_train_result = r2_train.result().numpy()

        print(f"Evaluación en datos de entrenamiento: MSE = {mse_train:.4f}, MAE = {mae_train:.4f}, R² = {r2_train_result:.4f}")

        # Check if 'val_loss' is in history
        if 'val_loss' in history.history:
            val_loss = history.history['val_loss'][-1]
            print(f"Pérdida en datos de validación: {val_loss:.4f}")
        else:
            print("No se realizaron evaluaciones en datos de validación.")


