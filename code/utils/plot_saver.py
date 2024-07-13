import matplotlib.pyplot as plt
import os


class PlotSaver:
    def __init__(self, base_dir, project_name):
        self.base_dir = base_dir
        self.project_name = project_name

    def save_plot(self, trial_id, y_true, y_pred):
        trial_dir = os.path.join(self.base_dir, self.project_name, f'trial_{trial_id}')
        if not os.path.exists(trial_dir):
            os.makedirs(trial_dir)
        
        plt.figure(figsize=(14, 7))
        plt.plot(y_true, label='True Value')
        plt.plot(y_pred, label='Predicted Value', linestyle='--', color='yellow', marker='o', markersize=2, linewidth=2)
        plt.title(f'Validation Prediction after Trial {trial_id}')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        plot_filename = os.path.join(trial_dir, f'trial_{trial_id}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved plot for trial {trial_id} at {plot_filename}")