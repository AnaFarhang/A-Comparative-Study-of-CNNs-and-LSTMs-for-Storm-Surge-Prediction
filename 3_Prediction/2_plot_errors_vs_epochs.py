import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# === Load your data ===
train1 = np.load('train1.npy')
train2 = np.load('train2.npy')
train3 = np.load('train3.npy')

test1 = np.load('test1.npy')
test2 = np.load('test2.npy')
test3 = np.load('test3.npy')

epochs = np.arange(1, train1.shape[1])

# === Set Seaborn style (same as your Excel example) ===
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
sns.set_palette("deep")  # same palette Seaborn uses by default for hue='model'

font = {'family': 'Times New Roman', 'size': 10}


# === Define reusable plotting function ===
def plot_metric(index, ylabel, filename, models):
    plt.figure(figsize=(6, 4))
    
    for arr, label in models:
        data = np.mean(arr[:, 1:, index], axis=0)
        sns.lineplot(
            x=epochs,
            y=data,
            label=label,
            marker='o',           # same as your Excel example
            linewidth=1.5
        )

    plt.title(f"{ylabel} Versus Epochs", fontsize=14, fontweight='bold')
    plt.xlabel("Epochs", fontsize=13)
    plt.ylabel(ylabel, fontsize=13)
    plt.tick_params(axis='both', labelsize=11)
    plt.legend(fontsize=10, title_fontsize=11, loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# === Define models with labels ===
models = [
    (train1, "CNN-LSTM"),
    (train2, "LSTM"),
    (train3, "3DCNN"),
]


plot_metric(2, "Train_MAE", "Train_MAE.pdf", [(m, f"{l} Training MAE") for m, l in models])
plot_metric(3, "Validation_MAE", "Val_MAE.pdf",   [(m, f"{l} Validation MAE") for m, l in models])
plot_metric(4, "Train_RMSE", "Train_RMSE.pdf",[(m, f"{l} Training RMSE") for m, l in models])
plot_metric(5, "Validation_RMSE", "Val_RMSE.pdf",  [(m, f"{l} Validation RMSE") for m, l in models])
plot_metric(6, "Train_MSE", "Train_MSE.pdf", [(m, f"{l} Training MSE") for m, l in models])
plot_metric(7, "Validation_MSE", "Val_MSE.pdf",   [(m, f"{l} Validation MSE") for m, l in models])