import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sweetdebug import sweetdebug
sweetdebug()

# Example data
taskset = type('TaskSet', (object,), {'num_sid': 30})()
repeat = 10
kNN_result = np.load("data/kNN_result.npy")
giaa_result = np.load("data/giaa_result.npy")

sns.set_theme()
fig, axs = plt.subplots(3, 1, figsize=(15, 15))

# Adjust the number of bars per group
num_bars = 2  # Number of models (PIAA and GIAA)

for i, metric_name in enumerate(["SROCC", "MSE", "MAE"]):
    ax = axs[i]
    kNN_metric = kNN_result[:, :, i]
    giaa_metric = giaa_result[:, :, i]
    
    kNN_mean = kNN_metric.mean(axis=1)
    kNN_std = kNN_metric.std(axis=1)
    giaa_mean = giaa_metric.mean(axis=1)
    giaa_std = giaa_metric.std(axis=1)
    
    x = np.arange(taskset.num_sid)
    width = 0.8 / num_bars  # the width of the bars
    
    # Plotting grouped bars
    ax.bar(x - width/2, kNN_mean, width, label="PIAA", alpha=0.7, align='center')
    ax.bar(x + width/2, giaa_mean, width, label="GIAA", alpha=0.7, align='center')

    ax.set_xticks(np.arange(min(x), max(x)+1, 1.0))
    
    ax.set_title(metric_name)
    ax.set_xlabel('Test workers’ ID')
    ax.set_ylabel(metric_name)
    ax.legend()

plt.tight_layout()
plt.savefig("data/plot.png")
