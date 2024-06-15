from piaa.piaa import PIAA
from piaa.para_taskset import ParaTaskset
from piaa.sac_taskset import SACTaskset
import scipy.stats
import numpy as np
from tqdm import tqdm, trange
from sweetdebug import sweetdebug
sweetdebug()

# Choose the task and input type.
task = "SAC" # "PARA" or "SAC"
input_type = "clip" # "clip" or "image"

# Load the PIAA model.
knn_model = PIAA(dataset=task, input_type=input_type, method="knn", device="cuda", k_neighbors=5, knn_distance_func="default", knn_weighted=False)
maml_model = PIAA(dataset=task, input_type=input_type, method="maml", device="cuda")
finetune_model = PIAA(dataset=task, input_type=input_type, method="finetune", device="cuda")

# Load the taskset.
if task == "PARA":
    taskset = ParaTaskset(split="test", support_size=10, query_size=-1, input_type=input_type, label_type="aestheticScore")
elif task == "SAC":
    taskset = SACTaskset(split="test", support_size=100, query_size=100, input_type=input_type)

# Pick a random user id and sample a task.
if task == "PARA":
    idx = np.random.randint(0, taskset.num_uid) # for PARA, it's called uid.
elif task == "SAC":
    idx = np.random.randint(0, taskset.num_sid) # for SAC, it's called sid.
sample = taskset.sample(idx)

# def compute_mae(piaa_model, sample):
#     support_x, support_y, query_x, query_y = sample
#     piaa_model.adapt(support_x, support_y, learning_rate=1e-4, steps=100)
#     y_pred = piaa_model.predict(query_x)
#     y_pred = y_pred.cpu().detach().numpy()
#     mae = np.abs(query_y-y_pred).mean()
#     return mae

def compute_metric(piaa_model, sample):
    support_x, support_y, query_x, query_y = sample
    # giaa.
    piaa_model.adapt(support_x, support_y, learning_rate=1e-4, steps=0)
    y_pred = piaa_model.predict(query_x)
    y_pred = y_pred.cpu().detach().numpy()
    srocc = scipy.stats.spearmanr(query_y, y_pred)[0]
    mse = ((query_y-y_pred)**2).mean()
    mae = np.abs(query_y-y_pred).mean()
    return srocc, mse, mae

# # MAML model test.
# metric_result = []
# for i in trange(taskset.num_sid):
#     sample_result = []
#     for r in range(1):
#         sample = taskset.sample(i)
#         sample_metric = compute_metric(maml_model, sample)
#         sample_result.append(sample_metric)
#     metric = np.mean(sample_result, axis=0)
# metric_result.append(metric)
# srocc, mse, mae = np.mean(metric_result, axis=0)
# print(f"MAML, SROCC: {srocc}, MSE: {mse}, MAE: {mae}")

# # Finetune model test.
# metric_result = []
# for i in trange(taskset.num_sid):
#     sample_result = []
#     for r in range(1):
#         sample = taskset.sample(i)
#         sample_metric = compute_metric(finetune_model, sample)
#         sample_result.append(sample_metric)
#     metric = np.mean(sample_result, axis=0)
# metric_result.append(metric)
# srocc, mse, mae = np.mean(metric_result, axis=0)
# print(f"finetune, SROCC: {srocc}, MSE: {mse}, MAE: {mae}")

# # GIAA model test.
# metric_result = []
# for i in trange(taskset.num_sid):
#     sample_result = []
#     for r in range(1):
#         sample = taskset.sample(i)
#         sample_metric = compute_metric(finetune_model, sample)
#         sample_result.append(sample_metric)
#     metric = np.mean(sample_result, axis=0)
# metric_result.append(metric)
# srocc, mse, mae = np.mean(metric_result, axis=0)
# print(f"finetune, SROCC: {srocc}, MSE: {mse}, MAE: {mae}")

# kNN model test.
metric_result = []
repeat = 10
kNN_result = np.zeros([taskset.num_sid, repeat, 3])
giaa_result = np.zeros([taskset.num_sid, repeat, 3])

for i in trange(taskset.num_sid):
    sample_result = []
    for r in range(repeat):
        sample = taskset.sample(i)
        knn_model.set_knn_params(k_neighbors=100, distance_func="default", weighted=True)
        knn_metric = compute_metric(knn_model, sample)
        giaa_metric = compute_metric(finetune_model, sample)
        kNN_result[i, r] = knn_metric
        giaa_result[i, r] = giaa_metric

# save results.
np.save("data/kNN_result.npy", kNN_result)
np.save("data/giaa_result.npy", giaa_result)

# plot three subplots each for srocc, mse, mae.
# use bar plot. each row of kNN_result and giaa_result is each bar.
# x-axis is the index of sid.
# y-axis is the value of metric.
# error bar is 95% confidence interval.
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
for i, metric_name in enumerate(["SROCC", "MSE", "MAE"]):
    ax = axs[i]
    kNN_metric = kNN_result[:, :, i]
    giaa_metric = giaa_result[:, :, i]
    kNN_mean = kNN_metric.mean(axis=1)
    kNN_std = kNN_metric.std(axis=1)
    giaa_mean = giaa_metric.mean(axis=1)
    giaa_std = giaa_metric.std(axis=1)
    ax.bar(range(taskset.num_sid), kNN_mean, yerr=1.96*kNN_std/np.sqrt(repeat), label="PIAA", alpha=0.7)
    ax.bar(range(taskset.num_sid), giaa_mean, yerr=1.96*giaa_std/np.sqrt(repeat), label="GIAA", alpha=0.7)
    ax.set_title(metric_name)
    ax.legend()
plt.tight_layout()

# Save the plot.
plt.savefig("data/plot.png")