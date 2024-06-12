from piaa.piaa import PIAA
from piaa.para_taskset import ParaTaskset
from piaa.sac_taskset import SACTaskset
import numpy as np

# Choose the task and input type.
task = "SAC" # "PARA" or "SAC"
input_type = "clip" # "clip" or "image"

# Load the PIAA model.
piaa_model = PIAA(dataset=task, input_type=input_type, method="knn", device="cuda", k_neighbors=5, knn_distance_func="default", knn_weighted=False)

# Load the taskset.
if task == "PARA":
    taskset = ParaTaskset(split="test", support_size=80, query_size=40, input_type=input_type, label_type="aestheticScore")
elif task == "SAC":
    taskset = SACTaskset(split="test", support_size=80, query_size=20, input_type=input_type)

# Pick a random user id and sample a task.
if task == "PARA":
    idx = np.random.randint(0, taskset.num_uid) # for PARA, it's called uid.
elif task == "SAC":
    idx = np.random.randint(0, taskset.num_sid) # for SAC, it's called sid.
sample = taskset.sample(idx)

def compute_mae(piaa_model, sample):
    support_x, support_y, query_x, query_y = sample
    piaa_model.adapt(support_x, support_y, learning_rate=1e-4, steps=100)
    y_pred = piaa_model.predict(query_x)
    y_pred = y_pred.cpu().detach().numpy()
    mae = np.abs(query_y-y_pred).mean()
    return mae

for k_neighbors in [3, 5, 10]:
    for distance_func in ["default", "euclidean"]:
        for weighted in [False, True]:
            piaa_model.set_knn_params(k_neighbors=k_neighbors, distance_func=distance_func, weighted=weighted)
            mae = compute_mae(piaa_model, sample)
            print(f"k_neighbors: {k_neighbors}, distance_func: {distance_func}, weighted: {weighted}, MAE: {mae}")

