from piaa.piaa import PIAA
from piaa.para_taskset import ParaTaskset
from piaa.sac_taskset import SACTaskset
import numpy as np

# Choose the task and input type.
task = "SAC" # "PARA" or "SAC"
input_type = "clip" # "clip" or "image"

# Load the PIAA model.
piaa_model = PIAA(dataset=task, input_type=input_type, method="maml", device="cuda")

# Load the taskset.
if task == "PARA":
    taskset = ParaTaskset(split="test", support_size=80, query_size=40, input_type=input_type, label_type="aestheticScore")
elif task == "SAC":
    taskset = SACTaskset(split="test", support_size=80, query_size=40, input_type=input_type)

# Pick a random user id and sample a task.
if task == "PARA":
    idx = np.random.randint(0, taskset.num_uid) # for PARA, it's called uid.
elif task == "SAC":
    idx = np.random.randint(0, taskset.num_sid) # for SAC, it's called sid.
support_x, support_y, query_x, query_y = taskset.sample(idx)

# Adapt the model to the user.
piaa_model.adapt(support_x, support_y, learning_rate=1e-4, steps=100)

# Predict the scores for the query set.
y_pred = piaa_model.predict(query_x)

# Calculate the mean absolute error.
y_pred = y_pred.cpu().detach().numpy()
mae = np.abs(query_y-y_pred).mean()
print(mae)