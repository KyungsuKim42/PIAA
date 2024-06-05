# Install

```bash
# Github에서 가져오거나
git pull https://github.com/KyungsuKim42/~~
# 아니면 디렉토리 전체 복사
cp -r /home/kyungsukim/git/piaa/ /my/directory/

cd /my/directory/
pip install .
```

# Example Usage

```bash
form piaa.piaa import PIAA
from piaa.para_taskset import ParaDataset
import numpy as np

piaa_model = PIAA(dataset="PARA", feature_extractor_type="clip", method="maml", device="cuda")
dataset = ParaDataset()

x, y = dataset[:30]
piaa.adapt(x, y, steps=10)
y_pred = piaa.predict(x)

y = np.array(y)
y_pred = y_pred.cpu().detach().numpy()
mae = np.abs(y-y_pred)
print(mae)
```