from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv
from PIL import Image


class ParaTaskset():
    def __init__(self, split="train", support_size=80, query_size=40, label_type="aestheticScore", input_type="clip", make_uid_list=False):
        df = pd.read_csv('/disk2/PARA/annotation/PARA-Images.csv')
        self.df = df
        if make_uid_list:
            uid_list = self.make_uid_list()
        else:
            uid_list = self.load_uid_list()

        self.uid_list = uid_list
        self.support_size = support_size
        self.query_size = query_size
        self.label_type = label_type
        self.input_type = input_type
        if split == "train":
            self.uid_list = uid_list[:398]
            self.num_uid = 398
        elif split == "test":
            self.uid_list = uid_list[398:]
            self.num_uid = 40

        self.df = df[df['userId'].isin(self.uid_list)]
        self.num_samples = self.df.shape[0] 
    
    def sample(self, uid, stochastic=True):
        uid = self.uid_list[uid]
        uid_df = self.df[self.df['userId'] == uid]
        num_samples = uid_df.shape[0]
        if self.query_size > 0:
            assert num_samples >= self.support_size+self.query_size, "Not enough samples for support and query"
        else:
            assert num_samples >= self.support_size, "Not enough samples for support and query"
        random_indices = np.random.permutation(num_samples)
        support_indices = random_indices[:self.support_size]
        if self.query_size > 0:
            query_indices = random_indices[self.support_size:min(self.support_size+self.query_size, num_samples)]
        else:
            query_indices = random_indices[self.support_size:]

        support_session_ids = uid_df.iloc[support_indices]['sessionId'].values
        support_img_names = uid_df.iloc[support_indices]['imageName'].values
        if self.input_type == "clip":
            support_x = self.load_embedding(support_session_ids, support_img_names)
        elif self.input_type == "image":
            support_x = self.load_image(support_session_ids, support_img_names)
        support_y = uid_df.iloc[support_indices][self.label_type].values

        query_session_ids = uid_df.iloc[query_indices]['sessionId'].values
        query_img_names = uid_df.iloc[query_indices]['imageName'].values
        if self.input_type == "clip":
            query_x = self.load_embedding(query_session_ids, query_img_names)
        elif self.input_type == "image":
            query_x = self.load_image(query_session_ids, query_img_names)
        query_y = uid_df.iloc[query_indices][self.label_type].values

        return support_x, support_y, query_x, query_y
    
    def load_embedding(self, session_ids, support_fnames):
        emb = np.zeros([len(support_fnames), 512])
        for i, (session_id,fname) in enumerate(zip(session_ids, support_fnames)):
            fname = fname.replace(".jpg", ".npy")
            emb[i] = np.load(f"/disk2/PARA/imgs/{session_id}/{fname.replace('.jpg','.npy')}")
        return emb
    
    def load_image(self, session_ids, support_fnames):
        imgs = []
        for session_id, fname in zip(session_ids, support_fnames):
            imgs.append(Image.open(f"/disk2/PARA/imgs/{session_id}/{fname}"))
        return imgs

    def make_uid_list(self):
        df = self.df
        uid_list = sorted(df['userId'].unique())
        small_uid = []
        large_uid = []
        for uid in uid_list:
            uid_df = df[df['userId'] == uid]
            num_samples = uid_df.shape[0]
            if num_samples == 70:
                small_uid.append(uid)
            else:
                large_uid.append(uid)
        uid_list = small_uid + large_uid
        return uid_list

    def load_uid_list(self):
        with open("/home/kyungsukim/git/piaa/data/uid_list.csv", "r") as f:
            reader = csv.reader(f)
            uid_list = list(reader)[0]
        return uid_list

class ParaDataset(Dataset):
    def __init__(self, split="train", load_emb=True):
        df = pd.read_csv('/disk2/PARA/annotation/PARA-Images.csv')
        uid_list = self.load_uid_list()
        if split == "train":
            uid_list = uid_list[:398]
        elif split == "test":
            uid_list = uid_list[398:]
        self.df = df[df['userId'].isin(uid_list)]
        self.load_emb = load_emb
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            x_list, y_list = [], []
            for i in range(*idx.indices(len(self))):
                x, y = self.get_item(i)
                x_list.append(x)
                y_list.append(y)
            return x_list, y_list
        else:
            return self.get_item(idx)
    
    def get_item(self, idx):
        session_id = self.df.iloc[idx]['sessionId']
        if self.load_emb:
            fname = self.df.iloc[idx]['imageName'].replace(".jpg", ".npy")
            x = np.load(f"/disk2/PARA/imgs/{session_id}/{fname}")
        else:
            fname = self.df.iloc[idx]['imageName']
            x = Image.open(f"/disk2/PARA/imgs/{session_id}/{fname}")
        y = self.df.iloc[idx]['aestheticScore']
        return x, y

    def __len__(self):
        return self.df.shape[0]

    def load_uid_list(self):
        with open("/home/kyungsukim/git/piaa/data/uid_list.csv", "r") as f:
            reader = csv.reader(f)
            uid_list = list(reader)[0]
        return uid_list
        


def main():
    taskset = ParaTaskset("train", support_size=80, query_size=40)
    for epoch in range(100):
        uid_list = np.random.permutation(taskset.num_uid)
        for uid in uid_list:
            support_x, support_y, query_x, query_y = taskset.sample(uid, stochastic=True)
            pass
    for task_batch in taskset:
        (support_x, support_y), (query_x, query_y) = task_batch



if __name__ == "__main__":
    main()