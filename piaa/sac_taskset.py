from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from PIL import Image


class SACTaskset():
    def __init__(self, split="train", support_size=80, query_size=20, data_path="/disk2/SAC/", input_type="clip"):
        df = pd.read_csv('/disk2/SAC/metadata.csv')
        
        assert split in ["train", "test"]
        sids = pd.read_csv(f'{data_path}/{split}_sids.csv')
        # self.sid_list = list(sids.columns)
        self.sid_list = [int(i) for i in sids.columns]
        self.num_sid = len(self.sid_list)
        
        self.support_size = support_size
        self.query_size = query_size
        self.input_type = input_type
        
        self.df = df[df['sid'].isin(self.sid_list)]
        self.num_samples = self.df.shape[0]
        
    def sample(self, sid_index):
        sid = self.sid_list[sid_index]
        sid_df = self.df[self.df['sid'] == sid]
        num_samples = sid_df.shape[0]
        
        # adjust support, query size for small number of samples
        if num_samples >= self.support_size + self.query_size:
            support_size = self.support_size
            query_size = self.query_size
        else:
            support_size = int(num_samples * self.support_size / (self.support_size + self.query_size))
            query_size = num_samples - support_size
        
        random_indices = np.random.permutation(num_samples)
        support_indices = random_indices[:support_size]
        query_indices = random_indices[support_size:support_size+query_size]
        
        # support_iids = sid_df.iloc[support_indices]['iid'].values
        support_img_names = sid_df.iloc[support_indices]['path'].values
        if self.input_type == "clip":
            support_x = self.load_embedding(support_img_names)
        elif self.input_type == "image":
            support_x = self.load_image(support_img_names)
        support_y = sid_df.iloc[support_indices]['rating'].values
        
        # query_iids = sid_df.iloc[query_indices]['iid'].values
        query_img_names = sid_df.iloc[query_indices]['path'].values
        if self.input_type == "clip":
            query_x = self.load_embedding(query_img_names)
        elif self.input_type == "image":
            query_x = self.load_image(query_img_names)
        query_y = sid_df.iloc[query_indices]['rating'].values

        return support_x, support_y, query_x, query_y
    
    def load_embedding(self, img_names):
        emb = np.zeros([len(img_names), 512])
        for i, img_name in enumerate(img_names):
            emb[i] = np.load(f"/disk2/SAC/clip_features/{img_name[:-4]}.npy")
        return emb
    
    def load_image(self, img_names):
        images = []
        for img_name in img_names:
            images.append(Image.open(f"/disk2/SAC/images/{img_name}"))
        return images

class SACDataset(Dataset):
    def __init__(self, split="train", data_path='/disk2/SAC/', load_emb=True):
        df = pd.read_csv('/disk2/SAC/metadata.csv')
        self.load_emb = load_emb
        
        assert split in ["train", "test"]
        sids = pd.read_csv(f'{data_path}/{split}_sids.csv')
        # self.sid_list = list(sids.columns)
        self.sid_list = [int(i) for i in sids.columns]
        self.num_sid = len(self.sid_list)
        
        self.df = df[df['sid'].isin(self.sid_list)]
        self.num_samples = self.df.shape[0]

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
        img_name = self.df.iloc[idx]['path']
        if self.load_emb:
            emb_fname = img_name[:-4] + '.npy'
            x = np.load(f"/disk2/SAC/clip_features/{emb_fname}")
        else:
            x = Image.open(f"/disk2/SAC/images/{img_name}")
        y = float(self.df.iloc[idx]['rating'])
        return x, y

    def __len__(self):
        return self.df.shape[0]


def main():
    taskset = SACTaskset()
    dataset = SACDataset()

    # Iterate over all samples in taskset
    for sid_index in tqdm(range(taskset.num_sid), desc="Taskset Progress"):
        try:
            support_x, support_y, query_x, query_y = taskset.sample(sid_index)
        except Exception as e:
            print(f"Error in sid {sid_index}: {e}")

    # Iterate over all samples in dataset
    for idx in tqdm(range(len(dataset)), desc="Dataset Progress"):
        try:
            x, y = dataset[idx]
        except Exception as e:
            print(f"Error in idx {idx}: {e}")
        


if __name__ == "__main__":
    main()