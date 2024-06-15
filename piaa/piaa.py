import torch
import torch.nn as nn
import torch.optim as optim
from .score_predictor import ScorePredictor
from torch.nn.functional import pairwise_distance
import learn2learn as l2l
import clip
from tqdm import trange
import torch.nn.functional as F

class PIAA:
    def __init__(self, dataset="PARA", input_type="clip", method="maml", device="cuda", loss="MAE", k_neighbors=5, knn_distance_func="default", knn_weighted=False):
        """
        Initialize the PIAA model with a specific dataset and checkpoint.

        Args:
        - dataset (set): A set containing task names (e.g., {"SAC", "PARA"}).
        - input_type (str): The type of input to use for adaptation.
        - method (str): The method to use for adaptation.
        - device (str): The device to use for computation.
        - loss (str): The loss function to use for adaptation.
        """
        self.dataset = dataset

        self.input_type = input_type
        if input_type == "image":
            self.feature_extractor, self.preprocess = clip.load("ViT-B/32", device=device)
        self.method = method
        self.device = device

        if method == "knn":
            self.model = KNNRegressor(k_neighbors=k_neighbors, distance_func=knn_distance_func, weighted=knn_weighted)
        else:
            self.model = self.load_model()
            
        if loss == "MAE":
            self.loss_fn = nn.L1Loss(reduction='mean') 
        elif loss == "MSE":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif loss == "MAE+MSE":
            self.loss_fn = lambda x, y: nn.L1Loss(reduction='mean')(x, y) + nn.MSELoss(reduction='mean')(x, y)

        return 
    
    def load_model(self):
        """
        Load the model from a checkpoint file.

        Returns:
        - model (nn.Module): The loaded model.
        """
        if self.method == "maml" and self.dataset == "PARA":
            model = ScorePredictor(width=1024, depth=4)
            model = l2l.algorithms.MAML(model, lr=1e-4, first_order=True)
            ckpt_fname = "para_maml.pt"
            # ckpt_fname = "ckpt/maml_para_MAE+MSE_20240530-052715/epoch_017_eval_0.463.pt"
        elif self.method == "maml" and self.dataset == "SAC":
            model = ScorePredictor(width=1024, depth=4)
            model = l2l.algorithms.MAML(model, lr=1e-4, first_order=True)
            ckpt_fname = "sac_maml.pt"
        elif self.method == "finetune" and self.dataset == "PARA":
            model = ScorePredictor(width=2048, depth=4)
            ckpt_fname = "para_finetune.pt"
        elif self.method == "finetune" and self.dataset == "SAC":
            model = ScorePredictor(width=2048, depth=4)
            ckpt_fname = "sac_finetune.pt"

        self.ckpt_fname = "/workspace/Projects/CRC/piaa/ckpt/" + ckpt_fname
        state_dict = torch.load(self.ckpt_fname)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model
    
    def set_knn_params(self, k_neighbors=5, distance_func="default", weighted=False):
        assert self.method == "knn", "This method is only available for kNN."
        self.model = KNNRegressor(k_neighbors=k_neighbors, distance_func=distance_func, weighted=weighted)
    
    def adapt(self, x, y, learning_rate=1e-4, steps=10, reload_model=True):
        """
        Adapt the model to a specific task with x and y.

        Args:
        - x (list): The input x or clip features.
        - y (list): The target y.
        - learning_rate (float): The learning rate for adaptation.
        - steps (int): The number of steps for adaptation.
        - reload_model (bool): Whether to reload the model from the checkpoint. This is to isolate each subtask.
        """
        
        if self.method == "knn":
            x = self.extract_features(x)
            self.model.fit(x, torch.tensor(y).float().to(self.device))
        
        else:
            # Initialize the model.
            if reload_model:
                self.model = self.load_model()       
            self.model.train()

            # Prepare the learner.
            if self.method == "maml":
                self.model.lr = learning_rate
                self.learner = self.model.clone()
            elif self.method == "finetune":
                self.learner = self.model
                self.optim = torch.optim.Adam(self.learner.parameters(), lr=learning_rate)
                
            x = self.extract_features(x)
            y = torch.tensor(y).float().unsqueeze(1).to(self.device)

            pbar = trange(steps)
            if self.method == "maml": 
                for step in pbar:
                    scores = self.learner(x)
                    loss = self.loss_fn(scores, y)
                    self.learner.adapt(loss)
                    pbar.set_description(f"Loss for Support Set: {loss.item():.4f}")
            elif self.method == "finetune":
                for step in pbar:
                    self.optim.zero_grad()
                    scores = self.learner(x)
                    loss = self.loss_fn(scores, y)
                    loss.backward()
                    self.optim.step()
                    pbar.set_description(f"Loss for Support Set: {loss.item():.4f}")

            return
    
    def predict(self, x):
        """
        Predict the scores for the given images using the adapted model.

        Args:
        - images (torch.Tensor): The input images.

        Returns:
        - scores (torch.Tensor): The predicted scores.
        """
        x = self.extract_features(x)
        
        if self.method == "knn":
            return self.model.predict(x) 
        
        assert hasattr(self, "learner"), "The model has not been adapted yet."

        self.learner.eval()
        with torch.no_grad():
            scores = self.learner(x)
        return scores
    
    def extract_features(self, x):
        """
        Extract features from the input x.
        """
        if self.input_type == "image":
            x = torch.stack([self.preprocess(x_) for x_ in x])
            x = x.to(self.device)
            x = self.feature_extractor.encode_image(x).float()
        elif self.input_type == "clip":
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x).float().to(self.device)
            else:
                x = x.float().to(self.device)
        return x

class KNNRegressor:
    def __init__(self, k_neighbors=5, distance_func="default", weighted=False):
        self.k_neighbors = k_neighbors
        self.support_features = None
        self.support_labels = None
        self.weighted = weighted
        
        if distance_func == "default":
            self.distance_func = self.default_distance_func
        elif distance_func == "euclidean":
            self.distance_func = self.euclidean_distance
        else:
            raise ValueError("Distance function is either 'default' or 'euclidean'.")
        
    def default_distance_func(self, x_query, x_support):
        """
        Default distance function using cosine similarity.
        """
        cos_sim = F.cosine_similarity(x_query.unsqueeze(1), x_support.unsqueeze(0), dim=2)
        distances = torch.maximum(1 - cos_sim, torch.zeros_like(cos_sim))
        return distances
    
    def euclidean_distance(self, x_query, x_support):
        """
        Calculate the Euclidean distance between the query and support features.
        """
        # return pairwise_distance(x_query, x_support, p=2)
        diff = x_query.unsqueeze(1) - x_support.unsqueeze(0)
        distances = torch.norm(diff, p=2, dim=2)
        return distances

    def fit(self, x_support, y_support):
        """
        Store the support features and labels for kNN prediction.
        """
        self.support_features = x_support
        self.support_labels = y_support

    def predict(self, x_query):
        """
        Predict the labels for the query set based on stored support features.
        """
        distances = self.distance_func(x_query, self.support_features)
        knn_indices = distances.topk(self.k_neighbors, largest=False, dim=1).indices
        knn_scores = torch.gather(self.support_labels.expand_as(distances), 1, knn_indices)
        if self.weighted:
            # Gather the k nearest neighbors' distances
            knn_distances = torch.gather(distances, 1, knn_indices)
            # Calculate the weights based on the inverse of distances
            weights = 1 / (knn_distances + 1e-5)  # Adding a small constant to avoid division by zero
            weights /= weights.sum(dim=1, keepdim=True)  # Normalize weights
            # Weighted average of scores
            predicted_scores = (weights * knn_scores).sum(dim=1)
        else:
            # Standard average of scores
            predicted_scores = knn_scores.mean(dim=1)

        return predicted_scores

if __name__ == "__main__":
    from piaa.piaa.para_taskset import ParaDataset
    dataset = ParaDataset(load_emb=False)

    # (Meta) Train
    piaa = PIAA(dataset="para", feature_extractor_type="clip", method="maml", device="cuda")

    x, y = dataset[:30]
    piaa.adapt(x, y, steps=0)
    y_pred = piaa.predict(x)
    mae = nn.L1Loss(reduction='mean')(y_pred, torch.tensor(y).cuda()).item()
    print("0 step mae: ",mae)

    piaa.adapt(x, y, steps=20)
    y_pred = piaa.predict(x)
    mae = nn.L1Loss(reduction='mean')(y_pred, torch.tensor(y).cuda()).item()
    # mae = nn.L1Loss(reduction='mean')(y_pred, y).item()
    print("20 step mae: ",mae)


    # # Test
    # x_test, y_test = dataset[10:20]
    # y_pred = piaa.predict(x_test)

