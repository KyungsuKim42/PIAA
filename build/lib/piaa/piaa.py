import torch
import torch.nn as nn
import torch.optim as optim
from .score_predictor import ScorePredictor
import learn2learn as l2l
import clip

class PIAA:
    def __init__(self, dataset="PARA", feature_extractor_type="clip", method="maml", device="cuda"):
        """
        Initialize the PIAA model with a specific dataset and checkpoint.

        Args:
        - dataset (set): A set containing task names (e.g., {"SAC", "PARA"}).
        - feature_extractor_type (str): The type of feature extractor to use.
        - method (str): The method to use for adaptation.
        - device (str): The device to use for computation.
        """
        self.dataset = dataset

        self.feature_extractor_type = feature_extractor_type
        if feature_extractor_type == "clip":
            self.feature_extractor, self.preprocess = clip.load("ViT-B/32", device=device)
        self.method = method
        self.device = device

        self.feature_extractor_type = feature_extractor_type

        self.model = self.load_model()
        self.loss_fn = lambda x, y: nn.L1Loss(reduction='mean')(x, y) + nn.MSELoss(reduction='mean')(x, y)

        return 
    
    def load_model(self):
        """
        Load the model from a checkpoint file.

        Returns:
        - model (nn.Module): The loaded model.
        """
        if self.method == "maml":
            model = ScorePredictor(width=1024, depth=4)
            model = l2l.algorithms.MAML(model, lr=1e-4, first_order=True)
            ckpt_fname = "maml.pt"
            # ckpt_fname = "ckpt/maml_para_MAE+MSE_20240530-052715/epoch_017_eval_0.463.pt"
        elif self.method == "finetune":
            model = ScorePredictor(width=2048, depth=4)
            ckpt_fname = "finetune.pt"

        self.ckpt_fname = "/home/kyungsukim/git/piaa/ckpt/" + ckpt_fname
        model.load_state_dict(torch.load(self.ckpt_fname))
        model.to(self.device)
        model.eval()

        return model
    
    def adapt(self, images, labels, learning_rate=1e-3, steps=10, reload_model=True):
        """
        Adapt the model to a specific task with images and labels.

        Args:
        - images list of (PIL Images): The input images.
        - labels list of float numbers: The corresponding labels. shape = (batch_size, 1)
        - learning_rate (float): The learning rate for adaptation.
        - steps (int): The number of steps for adaptation.
        - reload_model (bool): Whether to reload the model from the checkpoint.
        """
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
        
        # Prepare the feature.
        if self.feature_extractor_type == "clip":
            images = torch.stack([self.preprocess(image) for image in images])
            images = images.to(self.device)
            image_features = self.feature_extractor.encode_image(images).float()
        labels = torch.tensor(labels).float().unsqueeze(1).to(self.device)

        if self.method == "maml": 
            for step in range(steps):
                scores = self.learner(image_features)
                loss = self.loss_fn(scores, labels)
                self.learner.adapt(loss)
        elif self.method == "finetune":
            for step in range(steps):
                self.optim.zero_grad()
                scores = self.learner(image_features)
                loss = self.loss_fn(scores, labels)
                loss.backward()
                self.optim.step()

        return
    
    def predict(self, images):
        """
        Predict the scores for the given images using the adapted model.

        Args:
        - images (torch.Tensor): The input images.

        Returns:
        - scores (torch.Tensor): The predicted scores.
        """
        assert hasattr(self, "learner"), "The model has not been adapted yet."

        # Prepare the feature.
        if self.feature_extractor_type == "clip":
            images = torch.stack([self.preprocess(image) for image in images])
            images = images.to(self.device)
            image_features = self.feature_extractor.encode_image(images).float()

        self.learner.eval()
        with torch.no_grad():
            scores = self.learner(image_features)
        return scores

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

