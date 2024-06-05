from torch import nn

class ScorePredictor(nn.Module):

    def __init__(self, width=1024, depth=2, dropout_rate=0.5):
        super(ScorePredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, width),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            *[nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout_rate)) for _ in range(depth)],
            nn.Linear(width, 1),
        )

    def forward(self, x):
        return self.net(x)