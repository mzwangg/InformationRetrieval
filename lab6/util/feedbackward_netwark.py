import torch.nn as nn

class feedbackward_netwark(nn.Module):

    def __init__(self, input_szie, output_size):
        super(feedbackward_netwark, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_szie, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, output_size))

    def forward(self, x):
        out = self.model(x)
        return out