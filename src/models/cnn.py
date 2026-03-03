import torch.nn as nn

# CNN 
class CNN(nn.Module):
    def __init__(self, in_channels: int, in_features: int, out_features: int):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_features = out_features
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (in_features // 16), 256), 
            nn.ReLU(),
            nn.Linear(256, out_features),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

    def get_config(self):
        return {
            'scale': 16,
            'width': 256,
        }
