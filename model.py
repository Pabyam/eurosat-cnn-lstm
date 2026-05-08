import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, cnn_hidden=128, lstm_hidden=256):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, cnn_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_hidden),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.feature_spatial_size = 8 * 8
        self.cnn_hidden = cnn_hidden
        self.lstm = nn.LSTM(input_size=self.feature_spatial_size,
                            hidden_size=lstm_hidden,
                            batch_first=True,
                            bidirectional=False)
        self.classifier = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        batch_size, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, -1, C)
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = hn[-1]
        out = self.classifier(x)
        return out
