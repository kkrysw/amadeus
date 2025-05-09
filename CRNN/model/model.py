import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, n_mels=229, n_class=88, cnn_out_channels=128, lstm_hidden_size=256, lstm_layers=2):
        super(CRNN, self).__init__()

        # CNN encoder (preserves time dimension)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # downsample freq only

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(64, cnn_out_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Output: [B, C, 1, T]
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.frame_head = nn.Linear(lstm_hidden_size * 2, n_class)
        self.onset_head = nn.Linear(lstm_hidden_size * 2, n_class)

    def forward(self, x):  # x: [B, 1, n_mels, time]
        x = self.cnn(x)     # → [B, C, 1, T]
        x = x.squeeze(2)    # → [B, C, T]
        x = x.permute(0, 2, 1)  # → [B, T, C]

        x, _ = self.lstm(x)     # → [B, T, 2*H]
        frame_logits = self.frame_head(x)
        onset_logits = self.onset_head(x)
        return frame_logits, onset_logits
