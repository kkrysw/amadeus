import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, n_mels=229, n_class=88, cnn_out_channels=128, lstm_hidden_size=256, lstm_layers=2):
        super(CRNN, self).__init__()

        # CNN encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, cnn_out_channels, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((None, 1))  # [Batch, Channels, Time, 1]
        )

        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        # Output heads
        self.frame_head = nn.Linear(lstm_hidden_size * 2, n_class)
        self.onset_head = nn.Linear(lstm_hidden_size * 2, n_class)

    def forward(self, x):
        # x: [Batch, n_mels, Time]
        #x = x.unsqueeze(1)  # [Batch, 1, n_mels, Time]
        x = self.cnn(x)     # [Batch, Channels, Time, 1]
        x = x.squeeze(3)    # [Batch, Channels, Time]
        x = x.permute(0, 2, 1)  # [Batch, Time, Channels]

        x, _ = self.lstm(x)  # [Batch, Time, Hidden*2]

        frame_logits = self.frame_head(x)  # [Batch, Time, 88]
        onset_logits = self.onset_head(x)  # [Batch, Time, 88]

        return frame_logits, onset_logits
