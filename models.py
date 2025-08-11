import torch
import torch.nn as nn
import math 

# Imitate https://keras.io/api/layers/regularization_layers/gaussian_noise/
class GaussianNoise(nn.Module):
  def __init__(self, std=0.05):
    super(GaussianNoise, self).__init__()
    self.std = std

  def forward(self, x):
    if self.training:  # Apply noise only during training
      noise = torch.randn_like(x) * self.std
      return x + noise
    return x

class RNACNN_GN(nn.Module):
  def __init__(self):
    super(RNACNN_GN, self).__init__()

    self.features = nn.Sequential(
      GaussianNoise(),
      # First conv block
      nn.Conv1d(4, 128, kernel_size=24),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Second conv block
      nn.Conv1d(128, 64, kernel_size=12),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Third conv block
      nn.Conv1d(64, 64, kernel_size=6),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Fourth conv block
      nn.Conv1d(64, 32, kernel_size=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.AdaptiveMaxPool1d(1)
    )

    self.shared_dropout = nn.Dropout(0.5)

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(1),
      nn.Sigmoid()
    )

    self.regression = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(1),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.shared_dropout(x)
    logits = self.classifier(x)
    rate = self.regression(x)
    return logits, rate

  def name(self):
    return 'RNACNN_GN'


class RNACNN(nn.Module):
  def __init__(self):
    super(RNACNN, self).__init__()

    self.features = nn.Sequential(
      # First conv block
      nn.Conv1d(in_channels=4, out_channels=128, kernel_size=24, stride=1),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Second conv block
      nn.Conv1d(in_channels=128, out_channels=64, kernel_size=12, stride=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Third conv block
      nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, stride=1),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Fourth conv block
      nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.AdaptiveMaxPool1d(1)
    )

    self.shared_dropout = nn.Dropout(0.5)

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(1),
      nn.Sigmoid()
    )

    self.regression = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.features(x)
    x = self.shared_dropout(x)
    logits = self.classifier(x)
    rate = self.regression(x)
    return logits, rate

  def name(self):
    return 'RNACNN'


class RNACNN_FC_GN(nn.Module):
  def __init__(self):
    super(RNACNN_FC_GN, self).__init__()

    self.features = nn.Sequential(
      # First conv block
      GaussianNoise(),
      nn.Conv1d(4, 128, kernel_size=24),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Second conv block
      nn.Conv1d(128, 64, kernel_size=12),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Third conv block
      nn.Conv1d(64, 64, kernel_size=6),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Fourth conv block
      nn.Conv1d(64, 32, kernel_size=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.AdaptiveMaxPool1d(1)
    )

    self.shared_dropout = nn.Dropout(0.5)

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(10),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.LazyLinear(1),
      nn.Sigmoid()
    )

    self.regression = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(10),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.LazyLinear(1),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.shared_dropout(x)
    logits = self.classifier(x)
    rate = self.regression(x)
    return logits, rate

  def name(self):
    return 'RNACNN_FC_GN'


class RNACNN_FC(nn.Module):
  def __init__(self):
    super(RNACNN_FC, self).__init__()

    self.features = nn.Sequential(
      # First conv block
      nn.Conv1d(4, 128, kernel_size=24),
      nn.BatchNorm1d(128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Second conv block
      nn.Conv1d(128, 64, kernel_size=12),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Third conv block
      nn.Conv1d(64, 64, kernel_size=6),
      nn.BatchNorm1d(64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.MaxPool1d(2),

      # Fourth conv block
      nn.Conv1d(64, 32, kernel_size=3),
      nn.BatchNorm1d(32),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.AdaptiveMaxPool1d(1)
    )

    self.shared_dropout = nn.Dropout(0.5)

    self.classifier = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(10),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.LazyLinear(1),
      nn.Sigmoid()
    )

    self.regression = nn.Sequential(
      nn.Flatten(),
      nn.LazyLinear(10),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.LazyLinear(1),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.shared_dropout(x)
    logits = self.classifier(x)
    rate = self.regression(x)
    return logits, rate

  def name(self):
    return 'RNACNN_FC'

import torch.nn as nn

class RNACNN_pete(nn.Module):
    def __init__(self):
        super(RNACNN_pete, self).__init__()
        kernel_size = 3  # Fixed small kernel for local features
        dilations = [1, 7, 24, 32]  # Increasing to grow receptive field: ~3, ~17, ~65, ~129
        channels = [4, 32, 64, 128, 256]  # Input channels + increasing out_channels

        self.features = nn.Sequential()
        for i in range(4):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            d = dilations[i]
            pad = d * (kernel_size - 1) // 2  # Padding for 'same' output size
            self.features.add_module(f'conv{i}', nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=1, dilation=d, padding=pad))
            self.features.add_module(f'bn{i}', nn.BatchNorm1d(out_ch))
            self.features.add_module(f'relu{i}', nn.ReLU())
            self.features.add_module(f'drop{i}', nn.Dropout(0.5))

        self.features.add_module('adaptive_pool', nn.AdaptiveMaxPool1d(1))

        self.shared_dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )
        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.shared_dropout(x)
        logits = self.classifier(x)
        rate = self.regression(x)
        return logits, rate

    def name(self):
        return 'RNACNN_pete'