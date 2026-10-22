"""CNN and recurrent observation head for flattened image/action sequences."""

from math import prod

import torch
from torch import nn

from .component import Component


class ChannelRMSNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(1, keepdim=True) + 1e-6) * self.scale


class ImageLSTM(Component):
    """Accept [B,T,pixels+actions], with CHW pixels first and actions last.

    Training forwards start from zero recurrent state. forward_play retains state
    until reset is called. This is an observation head, not an output body.
    Constructor attributes are retained for eve_rl checkpoint reconstruction.
    """

    def __init__(self, image_shape, action_dim, channels, feature_dim,
                 hidden_size, num_layers, action_fusion="concat"):
        super().__init__()
        self.image_shape = list(image_shape)
        self.action_dim = action_dim
        self.channels = list(channels)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_fusion = action_fusion
        if len(image_shape) != 3 or any(size < 1 for size in image_shape):
            raise ValueError("image_shape must contain positive CHW dimensions")
        if not channels or any(size < 1 for size in channels):
            raise ValueError("channels must contain positive CNN widths")
        if action_dim < 0 or min(feature_dim, hidden_size, num_layers) < 1:
            raise ValueError("Invalid action, feature or recurrent dimensions")
        if action_fusion not in ("concat", "sum"):
            raise ValueError("action_fusion must be concat or sum")

        self.pixel_count = prod(image_shape)
        in_channels, height, width = image_shape
        blocks = []
        for out_channels in channels:
            if min(height, width) < 2:
                raise ValueError("Too many pooling blocks for image resolution")
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, 5, padding=2),
                nn.MaxPool2d(2), ChannelRMSNorm(out_channels), nn.SiLU(),
            ])
            in_channels, height, width = out_channels, height // 2, width // 2
        blocks.extend([nn.Flatten(), nn.Linear(in_channels * height * width, feature_dim)])
        self.encoder = nn.Sequential(*blocks)
        self.action_encoder = (
            nn.Sequential(nn.Linear(action_dim, feature_dim), nn.SiLU(),
                          nn.Linear(feature_dim, feature_dim))
            if action_dim and action_fusion == "sum" else None
        )
        lstm_inputs = feature_dim + (action_dim if action_fusion == "concat" else 0)
        self.lstm = nn.LSTM(lstm_inputs, hidden_size, num_layers, batch_first=True)
        self._hidden_state = None

    @property
    def n_inputs(self):
        return self.pixel_count + self.action_dim

    @n_inputs.setter
    def n_inputs(self, value):
        if value != self.n_inputs:
            raise ValueError(f"Expected {self.n_inputs} observation values, got {value}")

    @property
    def n_outputs(self):
        return self.hidden_size

    @property
    def device(self):
        return next(self.parameters()).device

    def _features(self, observations):
        if observations.ndim != 3 or observations.shape[-1] != self.n_inputs:
            raise ValueError("Expected [batch, time, flattened_observation]")
        batch, time, _ = observations.shape
        images = observations[..., :self.pixel_count].reshape(-1, *self.image_shape)
        features = self.encoder(images.float() / 255.0 - 0.5).reshape(batch, time, -1)
        if self.action_dim:
            actions = observations[..., self.pixel_count:].to(features.dtype)
            if self.action_fusion == "sum":
                features = features + self.action_encoder(actions)
            else:
                features = torch.cat((features, actions), dim=-1)
        return features

    def forward(self, obs_batch):
        output, _ = self.lstm(self._features(obs_batch))
        return output

    def forward_play(self, obs_batch):
        with torch.no_grad():
            output, self._hidden_state = self.lstm(
                self._features(obs_batch), self._hidden_state)
        return output

    def reset(self):
        self._hidden_state = None
