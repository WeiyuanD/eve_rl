from math import inf
import random
from typing import List
import torch
from torch.nn.utils.rnn import pad_sequence
from .replaybuffer import ReplayBuffer, Episode, Batch
import numpy as np


class VanillaEpisode(ReplayBuffer):
    def __init__(self, capacity: int, batch_size: int, image_size: int = 0):
        # Image observations have a flattened uint8 image prefix followed by
        # floating-point features (e.g. previous actions). Zero preserves state replay.
        if type(image_size) is not int or image_size < 0:
            raise ValueError("image_size must be a nonnegative integer")
        self.capacity = capacity
        self.image_size = image_size
        self._batch_size = batch_size
        self.buffer: List[Episode] = []
        self.position = 0

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def push(self, episode: Episode):
        if len(episode) < 1:
            return
        observations = np.array(episode.flat_obs)
        if self.image_size:
            if observations.ndim != 2 or observations.shape[1] < self.image_size:
                raise ValueError("Observation does not contain the configured image prefix")
            pixels = observations[:, :self.image_size]
            if (not np.isfinite(pixels).all() or np.any(pixels < 0)
                    or np.any(pixels > 255) or np.any(pixels != np.floor(pixels))):
                raise ValueError("Uint8 replay requires integer image values in [0, 255]")
            observations = (
                pixels.astype(np.uint8),
                # Copy: a view would retain the full float32 image allocation.
                observations[:, self.image_size:].astype(np.float32, copy=True),
            )
        episode_np = (
            observations,
            np.array(episode.actions),
            np.array(episode.rewards),
            np.array(episode.terminals),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = episode_np
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self) -> Batch:
        episodes = random.sample(self.buffer, self.batch_size)

        if self.image_size:
            # Restore the original flattened ordering before padding. Includes
            # reset observation and final next observation (T+1), as before.
            state_batch = [
                torch.cat((torch.from_numpy(episode[0][0]).float(),
                           torch.from_numpy(episode[0][1])), dim=-1)
                for episode in episodes
            ]
        else:
            state_batch = [torch.from_numpy(episode[0]) for episode in episodes]
        action_batch = [torch.from_numpy(episode[1]) for episode in episodes]
        reward_batch = [
            torch.from_numpy(episode[2]).unsqueeze(1) for episode in episodes
        ]
        done_batch = [torch.from_numpy(episode[3]).unsqueeze(1) for episode in episodes]

        state_batch = pad_sequence(state_batch, batch_first=True)
        action_batch = pad_sequence(action_batch, batch_first=True)
        reward_batch = pad_sequence(reward_batch, batch_first=True, padding_value=inf)
        done_batch = pad_sequence(done_batch, batch_first=True)

        padding_mask = torch.ones_like(reward_batch)
        padding_mask[reward_batch == inf] = 0
        reward_batch[reward_batch == inf] = 0
        return Batch(state_batch, action_batch, reward_batch, done_batch, padding_mask)

    def __len__(self):
        return len(self.buffer)

    def copy(self):
        copy = self.__class__(self.capacity, self.batch_size, self.image_size)
        for i in range(len(self.buffer)):
            copy.buffer.append(self.buffer[i])
        copy.position = self.position
        return copy

    def close(self):
        del self.buffer
