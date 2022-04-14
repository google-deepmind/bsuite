# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple, uniformly sampled replay buffer."""

from typing import Any, Optional, Sequence

import numpy as np


class Replay:
  """Uniform replay buffer. Allocates all required memory at initialization."""

  _data: Optional[Sequence[np.ndarray]]
  _capacity: int
  _num_added: int

  def __init__(self, capacity: int):
    """Initializes a new `Replay`.

    Args:
      capacity: The maximum number of items allowed in the replay. Adding
        items to a replay that is at maximum capacity will overwrite the oldest
        items.
    """
    self._data = None
    self._capacity = capacity
    self._num_added = 0

  def add(self, items: Sequence[Any]):
    """Adds a single sequence of items to the replay.

    Args:
      items: Sequence of items to add. Does not handle batched or nested items.
    """
    if self._data is None:
      self._preallocate(items)

    for slot, item in zip(self._data, items):
      slot[self._num_added % self._capacity] = item

    self._num_added += 1

  def sample(self, size: int) -> Sequence[np.ndarray]:
    """Returns a transposed/stacked minibatch. Each array has shape [B, ...]."""
    indices = np.random.randint(self.size, size=size)
    return [slot[indices] for slot in self._data]

  def reset(self,):
    """Resets the replay."""
    self._data = None

  @property
  def size(self) -> int:
    return min(self._capacity, self._num_added)

  @property
  def fraction_filled(self) -> float:
    return self.size / self._capacity

  def _preallocate(self, items: Sequence[Any]):
    """Assume flat structure of items."""
    as_array = []
    for item in items:
      if item is None:
        raise ValueError('Cannot store `None` objects in replay.')
      as_array.append(np.asarray(item))

    self._data = [np.zeros(dtype=x.dtype, shape=(self._capacity,) + x.shape)
                  for x in as_array]

  def __repr__(self):
    return 'Replay: size={}, capacity={}, num_added={}'.format(
        self.size, self._capacity, self._num_added)
