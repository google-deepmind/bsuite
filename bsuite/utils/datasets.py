# python3
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
"""Downloads and loads the MNIST dataset.

Adapted from https://github.com/google/jax/blob/master/examples/datasets.py
"""

import array
import gzip
import os
from os import path
import struct

from absl import logging
import numpy as np
from six.moves.urllib.request import urlretrieve


def _download(url, filename, directory="/tmp/mnist"):
  """Download a url to a file in the given directory."""
  if not path.exists(directory):
    os.makedirs(directory)
  out_file = path.join(directory, filename)
  if not path.isfile(out_file):
    urlretrieve(url, out_file)
    logging.info("Downloaded %s to %s", url, directory)


def load_mnist(directory="/tmp/mnist"):
  """Download and parse the raw MNIST dataset."""
  # CVDF mirror of http://yann.lecun.com/exdb/mnist/
  base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

  def parse_labels(filename):
    with gzip.open(filename, "rb") as fh:
      _ = struct.unpack(">II", fh.read(8))
      return np.array(array.array("B", fh.read()), dtype=np.uint8)

  def parse_images(filename):
    with gzip.open(filename, "rb") as fh:
      _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
      return np.array(array.array("B", fh.read()),
                      dtype=np.int8).reshape((num_data, rows, cols))

  for filename in ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
                   "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]:
    _download(base_url + filename, filename, directory)

  train_images = parse_images(
      path.join(directory, "train-images-idx3-ubyte.gz"))
  train_labels = parse_labels(
      path.join(directory, "train-labels-idx1-ubyte.gz"))
  test_images = parse_images(path.join(directory, "t10k-images-idx3-ubyte.gz"))
  test_labels = parse_labels(path.join(directory, "t10k-labels-idx1-ubyte.gz"))

  return (train_images, train_labels), (test_images, test_labels)
