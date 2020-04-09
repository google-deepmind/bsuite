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
"""Basic test coverage for agent training."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from bsuite import sweep
from bsuite.baselines.third_party.openai_dqn import run

import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS


class RunTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    FLAGS.num_episodes = 5
    FLAGS.logging_mode = 'terminal'

  @parameterized.parameters(*sweep.TESTING)
  def test_run(self, bsuite_id: str):
    with tf.variable_scope(bsuite_id):
      run.run(bsuite_id)


if __name__ == '__main__':
  absltest.main()
