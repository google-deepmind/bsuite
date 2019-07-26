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
"""Runs a random agent on a bsuite experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.random import random

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite setting identifier')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes to run')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  env = bsuite.load_from_id(FLAGS.bsuite_id)
  agent = random.default_agent(obs_spec=env.observation_spec(),
                               action_spec=env.action_spec(),
                               seed=FLAGS.seed)

  num_episodes = getattr(env, 'bsuite_num_episodes', FLAGS.num_episodes)

  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)

if __name__ == '__main__':
  app.run(main)
