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
"""Trains an OpenAI Baselines PPO agent on bsuite.

Note that OpenAI Gym is not installed with bsuite by default.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags

from baselines.common.vec_env import dummy_vec_env
from baselines.ppo2 import ppo2
from bsuite import bsuite
from bsuite.logging import terminal_logging
from bsuite.utils import gym_wrapper

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'directory for csv logs')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv if found')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

flags.DEFINE_string('network', 'mlp', 'name of network architecture')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('nsteps', 100, 'number of steps per ppo rollout')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')

FLAGS = flags.FLAGS


def main(_):
  def _load_env():
    raw_env = bsuite.load_and_record_to_csv(
        bsuite_id=FLAGS.bsuite_id,
        results_dir=FLAGS.results_dir,
        overwrite=FLAGS.overwrite,
    )
    if FLAGS.verbose:
      raw_env = terminal_logging.wrap_environment(raw_env, log_every=True)
    return gym_wrapper.GymWrapper(raw_env)
  env = dummy_vec_env.DummyVecEnv([_load_env])

  # Note: bsuite experiments do not actually need to run this many steps
  total_timesteps = int(1e6)

  ppo2.learn(
      env=env,
      network=FLAGS.network,
      lr=FLAGS.learning_rate,
      total_timesteps=total_timesteps,
      nsteps=FLAGS.nsteps,
      gamma=FLAGS.agent_discount,
  )


if __name__ == '__main__':
  app.run(main)
