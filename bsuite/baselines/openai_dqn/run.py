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
"""Trains an OpenAI Baselines DQN agent on bsuite.

Note that OpenAI Gym is not installed with bsuite by default.
"""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags

from baselines import deepq
from bsuite import bsuite
from bsuite.logging import terminal_logging
from bsuite.utils import gym_wrapper

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_string('results_dir', '/tmp/bsuite', 'directory for csv logs')
flags.DEFINE_boolean('overwrite', False, 'overwrite csv if found')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_integer('batch_size', 32, 'size of batches sampled from replay')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 128, 'min replay size before training.')
flags.DEFINE_integer('sgd_period', 1, 'steps between online net updates')
flags.DEFINE_integer('target_update_period', 4,
                     'steps between target net updates')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('epsilon', 0.05, 'fraction of exploratory random actions')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')

FLAGS = flags.FLAGS


def main(_):
  raw_env = bsuite.load_and_record_to_csv(
      bsuite_id=FLAGS.bsuite_id,
      results_dir=FLAGS.results_dir,
      overwrite=FLAGS.overwrite,
  )
  if FLAGS.verbose:
    raw_env = terminal_logging.wrap_environment(raw_env, log_every=True)
  env = gym_wrapper.GymWrapper(raw_env)
  num_episodes = getattr(env, 'bsuite_num_episodes', FLAGS.num_episodes)

  def callback(lcl, unused_glb):
    # Terminate after `num_episodes`.
    try:
      return lcl['num_episodes'] > num_episodes
    except KeyError:
      return False

  # Note: we should never run for this many steps as we end after `num_episodes`
  total_timesteps = int(1e9)

  deepq.learn(
      env=env,
      network='mlp',
      hiddens=[FLAGS.num_units] * FLAGS.num_hidden_layers,
      batch_size=FLAGS.batch_size,
      lr=FLAGS.learning_rate,
      total_timesteps=total_timesteps,
      buffer_size=FLAGS.replay_capacity,
      exploration_fraction=1./total_timesteps,  # i.e. immediately anneal.
      exploration_final_eps=FLAGS.epsilon,  # constant epsilon.
      print_freq=None,  # pylint: disable=wrong-arg-types
      learning_starts=FLAGS.min_replay_size,
      target_network_update_freq=FLAGS.target_update_period,
      callback=callback,  # pytype: disable=wrong-arg-types
      gamma=FLAGS.agent_discount,
  )


if __name__ == '__main__':
  app.run(main)
