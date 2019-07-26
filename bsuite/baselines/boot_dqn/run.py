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
"""Run agent on a bsuite experiment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags

from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.boot_dqn import boot_dqn

import tensorflow as tf

# Network options
flags.DEFINE_integer('num_ensemble', 20, 'number of ensemble networks')
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 50, 'number of units per hidden layer')
flags.DEFINE_float('prior_scale', 3., 'scale for additive prior network')

# Core DQN options
flags.DEFINE_integer('batch_size', 128, 'size of batches sampled from replay')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('replay_capacity', 100000, 'size of the replay buffer')
flags.DEFINE_integer('min_replay_size', 128, 'min transitions for sampling')
flags.DEFINE_integer('sgd_period', 1, 'steps between online net updates')
flags.DEFINE_integer('target_update_period', 4,
                     'steps between target net updates')
flags.DEFINE_float('mask_prob', 0.5, 'probability for bootstrap mask')
flags.DEFINE_float('noise_scale', 0.0, 'std of additive target noise')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')
flags.DEFINE_float('epsilon', 0.0, 'fraction of exploratory random actions')

# Experiment options
flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes to run')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.
  env = bsuite.load_from_id(FLAGS.bsuite_id)
  ensemble = boot_dqn.make_ensemble(
      num_actions=env.action_spec().num_values,
      num_ensemble=FLAGS.num_ensemble,
      num_hidden_layers=FLAGS.num_hidden_layers,
      num_units=FLAGS.num_units,
      prior_scale=FLAGS.prior_scale)
  target_ensemble = boot_dqn.make_ensemble(
      num_actions=env.action_spec().num_values,
      num_ensemble=FLAGS.num_ensemble,
      num_hidden_layers=FLAGS.num_hidden_layers,
      num_units=FLAGS.num_units,
      prior_scale=FLAGS.prior_scale)

  agent = boot_dqn.BootstrappedDqn(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      ensemble=ensemble,
      target_ensemble=target_ensemble,
      batch_size=FLAGS.batch_size,
      agent_discount=FLAGS.agent_discount,
      replay_capacity=FLAGS.replay_capacity,
      min_replay_size=FLAGS.min_replay_size,
      sgd_period=FLAGS.sgd_period,
      target_update_period=FLAGS.target_update_period,
      optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
      mask_prob=FLAGS.mask_prob,
      noise_scale=FLAGS.noise_scale,
      epsilon_fn=lambda x: FLAGS.epsilon,
      seed=FLAGS.seed)

  num_episodes = getattr(env, 'bsuite_num_episodes', FLAGS.num_episodes)

  experiment.run(
      agent=agent,
      environment=env,
      num_episodes=num_episodes,
      verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
