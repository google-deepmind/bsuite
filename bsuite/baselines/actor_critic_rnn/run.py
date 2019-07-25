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
"""Run an actor-critic agent instance on a bsuite experiment."""

from __future__ import absolute_import
from __future__ import division
# Standard __future__ imports.
from __future__ import print_function

from absl import app
from absl import flags
from bsuite import bsuite
from bsuite.baselines import experiment
from bsuite.baselines.actor_critic_rnn import actor_critic_rnn
import tensorflow as tf

flags.DEFINE_string('bsuite_id', 'catch/0', 'bsuite identifier')
flags.DEFINE_integer('seed', 42, 'seed for random number generation')
flags.DEFINE_integer('num_episodes', 10000, 'number of episodes to run')
flags.DEFINE_integer('num_hidden_layers', 2, 'number of hidden layers')
flags.DEFINE_integer('num_units', 64, 'number of units per hidden layer')
flags.DEFINE_float('learning_rate', 3e-3, 'the learning rate')
flags.DEFINE_integer('sequence_length', 32, 'mumber of transitions to batch')
flags.DEFINE_float('td_lambda', 0.9, 'mixing parameter for boostrapping')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')

FLAGS = flags.FLAGS
FLAGS.alsologtostderr = True


def main(argv):
  del argv  # Unused.
  env = bsuite.load_from_id(FLAGS.bsuite_id)

  num_actions = env.action_spec().num_values
  hidden_sizes = [FLAGS.num_units] * FLAGS.num_hidden_layers
  network = actor_critic_rnn.PolicyValueRNN(hidden_sizes, num_actions)

  agent = actor_critic_rnn.ActorCriticRNN(
      obs_spec=env.observation_spec(),
      action_spec=env.action_spec(),
      network=network,
      optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate),
      sequence_length=FLAGS.sequence_length,
      td_lambda=FLAGS.td_lambda,
      agent_discount=FLAGS.agent_discount,
      seed=FLAGS.seed,
  )

  experiment.run(
      agent, env, num_episodes=FLAGS.num_episodes, verbose=FLAGS.verbose)


if __name__ == '__main__':
  app.run(main)
