# Baselines

In this folder we include some simple baselines with runnable examples on
`bsuite`.

## Agents

Unless stated otherwise, all agents are implemented in
[TensorFlow 1.13](tensorflow.org). * `actor_critic`: A feed-forward
implementation of the advantage actor-critic (A2C) algorithm, with TD(lambda). *
`actor_critic_rnn`: A recurrent version of the above agent. * `dqn`: An
implementation of the deep Q-networks (DQN) algorithm. * `dqn_jax`: An example
implementation of the same algorithm in [JAX](github.com/google/jax). *
`dqn_tf2`: An example implementation of the same algorithm in
[TensorFlow 2](https://www.tensorflow.org/beta). * `boot_dqn`: An implementation
of the Bootstrapped DQN with randomized priors algorithm described in
[Osband et al. 2018.](https://arxiv.org/abs/1806.03335). * `random`: A
simple uniform random agent.

Additionally, we provide examples of running existing baselines from other
codebases:

*   `dopamine_dqn`: An implementation of DQN from
    [Dopamine](github.com/google/dopamine).
*   `openai_dqn`: An implementation of DQN from
    [OpenAI baselines](github.com/openai/baselines).
*   `openai_ppo`: An implementation of PPO from
    [OpenAI baselines](github.com/openai/baselines).

## Running the baselines

Inside each agent folder is a `run.py` file which will run the agent against one
`bsuite` environment. To run against the entire sweep, see
`scripts/run_sweep.py`.
