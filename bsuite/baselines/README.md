# Baselines

In this folder we include some simple baselines with runnable examples on
`bsuite`.

To make the main `bsuite` library independent of machine learning frameworks, we
don't include all of the agent dependencies by default. See below for
installation instructions. Note that installation of these dependencies is
optional.

We recommend using the [`venv`](https://docs.python.org/3/library/venv.html)
virtual environment system.

## Agents and installation instructions.

The below agents require [TensorFlow 1.13](tensorflow.org),
[TRFL](github.com/deepmind/trfl), and [Sonnet](github.com/deepmind/sonnet).
These dependencies can be installed by `pip3 install bsuite[baselines].`

*   `actor_critic`: A feed-forward implementation of the advantage actor-critic
    (A2C) algorithm, with TD(lambda).
*   `actor_critic_rnn`: A recurrent version of the above agent.
*   `dqn`: An implementation of the deep Q-networks (DQN) algorithm.
*   `boot_dqn`: An implementation of the Bootstrapped DQN with randomized priors
    algorithm described in
    [Osband et al. 2018.](https://arxiv.org/abs/1806.03335).
*   `random`: A simple uniform random agent.

Additionally, we provide examples of running existing baselines from other
codebases, which introduce their own dependencies.

*   `dopamine_dqn`: An implementation of DQN from
    [Dopamine](github.com/google/dopamine).

    ```bash
    pip3 install dopamine-rl
    ```

*   `openai_dqn` and `openai_ppo`: Implementation of DQN and PPO from
    [OpenAI baselines](github.com/openai/baselines).

    ```bash
    pip3 install git+https://github.com/openai/baselines
    ```

We also provide examples of DQN written in other machine learning libraries:

*   `dqn_jax`: An example implementation of the same algorithm in
    [JAX](github.com/google/jax).

    ```bash
    pip3 install jax
    pip3 install jaxlib
    ```

*   `dqn_tf2`: An example implementation of the same algorithm using
    [TensorFlow 2](https://www.tensorflow.org/beta) and Sonnet 2.

    Note that installing TensorFlow 2 will likely break many of the TensorFlow
    1.X-based agents that haven't yet switched to using `tf.compat.v1`. To
    revert to e.g. TF 1.13, use `pip3 install tensorflow=1.13`.

    ```bash
    pip3 install tensorflow==2.0.0-beta1
    pip3 install --upgrade git+https://github.com/deepmind/sonnet@v2
    ```

## Running the baselines

Inside each agent folder is a `run.py` file which will run the agent against a
single `bsuite` environment, or the entire behavior suite by passing
`--bsuite_id=SWEEP`. For example. from the `baselines/dqn` folder, you could
run:

```bash
python3 run.py --bsuite_id=SWEEP
```
