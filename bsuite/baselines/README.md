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

The below agents require [TensorFlow 2.X](https://www.tensorflow.org),
[TRFL](https://github.com/deepmind/trfl), and [Sonnet 2](https://github.com/deepmind/sonnet).
These dependencies can be installed via `pip install bsuite[baselines].`

*   `actor_critic`: A feed-forward implementation of the advantage actor-critic
    (A2C) algorithm, with TD(lambda).
*   `actor_critic_rnn`: A recurrent version of the above agent.
*   `dqn`: An implementation of the deep Q-networks (DQN) algorithm.
*   `boot_dqn`: An implementation of the Bootstrapped DQN with randomized priors
    algorithm described in [Osband et al. 2018].
*   `random`: A simple uniform random agent.

Additionally, we provide examples of running existing baselines from other
codebases, which introduce their own dependencies.

*   `dopamine_dqn`: An implementation of DQN from
    [Dopamine](https://github.com/google/dopamine).

    ```bash
    pip install dopamine-rl
    ```

*   `openai_dqn` and `openai_ppo`: Implementation of DQN and PPO from
    [OpenAI baselines](https://github.com/openai/baselines).

    ```bash
    pip install git+https://github.com/openai/baselines
    ```

*   `dqn_jax`: An example implementation of the same algorithm in
    [JAX](https://github.com/google/jax).

    ```bash
    pip install jax
    pip install jaxlib
    ```

## Running the baselines

Inside each agent folder is a `run.py` file which will run the agent against a
single `bsuite` environment, or the entire behavior suite by passing
`--bsuite_id=SWEEP`. For example. from the `baselines/dqn` folder, you could
run:

```bash
python3 run.py --bsuite_id=SWEEP
```

[Osband et al. 2018]: https://arxiv.org/abs/1806.03335
