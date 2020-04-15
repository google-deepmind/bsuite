# Baselines

In this folder we include some simple baselines with runnable examples on
`bsuite`.

## Agents and installation instructions

To make the main `bsuite` library independent of machine learning frameworks, we
don't include all of the agent dependencies by default. See below for
installation instructions. Note that installation of these dependencies is
optional.

We recommend using Python's
[`venv`](https://docs.python.org/3/library/venv.html) virtual environment
system to manage your dependencies and avoid version conflicts.

### TensorFlow agents

The below agents are built using [TensorFlow 2], [trfl], and [Sonnet 2].

To install these dependencies, run:

```bash
pip install bsuite[baselines]
```

*   `actor_critic`: A feed-forward implementation of the advantage actor-critic
    (A2C) algorithm, with TD(lambda).
*   `actor_critic_rnn`: A recurrent version of the above agent.
*   `dqn`: An implementation of the deep Q-networks (DQN) algorithm.
*   `boot_dqn`: An implementation of the Bootstrapped DQN with randomized priors
    algorithm described in [Osband et al. 2018].

### JAX agents

The below agents are built using [JAX], [rlax], and [Haiku].

To install these dependencies, run:

```bash
pip install bsuite[baselines_jax]
```

*   `actor_critic`: A feed-forward implementation of the advantage actor-critic
    (A2C) algorithm, with TD(lambda).
*   `actor_critic_rnn`: A recurrent version of the above agent.
*   `dqn`: An implementation of the deep Q-networks (DQN) algorithm.

### Third-party agents

Additionally, we provide examples of running existing external baselines from
other codebases, which introduce their own dependencies.

*   `dopamine_dqn`: An implementation of DQN from [Dopamine].

    ```bash
    pip install dopamine-rl
    ```

*   `openai_dqn` and `openai_ppo`: Implementation of DQN and PPO from
    [OpenAI baselines].

    ```bash
    pip install git+https://github.com/openai/baselines
    ```

## Running the baselines

Inside each agent folder is a `run.py` file which will run the agent against a
single `bsuite` environment, or the entire behavior suite by passing
`--bsuite_id=SWEEP`. For example. from the `baselines/dqn` folder, you could
run:

```bash
python3 run.py --bsuite_id=SWEEP
```

[Dopamine]: https://github.com/google/dopamine
[Haiku]: https://github.com/deepmind/haiku
[JAX]: https://github.com/google/jax
[OpenAI baselines]: https://github.com/openai/baselines
[Osband et al. 2018]: https://arxiv.org/abs/1806.03335
[rlax]: https://github.com/deepmind/rlax
[Sonnet 2]: https://github.com/deepmind/sonnet
[TensorFlow 2]: https://tensorflow.org
[trfl]: https://github.com/deepmind/trfl
