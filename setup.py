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
"""Install script for setuptools."""

from setuptools import find_packages
from setuptools import setup

# Additional requirements for TensorFlow baselines, excluding OpenAI & Dopamine.
# See baselines/README.md for more information.
baselines_require = [
    'dm-sonnet',
    'dm-tree',
    'tensorflow',
    'tensorflow_probability',
    'trfl',
    'tqdm',
]

# Additional requirements for JAX baselines.
# See baselines/README.md for more information.
baselines_jax_require = [
    'dm-haiku',
    'dm-tree',
    'jax',
    'jaxlib',
    'git+git://github.com/deepmind/rlax.git'
    'tqdm',
]

dependency_links = [
    'http://github.com/google/dopamine/tarball/master#egg=dopamine_rl',
]


setup(
    name='bsuite',
    description=('Core RL Behaviour Suite. '
                 'A collection of reinforcement learning experiments.'),
    author='DeepMind',
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=[
        'absl-py',
        'dm_env',
        'matplotlib',
        'numpy',
        'pandas',
        'plotnine',
        'scipy',
        'scikit-image',
        'six',
        'termcolor',
    ],
    tests_require=[
        'absl-py',
        'nose',
    ],
    extras_require={
        'baselines': baselines_require,
        'baselines_jax': baselines_jax_require,
    },
    test_suite='nose.collector',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
