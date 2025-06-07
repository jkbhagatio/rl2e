"""Tests for `bandits.py`."""

from itertools import product

import numpy as np

import rl2e.ch2.bandits as bandits


def test_defaults():
    """Tests all module Bandit class methods and functions with default params."""
    b = bandits.Bandit()
    b.gen_reward_data()
    b.run()
    bandits.plot_reward_dists(b.reward_data)


def test_runs():
    """Tests Bandit.run() with various combinations of params."""
    # <s Set up matrix of params for each of 4 combinations of action policy (e-greedy,
    # gradient) and action values estimation method (sample-average, weighted-average)
    e_vals = (0.01, 0.1, 0.2)
    initial_action_values = (np.zeros(5), (np.ones(5) * 5))
    alphas = (0.1, 0.5)
    use_unbiased_stepsizes = (True, False)
    use_ucbs = (True, False)
    c_vals = (1, 2, 4)
    egreedy_sampleavg_params = list(product(e_vals, initial_action_values, use_ucbs, c_vals))
    egreedy_weightedavg_params = list(
        product(
            e_vals,
            initial_action_values,
            alphas,
            use_unbiased_stepsizes,
            use_ucbs,
            c_vals,
        )
    )
    # gradient_sampleavg_params = None  # no params to change for gradient, sample-avg
    gradient_weightedavg_params = list(product(alphas, use_unbiased_stepsizes))
    # /s>

    # <s Perform bandit runs.
    # Run egreedy_sampleavg_params
    b = bandits.Bandit()
    b.action_policy = "e-greedy"
    b.action_value_est = "sample-average"
    b.n_arms = 5
    b.n_steps = 1000
    b.gen_reward_data()
    for p in egreedy_sampleavg_params:
        (b.e_val, b.initial_action_value, b.use_ucb, b.c_val) = p[0], p[1], p[2], p[3]
        b.run()
    df = b.get_outcome()
    assert df.shape[0] == len(egreedy_sampleavg_params)
    # Run egreedy_weightedavg_params
    b = bandits.Bandit()
    b.action_policy = "e-greedy"
    b.action_value_est = "weighted-average"
    b.n_arms = 5
    b.n_steps = 1000
    b.gen_reward_data()
    for p in egreedy_weightedavg_params:
        (
            b.e_val,
            b.initial_action_value,
            b.alpha,
            b.use_unbiased_stepsize,
            b.use_ucb,
            b.c_val,
        ) = (p[0], p[1], p[2], p[3], p[4], p[5])
        b.run()
    df = b.get_outcome()
    assert df.shape[0] == len(egreedy_weightedavg_params)
    # Run gradient_sampleavg_params
    b = bandits.Bandit()
    b.action_policy = "gradient"
    b.action_value_est = "sample-average"
    b.n_arms = 5
    b.n_steps = 1000
    b.gen_reward_data()
    b.run()
    df = b.get_outcome()
    assert df.shape[0] == 1
    # Run gradient_weightedavg_params
    b = bandits.Bandit()
    b.action_policy = "gradient"
    b.action_value_est = "weighted-average"
    b.n_arms = 5
    b.n_steps = 1000
    b.gen_reward_data()
    for p in gradient_weightedavg_params:
        b.alpha, b.use_unbiased_stepsize = p[0], p[1]
        b.run()
    df = b.get_outcome()
    assert df.shape[0] == len(gradient_weightedavg_params)
    # /s>
