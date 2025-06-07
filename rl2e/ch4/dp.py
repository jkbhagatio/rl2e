"""Sets up and runs dynamic programming agents and algos."""

from dataclasses import dataclass  # field
from typing import Callable, Union  # Literal, Mapping, Sequence, Annotated, List

import numpy as np
from numpy import typing as npt
from numpy.random import choice


@dataclass(slots=True)
class Dp:
    """Dynamic programming techniques to improve policies.

    Attributes:
        action_trans: Possible transitions from each state-action pair to its
            possible successor states. Each row represents a state, and each column
            represents a possible action for that state. Each row-column element
            itself can be an array, which would contain the possible multiple
            successor states accessible from that state-action pair.
        action_trans_p: Probabilities for the transitions in `action_trans`.
        action_rewards: Rewards corresponding to the transitions in
            `action_trans`.
        state_values: State values given the policy.
        action_values: State-action values given the policy.
        policy_probs: The probability of taking each possible action from a given
            state, according to the policy.
        policy_eval: Policy function for choosing an action from state-action values
            in `policy_evaluation()`.
        policy_improve: Policy function for choosing an action from state-action values
            in `policy_improvement()`.
        gamma: Reward discounting term.
    """

    action_trans: npt.NDArray[np.float_]
    action_trans_p: npt.NDArray[np.float_]
    action_rewards: npt.NDArray[np.float_]
    state_values: npt.NDArray[np.float_] = None
    action_values: npt.NDArray[np.float_] = None
    policy_probs: npt.NDArray[np.float_] = None
    # These policies will find the *action-value* of the policy, we then have to
    # reverse-index from this value to find the actual action.
    # By default, `policy_eval` will choose an action from a state weighted by the
    # relative action-values for all actions from that state (prob pick from softmax).
    policy_eval: Callable = lambda x: choice(x, p=((np.exp(x)) / np.sum(np.exp(x))))
    policy_improve: Callable = lambda x: np.max(x)
    gamma: float = 1

    def __post_init__(self):
        """Initializes some dependent attributes."""
        if self.state_values is None:
            self.state_values = np.zeros(self.action_trans.shape[0])
        if self.action_values is None:
            self.action_values = np.zeros_like(self.action_trans)
        if self.policy_probs is None:  # set to equiprobable by default
            self.policy_probs = np.ones_like(self.action_trans) / self.action_trans.shape[1]

    def policy_evaluation(
        self,
        term_thresh: float = 0.001,
        max_iter_ct: int = 100,
        do_value_iter: bool = False,
        use_log: bool = True,
        async_states: npt.NDArray[np.int_] | None = None,
    ) -> Union[None, tuple[int, float]]:
        """Evaluates a policy by updating state values via sweeps through state-space.

        Terminates after some max state-space iteration count, or when the maximum
        difference in state values between consecutive state-space sweeps is less
        than some value, `term_thresh`.

        Args:
            term_thresh: If state values between consecutive state-space sweeps are
                less than this value, the evaluation algorithm terminates.
            max_iter_ct: The maximum number of iterations to perform through the entire
                state-space before stopping the evaluation algorithm (if consecutive
                state-values do not converge earlier to some value less than
                `term_thresh`).
            do_value_iter: If True, perform value iteration instead of simple policy
                evaluation.
            use_log: If true, return the final `iter_ct` and `delta`.

        Returns:
            iter_ct: The number of iterations taken to converge to a stable policy.
            delta: The maximum difference in state values between consecutive sweeps.

        """
        for iter_ct in range(max_iter_ct):
            delta = 0
            states = async_states if async_states else range(len(self.state_values))
            for state in states:
                cur_val = self.state_values[state]
                # Mask all possible actions and successor states to those possible given
                # current state.
                s_s_set = self.action_trans[state]
                val_action_mask = ~np.isnan(s_s_set)
                s_s_set = s_s_set[val_action_mask].astype(int)
                val_actions = np.where(val_action_mask)[0]
                s_s_vals = np.zeros(s_s_set.shape[0])
                # Async, sequential state value updates.
                for i, (action, s_s) in enumerate(zip(val_actions, s_s_set)):
                    s_s_vals[i] = np.sum(
                        self.action_trans_p[state, action]
                        * (
                            self.action_rewards[state, action]
                            + self.gamma * self.state_values[s_s]
                        )
                    )
                # Update state and action values.
                if do_value_iter:
                    new_val = np.max(s_s_vals)
                else:
                    new_val = np.sum(self.policy_probs[state, val_actions] * s_s_vals)
                self.state_values[state] = new_val
                self.action_values[state, val_actions] = (
                    self.action_trans_p[state, val_actions]
                    * self.action_rewards[state, val_actions]
                    + self.gamma * self.state_values[s_s_set]
                )
                delta = np.max(np.array((delta, np.abs(cur_val - new_val))))
            if delta < term_thresh:
                break
        if use_log:
            return iter_ct, delta

    def policy_improvement(self, stable_thresh=0.1) -> bool:
        """Improves a policy via greedy action selection.

        Args:
            stable_thresh: The threshold for comparing policy probabilities between the
                current and new policies to determine if the current policy is stable.

        Returns:
            stable: If True, no more policy improvement can occur.
        """
        # Update `policy_probs` based on softmax of `action_values`.
        a_t, a_v = self.action_trans, self.action_values
        a_v -= np.max(a_v, axis=1, keepdims=True)  # subtract max for numerical stability.
        new_policy_probs = (
            np.exp(a_v) / np.tile(np.sum(np.exp(a_v), axis=1), (a_t.shape[1], 1)).transpose()
        )
        # Compare 'current' and 'new' policy probabilities to check for stability.
        policy_change = np.abs(new_policy_probs - self.policy_probs).max()
        self.policy_probs = new_policy_probs
        return policy_change < stable_thresh

    def policy_iteration(
        self,
        max_iter_ct: int = 100,
        eval_params: dict | None = None,
        improve_params: dict | None = None,
    ) -> int:
        """Runs policy iteration via chain of evaluation and improvement.

        Args:
            max_iter_ct: The maximum number of iterations of policy evaluation followed
                by policy improvement.
            eval_params: Parameters to pass to `policy_evaluation()`.

        Returns:
            iter_ct: The number of iterations taken to converge to a stable policy.
        """
        if eval_params is None:
            eval_params = {}
        if improve_params is None:
            improve_params = {}
        stable = False
        iter_ct = 0
        while (not stable) and (iter_ct < max_iter_ct):
            self.policy_evaluation(**eval_params)
            stable = self.policy_improvement(**improve_params)
            iter_ct += 1
        return iter_ct


class FunCallArray(np.ndarray):
    """Numpy array that can hold functions as elements that are called when indexed."""

    def __new__(cls, input_array):
        """Creates a view of the input array as an instance of this class."""
        obj = np.asarray(input_array).view(cls)
        return obj

    def __getitem__(self, index):
        """Returns the element at the given index, calling it if it's a function."""
        item = super().__getitem__(index)
        if callable(item):
            return item()
        return item
