"""Sets up and runs linear programming algos."""

from dataclasses import dataclass  # field

import numpy as np
from numpy import typing as npt
from tabulate import tabulate


@dataclass(slots=True)
class SimplexSolver:
    """A class to solve linear programming (LP) problems using the Simplex method.

    Solves maximizing objective functions - LP problems of the form:
    c.Tx -> max
    Ax <= b
    x >= 0
    where:
    'x': decision variables vector
    'c': vector containing the coefficients of the objective function: tableau[0, :-1]
    'c.Tx' : objective function value: tableau[0, -1]
    'A': constraints matrix coefficients: tableau[1:, :-1] (decision & slack var values)
    'b': RHS constraints vector: tableau[1:, -1]

    Attributes:
        tableau: The full simplex tableau: includes RHS col, but not 'Basis' col.
        basic_vars: Indices of basic variables.
        nonbasic_vars: Indices of non-basic variables.
        n_pivots: Number of pivot operations performed.
        pprint: Whether to pretty print the tableau after each pivot.
    """

    tableau: npt.NDArray[np.float_]
    basic_vars: npt.NDArray[np.int_] = None
    nonbasic_vars: npt.NDArray[np.int_] = None
    n_pivots: int = 0
    pprint: bool = False

    def __post_init__(self):
        """Initializes basic and nonbasic vars from the tableau."""
        n_vars = self.tableau.shape[1] - 1  # exclude RHS
        n_slack_vars = self.tableau.shape[0] - 1  # exclude objective row
        n_decision_vars = n_vars - n_slack_vars
        self.basic_vars = np.arange(n_decision_vars, n_vars, dtype=int)
        self.nonbasic_vars = np.arange(0, n_decision_vars, dtype=int)

    def solve(self):
        """
        Solves the maximizing objective LP problem using the Simplex method.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The optimal solution vector.
                - float: The optimal objective value.
        """
        # Solved if/when all nonbasic vars have nonpositive coefficients in obj fun
        while np.any(self.tableau[0, 0 : len(self.nonbasic_vars)] > 0):
            # Find entering var (representing steepest ascent direction)
            enter_col = np.argmax(self.tableau[0, 0 : len(self.nonbasic_vars)])
            # Find leaving variable (minimum ratio test: largest nonpos ratio):
            # find rows with valid entering var
            positive_coeffs = self.tableau[1:, enter_col] > 0
            if not np.any(positive_coeffs):
                raise ValueError("Problem is unbounded")
            ratios = np.full(len(positive_coeffs), -np.inf)
            ratios[positive_coeffs] = (
                self.tableau[1:, -1][positive_coeffs]
                / self.tableau[1:, enter_col][positive_coeffs]
            )
            leave_row = np.argmax(ratios) + 1
            # Perform pivot
            self.pivot(enter_col, leave_row)

        # Extract solution from tableau: RHS of obj fun and decision var rows
        solution = np.zeros(len(self.nonbasic_vars))
        decision_vars = self.basic_vars < len(self.nonbasic_vars)
        solution[self.basic_vars[decision_vars]] = self.tableau[1:, -1][decision_vars]

        return solution, -self.tableau[0, -1]

    def pivot(self, enter_col, leave_row):
        """
        Performs a pivot operation on the simplex tableau using Dantzig's rule.

        Args:
            enter_col (int): The index of the entering variable.
            leave_row (int): The index of the leaving variable.
        """
        # Perform substitution on the pivot row.
        self.tableau[leave_row] /= self.tableau[leave_row, enter_col]  # norm pivot row
        for i in range(self.tableau.shape[0]):
            if i != leave_row:  # skip pivot row, update all others
                self.tableau[i] -= self.tableau[i, enter_col] * self.tableau[leave_row]

        # Simultaneous swap of basic and nonbasic variables.
        self.basic_vars[leave_row - 1], self.nonbasic_vars[enter_col] = (
            self.nonbasic_vars[enter_col],
            self.basic_vars[leave_row - 1],
        )
        self.n_pivots += 1

        # import ipdb; ipdb.set_trace()

        # Pretty print tableau.
        if self.pprint:
            self.pprint_tableau()

    def pprint_tableau(self):
        """Pretty prints the current tableau."""
        headers = (
            ["Basic"]
            + [f"x{i+1}" for i in range(len(self.nonbasic_vars))]
            + [f"s{i+1}" for i in range(len(self.basic_vars))]
            + ["RHS"]
        )
        basic_col = ["Z"] + [headers[i + 1] for i in self.basic_vars]
        table_data = np.hstack((np.array(basic_col).reshape(-1, 1), self.tableau))
        print(f"Pivot number {self.n_pivots}: \n")
        print(tabulate(table_data, headers=headers, floatfmt=".3f"))
        print("\n")
