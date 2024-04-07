from __future__ import annotations
from contextlib import suppress
from abc import ABC, abstractmethod

import numpy as np

with suppress(ImportError):
    import cocoex  # type: ignore
import ioh  # type: ignore
from cma import CMAEvolutionStrategy, BoundPenalty  # type: ignore
from modcma import AskTellCMAES  # type: ignore


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def get(*, dim: int, fun: int, inst: int) -> Problem: ...

    @abstractmethod
    def __call__(self, x) -> float: ...

    @abstractmethod
    def __repr__(self) -> str: ...

    @property
    @abstractmethod
    def function_id(self) -> int: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    # TODO: change bounds to namedtuple?
    @property
    @abstractmethod
    def bounds_lower(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def bounds_upper(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def final_target_hit(self) -> bool:
        """https://github.com/numbbo/coco/blob/master/code-experiments/src/coco_problem.c#L443-L444"""

    @property
    @abstractmethod
    def evaluations(self) -> int: ...

    @property
    @abstractmethod
    def current_best_y(self) -> float: ...

    def is_end(self, budget: int) -> bool:
        return self.final_target_hit or self.evaluations >= budget

    def is_outside(self, point, *, tol=0) -> bool:
        """Returns `True` if the point is more than `tol` outside of the bounds."""
        return (
            (point < self.bounds_lower - tol) | (point > self.bounds_upper + tol)
        ).any()


class ProblemCocoex(Problem):
    @staticmethod
    def get(*, dim, fun, inst) -> ProblemCocoex:
        """>>> ProblemCocoex.get(dim=2, fun=1, inst=1)"""
        return ProblemCocoex(
            next(
                iter(
                    cocoex.Suite(  # pyright: ignore [reportPossiblyUnboundVariable]
                        "bbob",
                        "",
                        " ".join(
                            [
                                f"dimensions: {dim}",
                                f"function_indices: {fun}",
                                f"instance_indices: {inst}",
                            ]
                        ),
                    )
                )
            )
        )

    def __init__(self, problem):
        if not isinstance(problem, cocoex.interface.Problem):  # pyright: ignore [reportPossiblyUnboundVariable, reportAttributeAccessIssue]
            raise ValueError(f"Wrong problem type: {type(problem)}")
        self._problem = problem

    def __call__(self, x) -> float:
        return self._problem(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._problem)})"

    @property
    def function_id(self) -> int:
        return self._problem.id_function

    @property
    def dimension(self) -> int:
        return self._problem.dimension

    @property
    def bounds_lower(self) -> np.ndarray:
        return self._problem.lower_bounds

    @property
    def bounds_upper(self) -> np.ndarray:
        return self._problem.upper_bounds

    @property
    def final_target_hit(self) -> bool:
        return self._problem.final_target_hit

    @property
    def evaluations(self) -> int:
        return self._problem.evaluations

    @property
    def current_best_y(self) -> float:
        return self._problem.best_observed_fvalue1


class ProblemIoh(Problem):
    @staticmethod
    def get(*, dim, fun, inst) -> ProblemIoh:
        """>>> ProblemIoh.get(dim=2, fun=1, inst=1)"""
        return ProblemIoh(
            ioh.get_problem(
                fun,
                instance=inst,
                dimension=dim,
                problem_class=ioh.ProblemClass.BBOB,  # pyright: ignore [reportCallIssue]
            )
        )

    def __init__(self, problem: ioh.iohcpp.problem.BBOB):
        if not isinstance(problem, ioh.iohcpp.problem.BBOB):
            raise ValueError(f"Wrong problem type: {type(problem)}")
        self._problem = problem

    def __call__(self, x) -> float:
        return self._problem(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._problem)})"

    @property
    def function_id(self) -> int:
        return self._problem.meta_data.problem_id

    @property
    def dimension(self) -> int:
        return self._problem.meta_data.n_variables

    @property
    def bounds_lower(self) -> np.ndarray:
        return self._problem.bounds.lb

    @property
    def bounds_upper(self) -> np.ndarray:
        return self._problem.bounds.ub

    @property
    def final_target_hit(self) -> bool:
        return self._problem.optimum.y + 1e-8 >= self._problem.state.current_best.y

    @property
    def evaluations(self) -> int:
        return self._problem.state.evaluations

    @property
    def current_best_y(self) -> float:
        return self._problem.state.current_best.y

    @property
    def progress(self) -> float:
        optimum, curr_best = self._problem.optimum.y, self.current_best_y
        if optimum >= 0:
            if curr_best == 0:
                return 1
            else:
                return optimum / curr_best
        else:
            return curr_best / optimum


if __name__ == "__main__":
    problem_c = ProblemCocoex.get(dim=2, fun=1, inst=1)
    problem_i = ProblemIoh.get(dim=2, fun=1, inst=1)
