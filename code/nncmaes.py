# TODO: Add dtype to np.ndarray type hints e.g. [None, np.float64]
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Collection
from contextlib import suppress
from numbers import Real
import sys
from typing import Any, Callable

import numpy as np

with suppress(ImportError):
    import cocoex  # type: ignore
import ioh  # type: ignore
from cma import CMAEvolutionStrategy  # type: ignore
from modcma import AskTellCMAES  # type: ignore


def get_seed_np() -> np.uint32:
    _, keys, *_ = np.random.get_state()
    return keys[0]  # type: ignore[return-value] # pyright: ignore [reportReturnType]


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
        """<https://github.com/numbbo/coco/blob/master/code-experiments/src/coco_problem.c#L443-L444>"""

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
        """
        >>> ProblemCocoex.get(dim=2, fun=1, inst=1)
        ProblemCocoex(<cocoex.interface.Problem(), id='bbob_f001_i01_d02'>)
        """
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
        self.__problem = problem

    def __call__(self, x) -> float:
        return self.__problem(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.__problem)})"

    @property
    def function_id(self) -> int:
        return self.__problem.id_function

    @property
    def dimension(self) -> int:
        return self.__problem.dimension

    @property
    def bounds_lower(self) -> np.ndarray:
        return self.__problem.lower_bounds

    @property
    def bounds_upper(self) -> np.ndarray:
        return self.__problem.upper_bounds

    @property
    def final_target_hit(self) -> bool:
        return self.__problem.final_target_hit

    @property
    def evaluations(self) -> int:
        return self.__problem.evaluations

    @property
    def current_best_y(self) -> float:
        return self.__problem.best_observed_fvalue1


class ProblemIoh(Problem):
    @staticmethod
    def get(*, dim, fun, inst) -> ProblemIoh:
        """
        >>> ProblemIoh.get(dim=2, fun=1, inst=1)
        ProblemIoh(<RealSingleObjectiveProblem 1. Sphere (iid=1 dim=2)>)
        """
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
        self.__problem = problem

    def __call__(self, x) -> float:
        return self.__problem(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.__problem)})"

    @property
    def function_id(self) -> int:
        return self.__problem.meta_data.problem_id

    @property
    def dimension(self) -> int:
        return self.__problem.meta_data.n_variables

    @property
    def bounds_lower(self) -> np.ndarray:
        return self.__problem.bounds.lb

    @property
    def bounds_upper(self) -> np.ndarray:
        return self.__problem.bounds.ub

    @property
    def final_target_hit(self) -> bool:
        return self.__problem.optimum.y + 1e-8 >= self.__problem.state.current_best.y

    @property
    def evaluations(self) -> int:
        return self.__problem.state.evaluations

    @property
    def current_best_y(self) -> float:
        return self.__problem.state.current_best.y

    @property
    def progress(self) -> float:
        optimum, curr_best = self.__problem.optimum.y, self.current_best_y
        if optimum >= 0:
            if curr_best == 0:
                return 1
            else:
                return optimum / curr_best
        else:
            return curr_best / optimum


class Cma(ABC):
    @abstractmethod
    def __init__(self, *, x0: Collection[Real], lb, ub, lambda_=None, verbose=None): ...

    @property
    @abstractmethod
    def pop_size_initial(self) -> int: ...

    @property
    @abstractmethod
    def restarts(self) -> int: ...

    @property
    @abstractmethod
    def evals(self) -> int: ...

    @property
    @abstractmethod
    def mean(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def std(self) -> np.ndarray: ...

    @abstractmethod
    def ask(self) -> np.ndarray: ...

    @abstractmethod
    def tell(self, points, values): ...


class WrappedCma(Cma):
    def __init__(
        self,
        *,
        x0,
        lb=None,
        ub=None,
        lambda_=None,
        verbose=None,
        seed: Callable | Any = get_seed_np,
    ):
        """
        >>> WrappedCma(x0=np.zeros(2), verbose=-9)
        WrappedCma({'evals': 0, 'mean': array([0., 0.]), 'pop_size_initial': 6, 'restarts': 0, 'std': array([2.     , 2.00005])})

        seed:
            Passed to the CMAEvolutionStrategy. If callable, it's passed as seed().
            `Details how CMAEvolutionStrategy uses the seed. <https://github.com/CMA-ES/pycma/blob/development/cma/evolution_strategy.py#L477>`_
        """
        self.__restarts = 0
        self.__x0, self.__verbose, self.__lb, self.__ub = x0, verbose, lb, ub

        self.__seed = seed() if callable(seed) else seed

        self.__es = self.make_cmaes(
            x0=self.__x0,
            lambda_=lambda_,
            lb=self.__lb,
            ub=self.__ub,
            seed=self.__seed,
            verbose=self.__verbose,
        )
        self.__pop_size_initial = self.__es.popsize

    def __repr__(self) -> str:
        return "{class_name}({properties})".format(
            class_name=self.__class__.__name__,
            properties={
                p: getattr(self, p)
                for p in dir(self.__class__)
                if isinstance(getattr(self.__class__, p), property)
            },
        )

    @property
    def pop_size_initial(self) -> int:
        return self.__pop_size_initial

    @property
    def restarts(self) -> int:
        return self.__restarts

    @property
    def evals(self) -> int:
        return self.__es.countevals

    @property
    def mean(self) -> np.ndarray:
        return self.__es.mean

    @property
    def std(self) -> np.ndarray:
        """<https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/evolution_strategy.py#L3160-L3169>"""
        return self.__es.stds

    @staticmethod
    def make_cmaes(*, x0, lambda_, lb, ub, seed, verbose):
        # `Options <https://github.com/CMA-ES/pycma/blob/development/cma/evolution_strategy.py#L415-L524>`_
        return CMAEvolutionStrategy(
            x0,
            2,
            {
                "popsize": lambda_,
                "verbose": verbose,
                "seed": seed,
                # 'bounds': [list(lb), list(ub)], # TODO: when BoundPenalty with np.array does not work
                # 'BoundaryHandler': 'BoundTransform',
                # 'BoundaryHandler': BoundPenalty,
                # 'maxfevals': budget,
            },
        )

    def ask(self):
        if self.__es.stop():
            self.__restarts += 1
            lambda_ = 2**self.__restarts * self.__pop_size_initial
            self.__es = self.make_cmaes(
                x0=self.__x0,
                lambda_=lambda_,
                lb=self.__lb,
                ub=self.__ub,
                seed=self.__seed,
                verbose=self.__verbose,
            )
        return np.array(self.__es.ask())

    def tell(self, points, values):
        self.__es.tell(list(points), [v[0] for v in values])


def default_pop_size(dim):
    """used to be: (4 + np.floor(3 * np.log(dim))).astype(int)"""
    return WrappedCma(x0=np.zeros(dim), lb=None, ub=None).pop_size_initial


class WrappedModcma(Cma):
    def __init__(self, *, x0: np.ndarray, lb=None, ub=None, lambda_=None):
        """
        # >>> WrappedModcma(x0=np.zeros(2)) # TODO
        """
        # `Options <https://github.com/IOHprofiler/ModularCMAES/blob/master/modcma/parameters.py#L23-L313>`_
        self.__es = AskTellCMAES(
            d=x0.shape[-1],
            budget=sys.maxsize,  # None and float('inf') does not work
            # bound_correction='COTN',
            bound_correction="saturate",
            lb=lb,
            ub=ub,
            lambda_=lambda_,
            active=True,
            local_restart="IPOP",
        )
        self.__pop_size_initial = self.__es.parameters.lambda_

    @property
    def pop_size_initial(self) -> int:
        return self.__pop_size_initial

    @property
    def restarts(self) -> int:
        return len(self.__es.parameters.restarts) - 1

    @property
    def evals(self) -> int:
        return self.__es.parameters.used_budget

    def ask(self):
        return np.array(
            [self.__es.ask().squeeze() for _ in range(self.__es.parameters.lambda_)]
        )

    def tell(self, points, values):
        for p, v in zip(points, values):
            self.__es.tell(p[:, None], v)
