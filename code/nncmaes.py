# TODO: np.recarray((2,), dtype=[('y', float), *[(f'x{n+1}', float) for n in range(dim)]])
from __future__ import annotations
from abc import abstractmethod, ABC
import argparse
from collections.abc import Iterable
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps, partial
from itertools import product
import math
from numbers import Real
import os
import pickle
import random
from random import randint
import sys
import time
import types
from typing import (
    runtime_checkable,
    Callable,
    Generic,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Type,
    TypeVar,
)
import warnings

import numpy as np
from numpy import dtype, float64
from numpy.typing import NDArray
from scipy.stats import distributions, kendalltau  # type: ignore[import-untyped]
import tensorflow  # type: ignore[import-untyped]
import tensorflow.compat.v1 as tf  # type: ignore[import-untyped]
import torch
from torch.nn import Module, Linear, Dropout
import torch.nn.functional as F

with suppress(ImportError):
    import cocoex  # type: ignore
import ioh  # type: ignore
from cma import CMAEvolutionStrategy  # type: ignore
from modcma import AskTellCMAES  # type: ignore


def repr_default(self, *attributes, **attrs_with_name) -> str:
    def format_attr(attr) -> str:
        return (
            attr.__name__ if isinstance(attr, (types.FunctionType, type)) else str(attr)
        )

    def format_attributes() -> str:
        return ", ".join(
            [format_attr(a) for a in attributes]
            + [f"{name}={format_attr(a)}" for name, a in attrs_with_name.items()]
        )

    return (
        f"{self.__class__.__name__}({format_attributes()})"
        if self is not None
        else f"({format_attributes()})"
    )


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def get(*, dim: int, fun: int, inst: int, budget_coef: int) -> Problem: ...

    @abstractmethod
    def __call__(self, x) -> np.float64: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    @abstractmethod
    def function_id(self) -> int: ...

    @property
    @abstractmethod
    def instance(self) -> int: ...

    @property
    @abstractmethod
    def budget(self) -> Optional[int]: ...

    # TODO: change bounds to namedtuple?
    @property
    @abstractmethod
    def bounds_lower(self) -> NDArray[np.float64]: ...

    @property
    @abstractmethod
    def bounds_upper(self) -> NDArray[np.float64]: ...

    @property
    @abstractmethod
    def current_best_y(self) -> float: ...

    @property
    @abstractmethod
    def evaluations(self) -> int: ...

    @property
    @abstractmethod
    def evals_left(self) -> Optional[int]: ...

    @property
    @abstractmethod
    def all_evals_used(self) -> bool: ...

    @property
    @abstractmethod
    def final_target_hit(self) -> bool:
        """<https://github.com/numbbo/coco/blob/v2.6.3/code-experiments/src/coco_problem.c#L443-L444>"""

    def __repr__(self) -> str:
        return repr_default(self, dim=self.dimension, fun=self.function_id, inst=self.instance, budget=self.budget)

    def __str__(self) -> str:
        return '--'.join((
            f"d-{self.dimension}",
            f"f-{self.function_id}",
            f"i-{self.instance}",
            f"b-{self.budget}",
        ))

    def is_outside(self, point, *, tol=0) -> bool:
        """Returns `True` if the point is more than `tol` outside of the bounds."""
        return (
            (point < self.bounds_lower - tol) | (point > self.bounds_upper + tol)
        ).any()


class ProblemCocoex(Problem):
    @staticmethod
    def get(
        *, dim: int, fun: int, inst: int, budget_coef: int | None = None
    ) -> ProblemCocoex:
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
            ),
            budget_coef=budget_coef,
        )

    def __init__(
        self,
        problem: cocoex.interface.Problem,  # pyright: ignore [reportAttributeAccessIssue]
        *,
        budget_coef: int | None = None,
    ):
        if not isinstance(problem, cocoex.interface.Problem):  # pyright: ignore [reportPossiblyUnboundVariable, reportAttributeAccessIssue]
            raise ValueError(f"Wrong problem type: {type(problem)}")
        self.__problem = problem
        self.__budget = (
            budget_coef * self.dimension if budget_coef is not None else None
        )

    def __call__(self, x) -> np.float64:
        try:
            if self.evaluations >= self.__budget:  # type: ignore[operator]
                return np.float64(np.nan)
        except TypeError:
            pass
        return self.__problem(x)

    @property
    def dimension(self) -> int:
        return self.__problem.dimension

    @property
    def function_id(self) -> int:
        return self.__problem.id_function

    @property
    def instance(self) -> int:
        return self.__problem.id_instance

    @property
    def budget(self) -> Optional[int]:
        return self.__budget

    @property
    def bounds_lower(self) -> NDArray[np.float64]:
        return self.__problem.lower_bounds

    @property
    def bounds_upper(self) -> NDArray[np.float64]:
        return self.__problem.upper_bounds

    @property
    def current_best_y(self) -> float:
        return self.__problem.best_observed_fvalue1

    @property
    def evaluations(self) -> int:
        return self.__problem.evaluations

    @property
    def evals_left(self) -> Optional[int]:
        return self.__budget - self.evaluations if self.__budget is not None else None

    @property
    def all_evals_used(self) -> bool:
        return self.evaluations >= self.__budget if self.__budget is not None else False

    @property
    def final_target_hit(self) -> bool:
        return bool(self.__problem.final_target_hit)

    def __str__(self) -> str:
        return f"pycma--{super().__str__()}"

class ProblemIoh(Problem):
    @staticmethod
    def get(
        *, dim: int, fun: int, inst: int, budget_coef: int | None = None
    ) -> ProblemIoh:
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
            ),
            budget_coef=budget_coef,
        )

    def __init__(
        self, problem: ioh.iohcpp.problem.BBOB, *, budget_coef: int | None = None
    ):
        if not isinstance(problem, ioh.iohcpp.problem.BBOB):
            raise ValueError(f"Wrong problem type: {type(problem)}")
        self.__problem = problem
        self.__budget = (
            budget_coef * self.dimension if budget_coef is not None else None
        )

    def __call__(self, x) -> np.float64:
        try:
            if self.evaluations >= self.__budget:  # type: ignore[operator]
                return np.float64(np.nan)
        except TypeError:
            pass
        return np.float64(self.__problem(x))

    @property
    def dimension(self) -> int:
        return self.__problem.meta_data.n_variables

    @property
    def function_id(self) -> int:
        return self.__problem.meta_data.problem_id

    @property
    def instance(self) -> int:
        return self.__problem.meta_data.instance

    @property
    def budget(self) -> Optional[int]:
        return self.__budget

    @property
    def bounds_lower(self) -> NDArray[np.float64]:
        return self.__problem.bounds.lb

    @property
    def bounds_upper(self) -> NDArray[np.float64]:
        return self.__problem.bounds.ub

    @property
    def current_best_y(self) -> float:
        return self.__problem.state.current_best.y

    @property
    def evaluations(self) -> int:
        return self.__problem.state.evaluations

    @property
    def evals_left(self) -> Optional[int]:
        return self.__budget - self.evaluations if self.__budget is not None else None

    @property
    def all_evals_used(self) -> bool:
        return self.evaluations >= self.__budget if self.__budget is not None else False

    @property
    def final_target_hit(self) -> bool:
        return self.__problem.optimum.y + 1e-8 >= self.__problem.state.current_best.y

    @property
    def delta_to_optimum(self) -> float:
        return self.current_best_y - self.__problem.optimum.y

    def __str__(self) -> str:
        return f"iohexp--{super().__str__()}"


DIM = int
POPSIZE = int
ARCHSIZE = int
X = np.ndarray[tuple[DIM], np.dtype[np.float64]]
Y = np.ndarray[tuple[Literal[1]], np.dtype[np.float64]]
XPop = np.ndarray[tuple[POPSIZE, DIM], np.dtype[np.float64]]
YPop = np.ndarray[tuple[POPSIZE, Literal[1]], np.dtype[np.float64]]
XArch = np.ndarray[tuple[ARCHSIZE, DIM], np.dtype[np.float64]]
YArch = np.ndarray[tuple[ARCHSIZE, Literal[1]], np.dtype[np.float64]]


class Cma(ABC):
    @abstractmethod
    def __init__(self, *, x0: X, lb, ub, lambda_=None, verbose=None): ...

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
    def mean(self) -> X: ...

    @property
    @abstractmethod
    def std(self) -> X: ...

    @abstractmethod
    def mahalanobis_norm(self, delta) -> np.float64: ...

    @abstractmethod
    def ask(self) -> XPop: ...

    @abstractmethod
    def tell(self, points: Iterable[X], values: Iterable[Y]): ...

    def __str__(self) -> str:
        return f"{type(self).__name__.lower()}--init-pop-{self.pop_size_initial}"

class Pycma(Cma):
    def __init__(
        self,
        *,
        x0: Iterable[Real],
        lb=None,
        ub=None,
        lambda_=None,
        verbose=None,
        seed=None,
    ):
        """
        >>> Pycma(x0=np.zeros(2), verbose=-9)
        Pycma({'evals': 0, 'mean': array([0., 0.]), 'pop_size_initial': 6, 'restarts': 0, 'std': array([2.     , 2.00005])})

        `Details how CMAEvolutionStrategy uses the seed. <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/evolution_strategy.py#L473-L474>`_
        """
        self.__prev_evals = 0
        self.__restarts = 0
        self.__x0, self.__verbose, self.__lb, self.__ub = x0, verbose, lb, ub

        self.__seed = seed

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
        return self.__prev_evals + self.__es.countevals

    @property
    def mean(self) -> X:
        return self.__es.mean  # pyright: ignore [reportReturnType]

    @property
    def std(self) -> X:
        """<https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/evolution_strategy.py#L3160-L3169>"""
        return self.__es.stds

    def mahalanobis_norm(self, delta) -> np.float64:
        return self.__es.mahalanobis_norm(delta)

    @staticmethod
    def make_cmaes(*, x0, lambda_, lb, ub, seed, verbose):
        # `Options <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/evolution_strategy.py#L415-L517>`_
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

    def ask(self) -> XPop:
        if self.__es.stop():
            self.__prev_evals += self.__es.countevals
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

    def tell(self, points: Iterable[X], values: Iterable[Y]):
        self.__es.tell(list(points), [v.item() for v in values])


def default_pop_size(dim):
    """used to be: (4 + np.floor(3 * np.log(dim))).astype(int)"""
    return Pycma(x0=np.zeros(dim), lb=None, ub=None).pop_size_initial


class Modcma(Cma):
    def __init__(self, *, x0: X, lb=None, ub=None, lambda_=None):
        """
        # >>> Modcma(x0=np.zeros(2)) # TODO
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

    def ask(self) -> XPop:
        return np.array(
            [self.__es.ask().squeeze() for _ in range(self.__es.parameters.lambda_)]
        )

    def tell(self, points: Iterable[X], values: Iterable[Y]):
        for p, v in zip(points, values):
            self.__es.tell(p[:, None], v)


N = TypeVar("N", bound=int)
S = TypeVar("S", bound=tuple)


@dataclass
class Prediction(Generic[N]):
    mean: np.ndarray[tuple[N, Literal[1]], np.dtype[np.float64]]
    std: np.ndarray[tuple[N, Literal[1]], np.dtype[np.float64]]


class Norm(Protocol):
    def __call__(self, vector: NDArray[np.float64], /) -> np.float64: ...


class NormFactory(ABC, Norm):
    @abstractmethod
    def __init__(self, *args, **kwargs): ...


class Mahalanobis(NormFactory):
    def __init__(self, *, es):
        self.__es = es

    def __call__(self, vector: NDArray[np.float64]) -> np.float64:
        return self.__es.mahalanobis_norm(vector)


class Subset(ABC):
    @abstractmethod
    def __call__(
        self,
        *,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        es: Cma,
    ) -> NDArray[np.uint]: ...

    @abstractmethod
    def __repr__(self) -> str: ...


class LastN(Subset):
    def __repr__(self) -> str:
        return self.__class__.__name__

    def __call__(
        self,
        *,
        x_train: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[N, Literal[1]], np.dtype[np.float64]],
        x_test: XPop,
        es: Cma,
    ) -> NDArray[np.uint]:
        n_total, dim = x_train.shape
        max_model_size = max(x_test.shape[0], dim * (dim + 3) + 2)
        return np.arange(n_total).astype(np.uint)[-max_model_size:]


class ClosestToAnyTestPoint(Subset):
    """<https://github.com/bajeluk/surrogate-cmaes/blob/fe33fda66e11c6949fe857289184007788c34794/src/data/Archive.m#L204-L244>"""

    def __call__(
        self,
        *,
        x_train: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[N, Literal[1]], np.dtype[np.float64]],
        x_test: XPop,
        es: Cma,
    ) -> NDArray[np.uint]:
        raise NotImplementedError  # TODO .min(axis=2).argpartition(n_max-1).astype(np.uint)[:n_max]"""


class ClosestToEachTestPoint(Subset):
    """Inspired by Matlab `surrogate-cmaes <https://github.com/bajeluk/surrogate-cmaes/blob/fe33fda66e11c6949fe857289184007788c34794/src/data/Archive.m#L128-L202>`_
    >>> ClosestToEachTestPoint(n_max_coef=2)(y_train=None, es=None,
    ...     x_train=np.array([                 # idx
    ...         [-3, -3],           [-3,  3],  # 0 1
    ...                   [ 0,  0],            # 2
    ...         [ 3, -3],           [ 3,  3]   # 3 4
    ...     ]),
    ...     x_test=np.array([
    ...         [-2, -2], [-2,  2],
    ...         [ 2, -2], [ 2,  2]
    ...     ])
    ... )
    array([0, 1, 3, 4], dtype=uint64)

    >>> ClosestToEachTestPoint(n_max_coef=2)(y_train=None, es=None,
    ... x_train=np.array([                                                         # index
    ...     [-4, -4], [-4, -3],                               [-4,  3], [-4,  4],  #  0  1  2  3
    ...     [-3, -4],                                                   [-3,  4],  #  4  5
    ...                         [-1, -1],           [-1,  1],                      #  6  7
    ...                                   [ 0,  0],                                #  8
    ...                         [ 1, -1],           [ 1,  1],                      #  9 10
    ...     [ 3, -4],                                                   [ 3,  4],  # 11 12
    ...     [ 4, -4], [ 4, -3],                               [ 4,  3], [ 4,  4],  # 13 14 15 16
    ... ]),
    ... x_test=np.array([
    ...               [-3, -3],                               [-3,  3],
    ...               [ 3, -3],                               [ 3,  3],
    ... ]))
    array([ 1,  2, 11, 12], dtype=uint64)

    >>> ClosestToEachTestPoint(norm_max=lambda d: 1, n_max_coef=4)(y_train=None, es=None,
    ... x_train=np.array([
    ... # idx:  0                   1                             2         3
    ...     [-4, -4],           [-4, -2],                     [-4,  3], [-4,  4],
    ... # idx:                                          4                   5
    ...                                             [-3,  2],           [-3,  4],
    ... # idx:  6                                                 7
    ...     [-2, -4],                                         [-2,  3],
    ... # idx:  8                   9                                      10
    ...     [ 3, -4],           [ 3, -2],                               [ 3,  4],
    ... # idx: 11        12                                      13        14
    ...     [ 4, -4], [ 4, -3],                               [ 4,  3], [ 4,  4],
    ... ]),
    ... x_test=np.array([
    ...               [-3, -3],                               [-3,  3],
    ...               [ 3, -3],                               [ 3,  3],
    ... ]))
    array([ 2,  8, 10,  4,  9, 13,  5, 12], dtype=uint64)
    """

    n_max_coef = 20

    @staticmethod
    def norm_max(dim):
        """<https://github.com/bajeluk/surrogate-cmaes/blob/fe33fda66e11c6949fe857289184007788c34794/src/data/Archive.m#L255>"""
        return 4 * np.sqrt(distributions.chi2.ppf(0.99, df=dim))

    def __init__(
        self,
        *,
        norm: Norm | Type[NormFactory] = np.linalg.norm,
        norm_max: Optional[Callable[[int], np.float64]] = None,
        n_max_coef=None,
    ):
        self.__norm = norm
        self.__n_max_coef = n_max_coef if n_max_coef is not None else self.n_max_coef
        self.__norm_max = norm_max if norm_max is not None else self.norm_max

    def __repr__(self) -> str:
        return repr_default(self, self.__norm, n_max_coef=self.__n_max_coef)

    def __call__(
        self,
        *,
        x_train: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[N, Literal[1]], np.dtype[np.float64]],
        x_test: XPop,
        es: Cma,
    ) -> NDArray[np.uint]:
        n_max = self.__n_max_coef * x_train.shape[1]
        n_total, dim = x_train.shape
        norm_max = self.__norm_max(dim)
        # n_per_point = int(n_max / x_test.shape[0])
        norm = self.__norm if not isinstance(self.__norm, type) else self.__norm(es=es)

        # distance of train point (row) and test point (column)
        norms_2d = np.apply_along_axis(  # type: ignore[call-overload]
            norm,
            axis=2,
            arr=np.repeat(x_train[:, None, :], x_test.shape[0], axis=1) - x_test,
        )
        # Argsorts each column (train point index with smallest norm on top)
        # then flattens them by rows i.e. each test points matters the same*.
        near_indices = np.flatnonzero((norms_2d <= norm_max).any(axis=1))
        if n_max >= n_total:
            idx = near_indices.astype(np.uint)
        else:
            unique_indices, their_positions = np.unique(
                norms_2d.argpartition(kth=np.arange(n_max), axis=0)
                .astype(np.uint)[:n_max]
                .flatten(),
                return_index=True,
            )
            near_mask = np.isin(unique_indices, near_indices)
            idx = unique_indices[near_mask][their_positions[near_mask].argsort()][
                :n_max
            ]
        if __debug__:
            print(
                f"subset-max-norm={norms_2d[idx].min(axis=1).max()} | norm-max={norm_max}"
            )
        return idx


class InNormRange(Subset):
    norm_max = 2

    def __init__(self, *, norm: Norm | Type[NormFactory]):
        self.__norm = norm

    def __repr__(self) -> str:
        return repr_default(self, norm_max=self.norm_max)

    def __call__(
        self,
        *,
        x_train: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
        y_train: np.ndarray[tuple[N, Literal[1]], np.dtype[np.float64]],
        x_test: XPop,
        es: Cma,
    ) -> NDArray[np.uint]:
        n_total, dim = x_train.shape
        norm = self.__norm if not isinstance(self.__norm, type) else self.__norm(es=es)
        norms = np.array([norm(d) for d in x_train - es.mean])
        near_indices = np.flatnonzero(norms <= self.norm_max)
        idx = np.arange(n_total).astype(np.uint)[near_indices]
        return idx


def get_mean_and_std(
    samples, *, weights=None
) -> tuple[np.float64 | NDArray[np.float64], np.float64 | NDArray[np.float64]]:
    """
    >>> np.seterr('raise'); np.array([[-1e105], [6e154]]).std(axis=0)
    Traceback (most recent call last):
        ...
    FloatingPointError: overflow encountered in multiply
    """
    if samples.size > 0:
        if weights is not None:
            mean = np.average(samples, weights=weights, axis=0)
            std = np.sqrt(
                np.atleast_2d(
                    np.cov(samples, rowvar=False, ddof=0, aweights=weights.squeeze())
                ).diagonal()
            )
        else:
            mean = samples.mean(axis=0)
            std = samples.std(axis=0)
        std[std == 0] = 1  # when std==0, (-mean) makes any value (==0)
        if (infs := np.isposinf(std)).any():
            std[infs] = np.finfo(np.float64).max
    else:
        mean = float64(0)
        std = float64(1)
    return mean, std


class Archive(NamedTuple):
    x: XArch
    y: YArch


class Model(Protocol):
    def __call__(
        self, features: np.ndarray[tuple[N, DIM], np.dtype[np.float64]]
    ) -> Prediction[N]: ...


def predict_zeros_and_ones(
    features: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
) -> Prediction[N]:
    shape = features.shape[0], 1
    return Prediction(np.zeros(shape), np.ones(shape))


def predict_random(
    features: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
) -> Prediction[N]:
    shape = features.shape[0], 1
    return Prediction(
        np.random.randint(-100, 100, shape).astype(np.float64),
        np.random.rand(*shape),
    )


class ModelFactory(ABC):
    @abstractmethod
    def __call__(
        self,
        *,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]],
    ) -> Model: ...


class NN:
    """Taken from `RAFs <https://github.com/YanasGH/RAFs/blob/6a0ec46a7d9cd830e7d8e74358643aee1f65323d/main_experiments/rafs.py#L15-L91>`_"""

    def __init__(
        self,
        x_dim,
        y_dim,
        hidden_size,
        init_stddev_1_w,
        init_stddev_1_b,
        init_stddev_2_w,
        n,
        learning_rate,
        ens,
    ):
        # setting up as for a usual NN
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.n = n
        self.learning_rate = learning_rate

        # set up NN
        self.inputs = tf.placeholder(tf.float64, [None, x_dim], name="inputs")
        self.y_target = tf.placeholder(tf.float64, [None, y_dim], name="target")

        activation_fns = [
            tensorflow.keras.activations.selu,  # pyright: ignore [reportAttributeAccessIssue]
            tf.nn.tanh,
            tensorflow.keras.activations.gelu,  # pyright: ignore [reportAttributeAccessIssue]
            tensorflow.keras.activations.softsign,  # pyright: ignore [reportAttributeAccessIssue]
            tf.math.erf,
            tf.nn.swish,
            tensorflow.keras.activations.linear,  # pyright: ignore [reportAttributeAccessIssue]
        ]

        if ens <= len(activation_fns) - 1:
            self.layer_1_w = tf.layers.Dense(
                hidden_size,
                activation=activation_fns[ens],
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=init_stddev_1_w
                ),
                bias_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=init_stddev_1_b
                ),
            )
        else:
            af_ind = randint(0, 3)
            self.layer_1_w = tf.layers.Dense(
                hidden_size,
                activation=activation_fns[af_ind],
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=init_stddev_1_w
                ),
                bias_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=init_stddev_1_b
                ),
            )

        self.layer_1 = self.layer_1_w.apply(self.inputs)

        self.output_w = tf.layers.Dense(
            y_dim,
            activation=None,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(
                mean=0.0, stddev=init_stddev_2_w
            ),
        )

        self.output = self.output_w.apply(self.layer_1)

        # set up loss and optimiser - this is modified later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        self.mse_ = (
            1
            / tf.shape(self.inputs, out_type=tf.int64)[0]
            * tf.reduce_sum(tf.square(self.y_target - self.output))
        )
        self.loss_ = (
            1
            / tf.shape(self.inputs, out_type=tf.int64)[0]
            * tf.reduce_sum(tf.square(self.y_target - self.output))
        )
        self.optimizer = self.opt_method.minimize(self.loss_)
        return

    def get_weights(self, sess):
        """method to return current params"""

        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.output_w.kernel]
        w1, b1, w2 = sess.run(ops)

        return w1, b1, w2

    def anchor(self, sess, lambda_):  # lambda_anchor
        """regularise around initial parameters"""

        w1, b1, w2 = self.get_weights(sess)

        # get initial params
        self.w1_init, self.b1_init, self.w2_init = w1, b1, w2

        loss = lambda_[0] * tf.reduce_sum(
            tf.square(self.w1_init - self.layer_1_w.kernel)
        )
        loss += lambda_[1] * tf.reduce_sum(
            tf.square(self.b1_init - self.layer_1_w.bias)
        )
        loss += lambda_[2] * tf.reduce_sum(
            tf.square(self.w2_init - self.output_w.kernel)
        )

        # combine with original loss
        self.loss_ = self.loss_ + 1 / tf.shape(self.inputs, out_type=tf.int64)[0] * loss
        self.optimizer = self.opt_method.minimize(self.loss_)
        return

    def predict(self, x, sess):
        """predict method"""

        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred


class Raf(ModelFactory):
    data_noise: float = 0.01
    epochs: int = 1000

    def __init__(
        self,
        *,
        data_noise: Optional[float] = None,
        epochs: Optional[int] = None,
        debug: bool = False,
    ):
        """
        data_noise: estimated noise variance, feel free to experiment with different values
        """
        self.__data_noise = data_noise if data_noise is not None else self.data_noise
        self.__epochs = epochs if epochs is not None else self.epochs
        self.__debug = debug

    def __repr__(self) -> str:
        return repr_default(
            self,
            data_noise=self.__data_noise,
            **(dict(epochs=self.__epochs) if self.__epochs != self.epochs else {}),
        )

    def __call__(
        self,
        *,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None,
    ) -> Model:
        data_noise = self.__data_noise
        """Taken form `RAFs <https://github.com/YanasGH/RAFs/blob/6a0ec46a7d9cd830e7d8e74358643aee1f65323d/main_experiments/rafs.py#L94-L157>`_"""
        X_train, y_train, X_val = x_train, y_train, x_test

        n = X_train.shape[0]
        x_dim = X_train.shape[1]
        y_dim = y_train.shape[1]
        if __debug__ and self.__debug:
            _max_model_size = max(
                x_test.shape[0], x_dim * (x_dim + 3) + 2
            )  # for kendall only
            _n_kendall_archive = _max_model_size - 1

        n_ensembles = 5
        hidden_size = 100
        init_stddev_1_w = np.sqrt(10)
        init_stddev_1_b = init_stddev_1_w  # set these equal
        init_stddev_2_w = 1.0 / np.sqrt(hidden_size)  # normal scaling
        lambda_anchor = data_noise / (
            np.array([init_stddev_1_w, init_stddev_1_b, init_stddev_2_w]) ** 2
        )

        n_epochs = self.__epochs
        learning_rate = 0.01

        NNs = []
        y_prior = []
        tf.reset_default_graph()
        sess = tf.Session()

        # loop to initialise all ensemble members, get priors
        for ens in range(0, n_ensembles):
            NNs.append(
                NN(
                    x_dim,
                    y_dim,
                    hidden_size,
                    init_stddev_1_w,
                    init_stddev_1_b,
                    init_stddev_2_w,
                    n,
                    learning_rate,
                    ens,
                )
            )

            # initialise only unitialized variables - stops overwriting ensembles already created
            global_vars = tf.global_variables()
            is_not_initialized = sess.run(
                [tf.is_variable_initialized(var) for var in global_vars]
            )
            not_initialized_vars = [
                v for (v, f) in zip(global_vars, is_not_initialized) if not f
            ]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

            # do regularisation now that we've created initialisations
            NNs[ens].anchor(
                sess, lambda_anchor
            )  # Do that if you want to minimize the anchored loss

            # save their priors
            y_prior.append(NNs[ens].predict(X_val, sess))

        for ens in range(0, n_ensembles):
            feed_b = {}
            feed_b[NNs[ens].inputs] = X_train
            feed_b[NNs[ens].y_target] = y_train
            if __debug__ and self.__debug:
                print("\nNN:", ens)

            ep_ = 0
            while ep_ < n_epochs:
                ep_ += 1
                blank = sess.run(NNs[ens].optimizer, feed_dict=feed_b)  # noqa: F841
                if ep_ % (n_epochs / 5) == 0:
                    loss_mse = sess.run(NNs[ens].mse_, feed_dict=feed_b)
                    loss_anch = sess.run(NNs[ens].loss_, feed_dict=feed_b)
                    if __debug__ and self.__debug:
                        print(
                            "epoch:",
                            ep_,
                            ", mse_",
                            np.round(loss_mse * 1e3, 3),
                            ", loss_anch",
                            np.round(loss_anch * 1e3, 3),
                            "tau-train={:>5.2f}".format(
                                kendalltau(
                                    y_train[-_n_kendall_archive:],  # pyright: ignore [reportOperatorIssue,reportPossiblyUnboundVariable]
                                    np.array(
                                        NNs[ens].predict(
                                            X_train[-_n_kendall_archive:],  # pyright: ignore [reportOperatorIssue,reportPossiblyUnboundVariable]
                                            sess,
                                        )
                                    ),
                                ).statistic
                            ),
                        )
                    # the anchored loss is minimized, but it's useful to keep an eye on mse too

        if __debug__ and self.__debug:
            _tau = kendalltau(
                y_train[-_n_kendall_archive:],  # pyright: ignore [reportOperatorIssue,reportPossiblyUnboundVariable]
                np.mean(
                    np.array(
                        [nn.predict(X_train[-_n_kendall_archive:], sess) for nn in NNs]  # pyright: ignore [reportOperatorIssue,reportPossiblyUnboundVariable]
                    ),
                    axis=0,
                ),
            ).statistic
            # if _tau < 0.7:
            #     breakpoint()
            print(f"tau-train-ens={_tau}")

        def raf(features):
            y_pred = np.array([nn.predict(features, sess) for nn in NNs])
            return Prediction(
                np.mean(y_pred, axis=0),
                np.sqrt(np.square(np.std(y_pred, axis=0, ddof=1)) + data_noise),
            )

        return raf


class NeuralNetwork(Module):
    def __init__(self, activation, *, width=128, dropout_p=0.0, dim):
        super(NeuralNetwork, self).__init__()
        self.width = width
        self.dropout_p = dropout_p
        self.activation = activation

        # self.n_0 = nn.BatchNorm1d(dim)
        self.l_1 = Linear(dim, self.width)
        self.drop = Dropout(self.dropout_p) if self.dropout_p > 0 else None
        # self.l_2 = nn.Linear(self.width,self.width)
        self.l__1 = Linear(self.width, 1)
        self.transforms = (
            [
                # self.n_0,
                self.l_1,
            ]
            + ([self.drop] if self.drop is not None else [])
            + [
                activation,
                self.l__1,
            ]
        )

    def forward(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __str__(self) -> str:
        return "-".join(
            [
                "".join(
                    [str(tmp)[0] for tmp in self.transforms if isinstance(tmp, Module)]
                ),
                str(self.width),
                self.activation.__name__,
            ]
            + ([f"drop{self.dropout_p}"] if self.dropout_p > 0 else [])
        )


def train_network(
    network,
    x,
    y,
    *,
    weights,
    plot=None,
    epochs=1000,
    mse_stop=-np.inf,
    lr=0.001,
    device,
):
    best_loss = np.inf
    best_model = network.state_dict()

    if plot:
        (fig, ax, dh) = plot

    # lr = 3e-4
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # loss_fn = torch.nn.MSELoss()
    def loss_fn(output, target, weights):
        # return torch.mean(torch.abs((output - target)))
        # return torch.mean(torch.abs(weights * (output - target)))
        # return torch.mean(torch.abs(output - target) / target)
        # return torch.mean(torch.abs((output - target) / (1 + target - target.min())))
        # return torch.mean(torch.log(output + 1) - torch.log(target + 1))
        return torch.mean((weights * (output - target)) ** 2)

    loss = np.inf
    losses = []

    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)

    weights = torch.Tensor(weights.squeeze()[:, None])

    epochs_iter = range(1, epochs + 1)
    for epoch in epochs_iter:
        network.train()
        optimizer.zero_grad()
        loss_fn(y, network(x), weights).backward()
        optimizer.step()

        network.eval()
        with torch.no_grad():
            loss = loss_fn(y, network(x), weights).item()
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_model = deepcopy(network.state_dict())
        if loss < mse_stop:
            break

        if epoch % 200 == 0 and plot:
            ax.clear()  # pyright: ignore [reportPossiblyUnboundVariable]
            ax.semilogy(losses, ",")  # pyright: ignore [reportPossiblyUnboundVariable]
            dh.update(fig)  # pyright: ignore [reportPossiblyUnboundVariable]

    network.load_state_dict(best_model)
    return losses


class Eaf(ModelFactory):
    width = 128
    lr = 0.01
    epochs = 1000
    mse_stop = -np.inf

    def __init__(
        self,
        *,
        weights=None,
        width=None,
        lr=None,
        epochs=None,
        mse_stop=None,
        plot=None,
        **ignored,
    ):
        """Train and predict ensemble of NNs with different activation functions"""

    def __repr__(self) -> str:
        return type(self).__name__

    def __call__(
        self,
        *,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        weights: Optional[NDArray[np.float64]] = None,
    ) -> Model:
        # lr=0.06
        # width=1024
        afs = [
            F.sigmoid,
            # F.hardsigmoid,
            F.relu,
            # F.softplus,
            # F.gelu,
            F.silu,
            # F.mish,
            # F.leaky_relu,
            # F.elu,
            # F.celu,
            # F.selu,
            # F.tanh,
            F.hardtanh,
            # F.softsign,
            torch.erf,
        ]
        dim = x_train.shape[-1]
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        nets = {
            f.__name__: NeuralNetwork(f, width=self.width, dropout_p=0.0, dim=dim).to(
                device
            )
            for f in afs
        }

        if weights is None:
            # norms = np.linalg.norm(x_train - x_test.mean(axis=0), axis=1)
            # weights = (1 / (1 + norms))**2
            # weights = (1 / (1 + norms))**4
            weights = np.ones_like(y_train)

        losses = {}
        for name, net in nets.items():
            losses[name] = train_network(
                net,
                x_train,
                y_train,
                weights=weights,
                epochs=self.epochs,
                mse_stop=self.mse_stop,
                lr=self.lr,
                device=device,
            )

        if __debug__:
            with torch.no_grad():
                _max_model_size = max(
                    x_test.shape[0], dim * (dim + 3) + 2
                )  # for kendall only
                _n_kendall_archive = _max_model_size - 1
                x = torch.Tensor(x_train[-_n_kendall_archive:]).to(device)
                y = np.array([net.eval()(x).cpu().numpy() for net in nets.values()])
                tau = kendalltau(
                    y_train[-_n_kendall_archive:], np.mean(y, axis=0)
                ).statistic  # pyright: ignore [reportOperatorIssue,reportPossiblyUnboundVariable]
                # if tau < 0.7:
                #     breakpoint()
                print("tau-train-ens={}".format(tau))

        def eaf(
            features: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
        ) -> Prediction[N]:
            x_test_tsr = torch.Tensor(x_test).to(device)
            y_preds = {}
            with torch.no_grad():
                for name, net in nets.items():
                    y_preds[name] = net.eval()(x_test_tsr).cpu().numpy()

            # y_preds_scaled = {name: scale_y_back(y_) for name, y_ in y_preds.items()}
            y_pred_arr = np.array(list(y_preds.values()))

            pred_mean = y_pred_arr.mean(axis=0)
            pred_std = y_pred_arr.std(axis=0)

            if len(pred_mean) < len(x_test):
                raise ValueError("len(pred_mean) < len(x_test)")
            return Prediction(pred_mean, pred_std)

        return eaf


class SurrogateCallable(Protocol):
    def __call__(
        self,
        *,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        es: Cma,
    ) -> Model: ...


class Transformation(ABC):
    def __repr__(self) -> str:
        return type(self).__name__

    @abstractmethod
    def __init__(
        self,
        data: np.ndarray[tuple[N], np.dtype[np.float64]],
        /,
        *,
        x_train=None,
        y_train=None,
        x_test=None,
        es: Optional[Cma] = None,
        weights=None,
    ): ...

    @abstractmethod
    def transform(
        self, data: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]: ...

    @abstractmethod
    def transform_inv(
        self, data: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]: ...

    @abstractmethod
    def transform_inv_std(
        self, std: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]: ...


class ShiftAndScale(Transformation):
    @property
    @abstractmethod
    def shift(self) -> np.float64 | NDArray[np.float64]: ...

    @property
    @abstractmethod
    def scale(self) -> np.float64 | NDArray[np.float64]: ...

    def transform(
        self, data: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]:
        return (data - self.shift) / self.scale

    def transform_inv(
        self, data: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]:
        return data * self.scale + self.shift

    def transform_inv_std(
        self, std: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]:
        return std * self.scale


class Standardization(ShiftAndScale):
    def __init__(
        self,
        data: np.ndarray[tuple[N], np.dtype[np.float64]],
        /,
        *,
        x_train=None,
        y_train=None,
        x_test=None,
        es: Cma,
        weights=None,
    ):
        self.__mean, self.__std = get_mean_and_std(data, weights=weights)

    @property
    def shift(self) -> np.float64 | NDArray[np.float64]:
        return self.__mean

    @property
    def scale(self) -> np.float64 | NDArray[np.float64]:
        return self.__std


class ShiftAndScaleByEs(ShiftAndScale):
    def __init__(
        self,
        data: np.ndarray[tuple[N], np.dtype[np.float64]],
        *,
        x_train=None,
        y_train=None,
        x_test=None,
        es: Cma,
        weights=None,
    ):
        self.__mean = es.mean
        self.__std = es.std

    @property
    def shift(self):
        return self.__mean

    @property
    def scale(self):
        return self.__std


class MinAdjustedLog(Transformation):
    min_offset: float = 1e-12

    def __init__(
        self,
        data: np.ndarray[tuple[N], np.dtype[np.float64]],
        /,
        *,
        x_train=None,
        y_train=None,
        x_test=None,
        es: Optional[Cma] = None,
        weights=None,
    ):
        self.__shift = 0

    def transform(
        self, data: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]:
        self.__shift = data.min()
        return np.log(data - self.__shift + self.min_offset)

    def transform_inv(
        self, data: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]:
        return np.e**data - self.min_offset + self.__shift

    def transform_inv_std(
        self, std: np.ndarray[S, dtype[float64]], /
    ) -> np.ndarray[S, dtype[float64]]:
        return np.e**std


class Surrogate:
    x_transf: Type[Transformation] = ShiftAndScaleByEs
    y_transf: Type[Transformation] = MinAdjustedLog

    def __init__(
        self,
        *,
        model: Optional[Model | ModelFactory] = None,
        subset: Optional[Subset] = None,
        x_transf: Optional[Type[Transformation]] = None,
        y_transf: Optional[Type[Transformation]] = None,
    ):
        self.__model = model if model is not None else Raf()
        self.__subset = (
            subset
            if subset is not None
            else ClosestToEachTestPoint(n_max_coef=20, norm=Mahalanobis)
        )
        self.__x_transf = x_transf if x_transf is not None else self.x_transf
        self.__y_transf = y_transf if y_transf is not None else self.y_transf

    def __repr__(self) -> str:
        return repr_default(
            self,
            model=self.__model,
            subset=self.__subset,
            x_tf=self.__x_transf,
            y_tf=self.__y_transf,
        )

    def __call__(
        self,
        *,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        x_test: NDArray[np.float64],
        es: Cma,
    ) -> Model:
        subset_idx = self.__subset(
            x_train=x_train, y_train=y_train, x_test=x_test, es=es
        )
        subset_idx = np.sort(subset_idx)  # Important for some ECs
        if __debug__:
            print(f"subset-idx-size={subset_idx.size}")
        x_subset, y_subset = x_train[subset_idx], y_train[subset_idx]

        weights = None
        # # Hansen's 20..1
        # weights = np.linspace(1, 20, y_subset.shape[0])[:, None]
        # # Experimental weights inversely proportional to the Mahalanobis norm
        # norms = np.array([es.mahalanobis_norm(x) for x in x_subset - es.mean])
        # weights = np.e ** (-norms*2)

        x_transf = self.__x_transf(
            x_subset,
            x_train=x_subset,
            y_train=y_subset,
            x_test=x_test,
            es=es,
            weights=weights,
        )
        y_transf = self.__y_transf(
            y_subset,
            x_train=x_subset,
            y_train=y_subset,
            x_test=x_test,
            es=es,
            weights=weights,
        )

        x_train_transf = x_transf.transform(x_subset)
        y_train_transf = y_transf.transform(y_subset)
        x_test_transf = x_transf.transform(x_test)

        model = (
            self.__model(
                x_train=x_train_transf,
                y_train=y_train_transf,
                x_test=x_test_transf,
                weights=weights,
            )
            if isinstance(self.__model, ModelFactory)
            else self.__model
        )

        @wraps(model)
        def surrogate_model(
            features: np.ndarray[tuple[N, DIM], np.dtype[np.float64]],
        ) -> Prediction[N]:
            features_transf = x_transf.transform(features)
            pred_transf = model(features_transf)
            return Prediction(
                y_transf.transform_inv(pred_transf.mean),
                y_transf.transform_inv_std(pred_transf.std),
            )

        return surrogate_model


class ValuesAndEvaluatedIdx(NamedTuple):
    values: YPop
    evaluated_idx: np.ndarray[tuple[POPSIZE], np.dtype[np.uint]]


@runtime_checkable
class EvolutionControl(Protocol):
    def __call__(
        self, *, model: Model, points: XPop, problem: Problem, archive: Archive
    ) -> ValuesAndEvaluatedIdx: ...


def evaluate_all(
    *,
    model: Model,
    points: NDArray[np.float64],
    problem: Problem,
    archive: Archive,
    debug_tau=False,
    value_after_target_hit=np.float64(np.nan),
) -> ValuesAndEvaluatedIdx:
    """
    >>> isinstance(evaluate_all, EvolutionControl)
    True
    """
    evals_left = problem.evals_left
    evaluated_idx = np.arange(points.shape[0]).astype(np.uint)[:evals_left]
    values = np.array(
        [
            problem(p) if not problem.final_target_hit else value_after_target_hit
            for p in points[:evals_left]
        ]
        if value_after_target_hit is not None
        else [problem(p) for p in points[:evals_left]]
    )[:, None]
    if __debug__ and debug_tau:
        _tau_pop = kendalltau(values, model(points[:evals_left]).mean).statistic
        print(f"tau-population={_tau_pop:>5.2f}, final-target-hit={problem.final_target_hit}")
    return ValuesAndEvaluatedIdx(values=values, evaluated_idx=evaluated_idx)


class AcquisitionFunction(Protocol):
    def __call__(
        self, pred: Prediction[N]
    ) -> np.ndarray[tuple[N], np.dtype[np.float64]]: ...


def mean_criterion(pred: Prediction[N]) -> np.ndarray[tuple[N], np.dtype[np.float64]]:
    return pred.mean.squeeze()


def pi_criterion(pred: Prediction[N]) -> np.ndarray[tuple[N], np.dtype[np.float64]]:
    raise NotImplementedError


def ranking_difference_error():
    """<https://github.com/bajeluk/surrogate-cmaes/blob/fe33fda66e11c6949fe857289184007788c34794/src/util/errRankMu.m#L8-L93>"""


class EvaluateBestPointsByCriterion(EvolutionControl):
    def __init__(
        self,
        *,
        eval_ratio: float = float(np.finfo(np.float64).eps),
        criterion: AcquisitionFunction,
        offset_non_evaluated: bool = False,
    ):
        self.__criterion = criterion
        self.__eval_ratio = eval_ratio
        self.__offset_non_evaluated = offset_non_evaluated

    def __repr__(self) -> str:
        return repr_default(
            self,
            self.__criterion,
            eval_ratio=self.__eval_ratio,
            offset=self.__offset_non_evaluated,
        )

    def __call__(
        self, *, model: Model, points: XPop, problem: Problem, archive: Archive
    ) -> ValuesAndEvaluatedIdx:
        """
        Order of operations matter (grouped left to right):
        >>> 1e100 - 1e100 + 1e-15
        1e-15
        >>> 1e-15 + 1e100 - 1e100
        0.0
        >>> 1e100 + 1e-15 - 1e100
        0.0

        Next bigger float (using np.spacing instead of np.finfo().eps):
        >>> np.float64(n:=0) < (np.float64(n) + np.spacing(np.float64(n)))
        True
        >>> np.float64(n:=0) < (np.float64(n) + np.spacing(np.float64(n))/2)
        False
        """
        pred = model(points)
        eval_order_idx = (
            self.__criterion(pred).argsort(axis=0).astype(np.uint).squeeze()
        )
        n_eval = math.ceil(self.__eval_ratio * points.shape[0])
        try:
            n_eval = min(n_eval, problem.evals_left)  # type: ignore[assignment, type-var]
        except:  # noqa: E722
            pass
        pred_mean = pred.mean
        evaluated = eval_order_idx[:n_eval]
        for i in evaluated:
            pred_mean[i] = (
                problem(points[i])
                if not problem.final_target_hit
                else np.float64(np.nan)
            )
        # `Smart offset <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L234-L243>`_
        if (
            self.__offset_non_evaluated
            and (not_evaluated := eval_order_idx[n_eval:]).size > 0
            and evaluated.size > 0
        ):
            eval_min = np.nanmin(pred_mean[evaluated])
            pred_mean[not_evaluated] = (
                pred_mean[not_evaluated]
                - pred_mean[not_evaluated].min()
                + eval_min
                + np.spacing(eval_min)
            )
        return ValuesAndEvaluatedIdx(pred_mean, evaluated)


class EvaluateUntilKendallThreshold(EvolutionControl):
    """Inspired by pycma's `fitness_models <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L258-L323>`_"""

    def __init__(
        self,
        *,
        criterion: AcquisitionFunction,
        tau_thold: float = 0.85,
        offset_non_evaluated: bool = False,
        verbose=True,
        debug=False,
    ):
        self.__criterion = criterion
        self.__tau_thold = tau_thold
        self.__offset_non_evaluated = offset_non_evaluated
        self.__verbose = verbose
        self.__debug = debug
        # <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L83>
        self.n_for_tau = lambda popsi, nevaluated: int(
            max(15, min(1.2 * nevaluated, 0.75 * popsi))
        )
        # <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L87>
        self.min_evals_percent = 2
        # <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L370-L372>
        self.truncation_ratio = max(
            (3 / 4, (3 - (_max_relative_size_end := 2)) / 2)
        )  # use only truncation_ratio best in _n_for_model_building

    def __repr__(self) -> str:
        return repr_default(
            self,
            self.__criterion,
            tau_threshold=self.__tau_thold,
            offset=self.__offset_non_evaluated,
        )

    def __call__(
        self, *, model: Model, points: XPop, problem: Problem, archive: Archive
    ) -> ValuesAndEvaluatedIdx:
        archive_size = archive.y.shape[0]
        dim = problem.dimension
        n_points = points.shape[0]
        max_model_size = max(n_points, dim * (dim + 3) + 2)
        model_size = min(archive_size, max_model_size)
        pred = model(points)

        # TODO: evaluate the max size, and take the subsets later
        # <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L296-L297>
        n_evaluated = int(
            1
            + max(
                n_points * self.min_evals_percent / 100,
                3 / self.truncation_ratio - model_size,
            )
        )
        try:
            n_evaluated = min(n_evaluated, problem.evals_left)  # type: ignore[assignment, type-var]
        except:  # noqa: E722
            pass
        if __debug__:
            if (
                _n_archive := min(self.n_for_tau(n_points, n_evaluated), model_size)
                - n_evaluated
            ) > 0:
                print(
                    "tau-archive={}".format(
                        kendalltau(
                            archive.y[-_n_archive:], model(archive.x[-_n_archive:]).mean
                        ).statistic
                    )
                )
                # TODO test Kendall weighted

        eval_order_idx = (
            self.__criterion(pred).argsort(axis=0).astype(np.uint).squeeze()
        )

        tau = np.nan
        n_kendall = None
        (values := np.empty((n_points, 1))).fill(np.nan)
        evaluated = ~(not_evaluated := np.isnan(values.squeeze()))
        while not_evaluated.any():
            for i in (idx := eval_order_idx[:n_evaluated])[not_evaluated[idx]]:
                values[i] = (
                    problem(points[i])
                    if not problem.final_target_hit
                    else np.float64(np.nan)
                )
            evaluated = ~(not_evaluated := np.isnan(values.squeeze()))
            model_size = min(archive_size + n_evaluated, max_model_size)
            n_kendall = min(self.n_for_tau(n_points, n_evaluated), model_size)
            n_kendall_archive = max(n_kendall - n_evaluated, 0)
            # <https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/fitness_models.py#L780-L794>
            tau = (
                kendalltau(
                    np.concatenate([archive.y[-n_kendall_archive:], values[evaluated]]),
                    model(
                        np.concatenate(
                            [archive.x[-n_kendall_archive:], points[evaluated]]
                        )
                    ).mean,
                ).statistic
                if n_kendall >= 2
                else np.nan
            )  # noqa: F841
            if (
                tau >= self.__tau_thold
                or problem.all_evals_used
                or problem.final_target_hit
            ):
                pred_mean_not_evaluated = pred.mean[not_evaluated]
                if self.__offset_non_evaluated and pred_mean_not_evaluated.size > 0:
                    eval_min = np.nanmin(values[evaluated])
                    values[not_evaluated] = (
                        pred_mean_not_evaluated
                        - pred_mean_not_evaluated.min()
                        + eval_min
                        + np.spacing(eval_min)
                    )
                else:
                    values[not_evaluated] = pred_mean_not_evaluated
                break
            eval_next = math.ceil(n_evaluated / 2)
            try:
                eval_next = min(eval_next, problem.evals_left)  # type: ignore[assignment, type-var]
            except:  # noqa: E722
                pass
            n_evaluated += eval_next

        # Uncomment for diagnostics, comment out for performance
        std_of_means = pred.mean.std()
        mean_of_stds = pred.std.mean()
        if self.__verbose:
            print(
                " | ".join(
                    [
                        f"eval={evaluated.sum()}",
                        f"not-eval={not_evaluated.sum()}",
                        f"{tau=:>5.2f}",
                        f"{n_kendall=:>2}",
                        f"std(means)/mean(stds)={std_of_means/mean_of_stds:>6.2f}",
                        f"std(means)={std_of_means:.2e}",
                        f"mean(stds)={mean_of_stds:.2e}",
                    ]
                )
            )
        if __debug__ and self.__debug:
            if hasattr(problem, "delta_to_optimum"):
                print(f"delta_f={problem.delta_to_optimum:.1e}")
            try:
                _values_true = np.array([PROBLEM_DEBUG(p) for p in points])
                _tau_pop = kendalltau(_values_true, model(points).mean).statistic
                _tau_pop_off = kendalltau(_values_true, values).statistic
                print(f"tau-population={_tau_pop:>5.2f} | tau-pop-offset={_tau_pop_off:>5.2f} | final-target-hit={problem.final_target_hit}")
                # if _tau_pop < 0.25:
                #     breakpoint()
                # _preds = model(points)
            except:  # noqa: E722
                pass
        return ValuesAndEvaluatedIdx(values, np.flatnonzero(evaluated).astype(np.uint))


@runtime_checkable
class NMinCallable(Protocol):
    def __call__(self, *, dim: int, **kwargs) -> int:
        """
        >>> isinstance(lambda **kwargs: 1, NMinCallable)
        True

        >>> isinstance(lambda dim: dim, NMinCallable)
        True
        """
        ...


def get_dim(*, dim: int, **kwargs) -> int:
    """
    >>> isinstance(get_dim, NMinCallable)
    True
    """
    return dim


class SurrogateAndEc(NamedTuple):
    surrogate: SurrogateCallable
    evolution_control: EvolutionControl
    get_n_min: NMinCallable = get_dim

    def __repr__(self) -> str:
        return repr_default(
            None, self.surrogate, self.evolution_control, n_min_fn=self.get_n_min
        )


def seek_minimum(
    problem: Problem,
    *,
    es: Cma,
    surrogate_and_ec: Optional[SurrogateAndEc] = None,
    sort_archive=True,
    log: Optional[Callable] = None,  # TODO: Log Protocol
    verbose=False,
) -> Archive:
    t_0 = time.perf_counter()
    if verbose:
        print('__'.join([str(i) for i in (problem, es, surr_and_ec)]))

    if surrogate_and_ec is not None:
        surrogate, evolution_control, get_n_min = surrogate_and_ec
        n_min = get_n_min(dim=problem.dimension)

        def get_values_and_eval_idx(
            problem, es, points, archive
        ) -> ValuesAndEvaluatedIdx:
            return (
                evolution_control(
                    model=surrogate(
                        x_train=archive.x, y_train=archive.y, x_test=points, es=es
                    ),
                    points=points,
                    problem=problem,
                    archive=archive,
                )
                if archive.x.size > n_min
                else evaluate_all(
                    model=predict_zeros_and_ones,
                    points=points,
                    problem=problem,
                    archive=archive,
                )
            )
    else:

        def get_values_and_eval_idx(
            problem, es, points, archive
        ) -> ValuesAndEvaluatedIdx:
            return evaluate_all(
                model=predict_zeros_and_ones,
                points=points,
                problem=problem,
                archive=archive,
            )

    archive = Archive(np.empty((0, problem.dimension)), np.empty((0, 1)))

    end = problem.final_target_hit or problem.all_evals_used
    while not end:
        points = es.ask()

        values, eval_idx = get_values_and_eval_idx(problem, es, points, archive)

        if sort_archive:
            eval_idx = eval_idx[values.squeeze()[eval_idx].argsort()[::-1]]

        archive = Archive(
            np.concatenate((archive.x, points[eval_idx])),
            np.concatenate((archive.y, values[eval_idx])),
        )
        if end := problem.final_target_hit or problem.all_evals_used:
            break
        es.tell(points, values)

    t = time.perf_counter() - t_0
    if log is not None:
        log(
            problem=problem,
            es=es,
            surrogate_and_ec=surrogate_and_ec,
            secs=t,
        )

    return archive


def report(*, problem, es, surrogate_and_ec, secs, seed=None) -> str:
    return " ".join(
        filter(
            None,
            (
                f"{problem.dimension:>2}D",
                f"{problem.function_id:>2}-fun",
                f"{problem.instance:>3}-inst",
                f"{problem.budget:>4}-budget",
                f"{str(problem.final_target_hit):>5}-solved",
                f"{problem.evaluations:>3}-evals",
                f"{es.evals:>4}-cma-evals",
                f"{secs:>4.0f}s",
                f"{es.restarts}-restarts",
                f"{str(es.pop_size_initial):>2}-init-pop",
                f"{surrogate_and_ec}" if surrogate_and_ec is not None else None,
                f"{problem.delta_to_optimum:.1e}-delta-f"
                if hasattr(problem, "delta_to_optimum")
                else None,
                f"{seed:>10}-seed" if seed is not None else None,
            ),
        )
    )


def set_seed(seed: int = 42, verbose=False):
    """<https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy>"""
    np.random.seed(seed)
    random.seed(seed)
    with suppress(NameError):
        torch.manual_seed(seed)  # type: ignore[name-defined] # noqa: F821
        torch.cuda.manual_seed(seed)  # type: ignore[name-defined] # noqa: F821
        torch.backends.cudnn.deterministic = True  # type: ignore[name-defined] # noqa: F821
        torch.backends.cudnn.benchmark = False  # type: ignore[name-defined] # noqa: F821
    # tf.random.set_seed(seed) # module 'tensorflow._api.v2.compat.v1.random' has no attribute 'set_seed'
    tensorflow.random.set_seed(seed)
    tf.random.set_random_seed(seed)
    tensorflow.experimental.numpy.random.seed(
        seed
    )  # module 'tensorflow._api.v2.compat.v1.experimental' has no attribute 'numpy'
    tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    # os.environ['TF_DETERMINISTIC_OPS'] = '1' # Random ops require a seed to be set when determinism is enabled. Please set a seed before running the op, e.g. by calling tf.random.set_seed(1).
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if verbose:
        print(f"Random seed set as {seed}")


def get_seed_np() -> np.uint32:
    _, keys, *_ = np.random.get_state()
    return keys[0]  # type: ignore[return-value] # pyright: ignore [reportReturnType]

@dataclass
class Arguments:
    dim: int
    fun: int
    inst: list[int]
    crit: AcquisitionFunction

def parse_args(verbose=False) -> Arguments:
    parser = argparse.ArgumentParser(description='CMAES with NN Surrogate Experiment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dim', type=int, choices=[2, 3, 5, 10, 20, 40], required=True,
                        help='Input dimensionality')
    parser.add_argument('--fun', type=int, choices=list(range(1, 24 + 1)), required=True,
                        help='Function id')
    parser.add_argument('--inst', nargs='+', type=int, choices=list(range(1, 15 + 1)),
                        help='Instance suite numbers, all tested if not given')
    criteria = {
        c.__name__.removesuffix('_criterion'): c
        for c in [mean_criterion, pi_criterion]
    }
    parser.add_argument('--crit', choices=list(criteria.keys()), default='mean',
                        help='Criterion for Evolution Control')
    args = parser.parse_args()
    if verbose:
        print(f"{args = }")
    return Arguments(dim=args.dim, fun=args.fun, inst=args.inst, crit=criteria[args.crit])

def test(
    *,
    dim=None,
    fun=None,
    inst=None,
    budget_coef=None,
    seed=None,
    pickle_prefix=None,
    offset=True,
    problem_class,
    es_class,
    surr_and_ec,
    ):
    dim = dim if dim is not None else 2
    fun = fun if fun is not None else 1
    inst = inst if inst is not None else 1
    budget_coef = budget_coef if budget_coef is not None else 250
    seed = seed if seed is not None else 1


    warnings.simplefilter("error")
    np.seterr("raise")
    set_seed(seed)
    tf.compat.v1.disable_eager_execution()

    set_seed(seed)
    problem = problem_class.get(
        dim=dim, fun=fun, inst=inst, budget_coef=budget_coef
    )
    if __debug__:
        global PROBLEM_DEBUG
        PROBLEM_DEBUG = problem_class.get(dim=dim, fun=fun, inst=inst)
    es = es_class(x0=np.zeros(problem.dimension), seed=seed)

    name = '__'.join([str(i) for i in (problem, es, surr_and_ec)])
    print('===', name)

    x, y = seek_minimum(
        problem,
        es=es,
        surrogate_and_ec=surr_and_ec,
        log=lambda **kwargs: print(report(seed=seed, **kwargs)),
    )
    if pickle_prefix is not None:
        with open(f"{pickle_prefix}_x-y.pickle", "wb") as f:
            pickle.dump((x, y), f, protocol=pickle.HIGHEST_PROTOCOL)

def test_all(**test_kwargs):
    problems = [ProblemCocoex, ProblemIoh]

    es_list = [Pycma, Modcma]

    models = [predict_zeros_and_ones, predict_random, Raf(debug=True), Eaf()][0:1]
    subsets = [
        ClosestToEachTestPoint(norm=Mahalanobis),
        ClosestToEachTestPoint(n_max_coef=6, norm=Mahalanobis),
        LastN(),
        InNormRange(norm=Mahalanobis),
    ]
    surrogates = [
        Surrogate(model=model, subset=subset) for model, subset in product(models, subsets)
    ]

    criteria = [mean_criterion]
    offsets = [True, False]
    ec_list = [
        partial(evaluate_all, debug_tau=True),
        *[
            EvaluateBestPointsByCriterion(
                criterion=mean_criterion, eval_ratio=0.1, offset_non_evaluated=offset
            ) for criterion, offset in product(criteria, offsets)
        ],
        *[
            EvaluateUntilKendallThreshold(
                criterion=criterion,
                offset_non_evaluated=offset,
                debug=True,
            ) for criterion, offset in product(criteria, offsets)
        ],
    ]

    surr_ec_list = [
        None,
        *[
            SurrogateAndEc(surrogate=surrogate, evolution_control=ec, get_n_min=get_dim)
            for surrogate, ec in product(surrogates, ec_list)
        ],
    ]

    for problem_class, es_class, surr_and_ec in product(
        problems, es_list, surr_ec_list
    ):
        print('__'.join([str(i) for i in (problem_class, es_class, surr_and_ec)]))
        test(problem_class=problem_class, es_class=es_class, surr_and_ec=surr_and_ec, **test_kwargs)


if __name__ == "__main__":
    try:
        should_test_all = bool(os.environ['TESTALL'])
    except:
        should_test_all = False

    args = parse_args(verbose=True)

    if should_test_all:
        test_all(dim=args.dim, fun=args.fun, inst=args.inst)
    else:
        surr_and_ec = SurrogateAndEc(
            surrogate=Surrogate(
                model=Raf(debug=True),
                subset=ClosestToEachTestPoint(norm=Mahalanobis),
            ), evolution_control=EvaluateUntilKendallThreshold(
                criterion=args.crit,
                offset_non_evaluated=True,
                debug=True,
            ),
        )
        test(
            dim=args.dim,
            fun=args.fun,
            inst=args.inst,
            problem_class=ProblemIoh,
            es_class=Pycma,
            surr_and_ec=surr_and_ec,
        )
