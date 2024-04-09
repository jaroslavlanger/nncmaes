from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
from contextlib import suppress
from numbers import Real
import os
import random
import sys
import time
import types
from typing import Callable, Optional, Protocol, Type

import numpy as np
import numpy.typing as npt

with suppress(ImportError):
    import cocoex  # type: ignore
import ioh  # type: ignore
from cma import CMAEvolutionStrategy  # type: ignore
from modcma import AskTellCMAES  # type: ignore


def repr_default(self, *attributes) -> str:
    return f"{self.__class__.__name__}({', '.join([a.__name__ if isinstance(a, (types.FunctionType, type)) else str(a) for a in attributes])})"


class Problem(ABC):
    @staticmethod
    @abstractmethod
    def get(*, dim: int, fun: int, inst: int, budget_coef: int) -> Problem: ...

    @abstractmethod
    def __call__(self, x) -> np.float64: ...

    @abstractmethod
    def __repr__(self) -> str: ...

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
    def bounds_lower(self) -> npt.NDArray[np.float64]: ...

    @property
    @abstractmethod
    def bounds_upper(self) -> npt.NDArray[np.float64]: ...

    @property
    @abstractmethod
    def current_best_y(self) -> float: ...

    @property
    @abstractmethod
    def evaluations(self) -> int: ...

    @property
    @abstractmethod
    def all_evals_used(self) -> bool: ...

    @property
    @abstractmethod
    def final_target_hit(self) -> bool:
        """<https://github.com/numbbo/coco/blob/master/code-experiments/src/coco_problem.c#L443-L444>"""

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
        return self.__problem(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.__problem)})"

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
    def bounds_lower(self) -> npt.NDArray[np.float64]:
        return self.__problem.lower_bounds

    @property
    def bounds_upper(self) -> npt.NDArray[np.float64]:
        return self.__problem.upper_bounds

    @property
    def current_best_y(self) -> float:
        return self.__problem.best_observed_fvalue1

    @property
    def evaluations(self) -> int:
        return self.__problem.evaluations

    @property
    def all_evals_used(self) -> bool:
        return self.evaluations >= self.__budget if self.__budget is not None else False

    @property
    def final_target_hit(self) -> bool:
        return bool(self.__problem.final_target_hit)


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
        return np.float64(self.__problem(x))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.__problem)})"

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
    def bounds_lower(self) -> npt.NDArray[np.float64]:
        return self.__problem.bounds.lb

    @property
    def bounds_upper(self) -> npt.NDArray[np.float64]:
        return self.__problem.bounds.ub

    @property
    def current_best_y(self) -> float:
        return self.__problem.state.current_best.y

    @property
    def evaluations(self) -> int:
        return self.__problem.state.evaluations

    @property
    def all_evals_used(self) -> bool:
        return self.evaluations >= self.__budget if self.__budget is not None else False

    @property
    def final_target_hit(self) -> bool:
        return self.__problem.optimum.y + 1e-8 >= self.__problem.state.current_best.y

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
    def __init__(self, *, x0: Iterable[Real], lb, ub, lambda_=None, verbose=None): ...

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
    def mean(self) -> npt.NDArray[np.float64]: ...

    @property
    @abstractmethod
    def std(self) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def mahalanobis_norm(self, delta) -> np.float64: ...

    @abstractmethod
    def ask(self) -> npt.NDArray[np.float64]: ...

    @abstractmethod
    def tell(
        self,
        points: Iterable[npt.NDArray[np.float64]],
        values: Iterable[npt.NDArray[np.float64]],
    ): ...


class WrappedCma(Cma):
    def __init__(
        self,
        *,
        x0,
        lb=None,
        ub=None,
        lambda_=None,
        verbose=None,
        seed=None,
    ):
        """
        >>> WrappedCma(x0=np.zeros(2), verbose=-9)
        WrappedCma({'evals': 0, 'mean': array([0., 0.]), 'pop_size_initial': 6, 'restarts': 0, 'std': array([2.     , 2.00005])})

        `Details how CMAEvolutionStrategy uses the seed. <https://github.com/CMA-ES/pycma/blob/development/cma/evolution_strategy.py#L477>`_
        """
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
        return self.__es.countevals

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        return self.__es.mean  # pyright: ignore [reportReturnType]

    @property
    def std(self) -> npt.NDArray[np.float64]:
        """<https://github.com/CMA-ES/pycma/blob/r3.3.0/cma/evolution_strategy.py#L3160-L3169>"""
        return self.__es.stds

    def mahalanobis_norm(self, delta) -> np.float64:
        return self.__es.mahalanobis_norm(delta)

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

    def ask(self) -> npt.NDArray[np.float64]:
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

    def tell(
        self,
        points: Iterable[npt.NDArray[np.float64]],
        values: Iterable[npt.NDArray[np.float64]],
    ):
        self.__es.tell(list(points), [v.item() for v in values])


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

    def ask(self) -> npt.NDArray[np.float64]:
        return np.array(
            [self.__es.ask().squeeze() for _ in range(self.__es.parameters.lambda_)]
        )

    def tell(
        self,
        points: Iterable[npt.NDArray[np.float64]],
        values: Iterable[npt.NDArray[np.float64]],
    ):
        for p, v in zip(points, values):
            self.__es.tell(p[:, None], v)


class Norm(ABC):
    @abstractmethod
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, vector: npt.NDArray[np.float64]) -> np.float64: ...


class Mahalanobis(Norm):
    def __init__(self, *, es, **ignored):
        self.__es = es

    def __call__(self, vector: npt.NDArray[np.float64]) -> np.float64:
        return self.__es.mahalanobis_norm(vector)


class Subset(ABC):
    @abstractmethod
    def __call__(
        self,
        *,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        x_test: npt.NDArray[np.float64],
        es: Cma,
    ) -> npt.NDArray[np.uint]: ...

    @abstractmethod
    def __repr__(self) -> str: ...


class ClosestToAnyTestPoint(Subset):
    """TODO .min(axis=2).argpartition(n_max-1).astype(np.uint)[:n_max]"""


class ClosestToEachTestPoint(Subset):
    """`Matlab inspiration <https://github.com/bajeluk/surrogate-cmaes/blob/master/src/data/Archive.m#L128-L202>`_
    >>> ClosestToEachTestPoint(n_max_coef=2)(y_train=None, es=None,\
x_train=np.array([[-3, -3], [-3,  3], [ 0,  0], [ 3, -3], [ 3,  3]]),\
x_test=np.array([\
[-2, -2], [-2,  2],\
[ 2, -2], [ 2,  2]]))
    array([0, 1, 3, 4], dtype=uint64)

    >>> ClosestToEachTestPoint(n_max_coef=2)(y_train=None, es=None,\
x_train=np.array([\
[-4, -4], [-4, -3], [-3, -4],\
[-4,  4], [-4,  3], [-3,  4],\
[-1, -1], [-1,  1], [ 0,  0], [ 1, -1], [ 1,  1],\
[ 3, -4], [ 4, -3], [ 4, -4],\
[ 3,  4], [ 4,  3], [ 4,  4],\
]),\
x_test=np.array([\
[-3, -3], [-3,  3],\
[ 3, -3], [ 3,  3],\
]))
    array([ 1,  4, 11, 14], dtype=uint64)
    """

    def __init__(
        self,
        *,
        norm: Callable[[np.ndarray], np.float64] | Type[Norm] = np.linalg.norm,
        n_max_coef=20,
    ):
        self.__norm = norm
        self.__n_max_coef = n_max_coef

    def __repr__(self) -> str:
        return repr_default(self, self.__n_max_coef, self.__norm)

    def __call__(
        self,
        *,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        x_test: npt.NDArray[np.float64],
        es: Cma,
    ) -> npt.NDArray[np.uint]:
        n_max = self.__n_max_coef * x_train.shape[1]
        if n_max >= (n_total := x_train.shape[0]):
            return np.arange(n_total).astype(np.uint)
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
        unique, unique_idx = np.unique(
            norms_2d.argpartition(kth=np.arange(n_max), axis=0)
            .astype(np.uint)[:n_max]
            .flatten(),
            return_index=True,
        )
        return unique[unique_idx.argsort()][:n_max]


def get_mean_and_std(points, *, weights=None):
    if points.size > 0:
        if weights is not None:
            mean = np.average(points, weights=weights, axis=0)
            std = np.sqrt(
                np.atleast_2d(
                    np.cov(points, rowvar=False, ddof=0, aweights=weights.squeeze())
                ).diagonal()
            )
        else:
            mean = points.mean(axis=0)
            std = points.std(axis=0)
        std[std == 0] = 1  # when std==0, (-mean) makes any value (==0)
    else:
        mean = 0
        std = 1
    return mean, std


def shift_and_scale_x_by_es_y_by_train(*, x_train, y_train, x_test, es):
    return (es.mean, es.std), get_mean_and_std(y_train)


def predict_zeros(*, x_train, y_train, x_test, **ignored):
    zeros = np.zeros((x_test.shape[0], y_train.shape[-1]))
    return zeros, zeros


def evaluate_all(
    *, pred=None, pred_std=None, points, problem
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint]]:
    eval_idx = np.arange(points.shape[0], dtype=np.uint)
    values = np.array(
        [
            problem(p) if not problem.all_evals_used else np.float64(np.nan)
            for p in points
        ]
    )[:, None]
    return values, eval_idx


class SurrogateCallable(Protocol):
    def __call__(
        self,
        *,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        x_test: npt.NDArray[np.float64],
        es: Cma,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...


class Surrogate:
    def __init__(self, *, model, subset, shift_and_scale):
        self.__model = model
        self.__subset = subset
        self.__shift_and_scale = shift_and_scale

    def __repr__(self) -> str:
        return repr_default(self, self.__model, self.__subset, self.__shift_and_scale)

    def __call__(
        self,
        *,
        x_train: npt.NDArray[np.float64],
        y_train: npt.NDArray[np.float64],
        x_test: npt.NDArray[np.float64],
        es: Cma,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        subset_idx = self.__subset(
            x_train=x_train, y_train=y_train, x_test=x_test, es=es
        )
        x_subset, y_subset = x_train[subset_idx], y_train[subset_idx]

        (x_shift, x_scale), (y_shift, y_scale) = self.__shift_and_scale(
            x_train=x_subset, y_train=y_subset, x_test=x_test, es=es
        )

        x_train_scaled = (x_subset - x_shift) / x_scale
        y_train_scaled = (y_subset - y_shift) / y_scale
        x_test_scaled = (x_test - x_shift) / x_scale

        pred_mean_scaled, pred_std_scaled = self.__model(
            x_train=x_train_scaled, y_train=y_train_scaled, x_test=x_test_scaled
        )

        pred_mean, pred_std = (
            pred_mean_scaled * y_scale + y_shift,
            pred_std_scaled * y_scale,
        )
        return pred_mean, pred_std


def seek_minimum(
    problem: Problem,
    *,
    es: Cma,
    surrogate: Optional[SurrogateCallable] = None,
    n_min_coef: float = 1e-8,
    evolution_control=evaluate_all,
    seed=None,
    log: Optional[Callable] = None,
):
    t_0 = time.perf_counter()

    n_min = n_min_coef * problem.dimension
    x_archive = np.empty((0, problem.dimension))
    y_archive = np.empty((0, 1))

    while not (problem.final_target_hit or problem.all_evals_used):
        points = es.ask()

        if surrogate is not None and x_archive.size > n_min:
            pred, pred_std = surrogate(
                x_train=x_archive, y_train=y_archive, x_test=points, es=es
            )
            values, eval_idx = evolution_control(
                pred=pred, pred_std=pred_std, points=points, problem=problem
            )
        else:
            values, eval_idx = evaluate_all(points=points, problem=problem)

        x_archive = np.concatenate((x_archive, points[eval_idx]))
        y_archive = np.concatenate((y_archive, values[eval_idx]))
        if problem.all_evals_used:
            break
        es.tell(points, values)

    t = time.perf_counter() - t_0
    if log is not None:
        log(
            problem=problem,
            es=es,
            surrogate=surrogate,
            ec=evolution_control,
            n_min=n_min,
            secs=t,
            seed=seed,
        )

    return x_archive, y_archive


def report(*, problem, es, surrogate, ec, n_min, secs, seed) -> str:
    return " ".join(
        [
            f"{problem.dimension:>2}D",
            f"{problem.function_id:>2}-fun",
            f"{problem.instance:>2}-inst",
            f"{problem.budget:>4}-budget",
            f"{str(es.pop_size_initial):>2}-init-pop",
            f"{surrogate}",
            f"{problem.evaluations}-evals",
            f"{es.evals}-cma-evals",
            f"{secs:>4.0f}s",
            *([f"{problem.progress:.8%}"] if hasattr(problem, "progress") else []),
            f"{str(problem.final_target_hit):>5}-solved",
            f"{es.restarts}-restarts",
            f"{seed:>10}-seed",
        ]
    )


def set_seed(seed: int = 42, verbose=False) -> None:
    """<https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy>"""
    np.random.seed(seed)
    random.seed(seed)
    with suppress(NameError):
        torch.manual_seed(seed)  # type: ignore[name-defined] # noqa: F821
        torch.cuda.manual_seed(seed)  # type: ignore[name-defined] # noqa: F821
        torch.backends.cudnn.deterministic = True  # type: ignore[name-defined] # noqa: F821
        torch.backends.cudnn.benchmark = False  # type: ignore[name-defined] # noqa: F821
    # # tf.random.set_seed(seed) # module 'tensorflow._api.v2.compat.v1.random' has no attribute 'set_seed'
    # tensorflow.random.set_seed(seed)
    # tf.random.set_random_seed(seed)
    # tensorflow.experimental.numpy.random.seed(seed) # module 'tensorflow._api.v2.compat.v1.experimental' has no attribute 'numpy'
    # tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    if verbose:
        print(f"Random seed set as {seed}")


def get_seed_np() -> np.uint32:
    _, keys, *_ = np.random.get_state()
    return keys[0]  # type: ignore[return-value] # pyright: ignore [reportReturnType]


if __name__ == "__main__":
    for problem_class in [ProblemCocoex, ProblemIoh]:
        problem = problem_class.get(dim=2, fun=2, inst=1, budget_coef=250 * 10**0)
        set_seed()
        seed = get_seed_np()
        x, y = seek_minimum(
            problem,
            es=WrappedCma(x0=np.zeros(problem.dimension), seed=seed),
            surrogate=Surrogate(
                model=predict_zeros,
                subset=ClosestToEachTestPoint(n_max_coef=20, norm=Mahalanobis),
                shift_and_scale=shift_and_scale_x_by_es_y_by_train,
            ),
            n_min_coef=1,
            seed=seed,
            log=lambda **kwargs: print(report(**kwargs)),
        )
