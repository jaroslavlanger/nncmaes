# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# pip install -q --upgrade pip
# # pip install -q --upgrade jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# # pip install -q --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
# # pip install -q git+https://www.github.com/google/neural-tangents
# pip install ioh
# pip install modcma
# # pip install cocopp
# pip install cma
# pip install torchinfo
# -----------------------------------------------------------------------------
import copy
import itertools
from functools import partial
import time
import random
import os
import sys
from contextlib import suppress
import re

import numpy as np
import numpy.typing as npt
from scipy import stats
from scipy.special import ndtr

# matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cbook, cm
from matplotlib.colors import LightSource

import torch.nn as nn
import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
# from jax import random
from jax.example_libraries import optimizers
from jax import jit, grad, vmap

# from clu import metrics
# from flax.training import train_state  # Useful dataclass to keep train state
# from flax import struct                # Flax dataclasses
# import optax                           # Common loss functions and optimizers

# import neural_tangents as nt
# from neural_tangents import stax

import tensorflow
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from random import randint

from tqdm.auto import tqdm
from tqdm.auto import trange

with suppress(ImportError): import ioh
with suppress(ImportError): import cocoex
with suppress(ImportError): import cma
with suppress(ImportError): from modcma import AskTellCMAES
with suppress(ImportError): from torchinfo import summary
with suppress(ImportError): import matplotlib_inline
with suppress(ImportError): import seaborn as sns
with suppress(ImportError): import plotly.graph_objects as go
with suppress(ImportError): from IPython.display import clear_output

def get_problem_ioh(function, *, instance, dimension):
    """>>> get_problem_ioh(1, instance=1, dimension=2)"""
    return ioh.get_problem(function, instance=instance, dimension=dimension, problem_class=ioh.ProblemClass.BBOB)

def final_target_hit(problem):
    """https://github.com/numbbo/coco/blob/master/code-experiments/src/coco_problem.c#L443-L444"""
    if isinstance(problem, ioh.iohcpp.problem.BBOB):
        return problem.optimum.y + 1e-8 >= problem.state.current_best.y
    elif isinstance(problem, cocoex.interface.Problem):
        return problem.final_target_hit
    else:
        raise NotImplementedError

def get_dimension(problem):
    if isinstance(problem, ioh.iohcpp.problem.BBOB):
        return problem.meta_data.n_variables
    elif isinstance(problem, cocoex.interface.Problem):
        return problem.dimension
    else:
        raise NotImplementedError

def get_evaluations(problem):
    if isinstance(problem, ioh.iohcpp.problem.BBOB):
        return problem.state.evaluations
    elif isinstance(problem, cocoex.interface.Problem):
        return problem.evaluations
    else:
        raise NotImplementedError

def current_best_y(problem):
    if isinstance(problem, ioh.iohcpp.problem.BBOB):
        return problem.state.current_best.y
    elif isinstance(problem, cocoex.interface.Problem):
        return problem.best_observed_fvalue1
    else:
        raise NotImplementedError

def progress(problem):
    if isinstance(problem, ioh.iohcpp.problem.BBOB):
        optimum = problem.optimum.y
        curr_best = problem.state.current_best.y
    else:
        raise NotImplementedError
    if optimum >= 0:
        if curr_best == 0:
            return 1
        else:
            return optimum / curr_best
    else:
        return curr_best / optimum


class WrappedCma:
    restarts = 0

    def __init__(self, *, x0, lambda_=None):
        self.x0 = x0
        self.es = self.make_cmaes(x0=x0, lambda_=lambda_)
        self.lambda_ = self.es.popsize

    @staticmethod
    def make_cmaes(*, x0, lambda_):
        return cma.CMAEvolutionStrategy(x0, 2, {
            'popsize': lambda_,
            'verbose': -9,
            # 'maxfevals': budget,
            # 'seed': SEED,
        })

    def ask(self):
        if self.es.stop():
            self.lambda_ *= 2
            self.es = self.make_cmaes(x0=self.x0, lambda_=self.lambda_)
            self.restarts += 1
        return np.array(self.es.ask())

    def tell(self, points, values):
        self.es.tell(list(points), [v[0] for v in values])


class WrappedModcma:
    def __init__(self, *, dim, lambda_=None):
        self.es = AskTellCMAES(d=dim,
                        budget=sys.maxsize, # None and float('inf') does not work
                        # bound_correction='COTN',
                        bound_correction="saturate",
                        # lb=problem.bounds.lb,
                        # ub=problem.bounds.ub,
                        lambda_=lambda_,
                        active=True,
                        local_restart="IPOP",
                        )
    def ask(self):
        return np.array([self.es.ask().squeeze() for _ in range(self.es.parameters.lambda_)])

    def tell(self, points, values):
        for p, v in zip(points, values):
            self.es.tell(p[:, None], v)

    @property
    def restarts(self):
        return len(self.es.parameters.restarts) -1


def solve_cmaes_get_data(problem, *, pbar=False, lambda_=None):
    dim = get_dimension(problem)
    budget = int(2*1e5*dim)
    problem.reset()
    cma = make_cmaes(dim, lambda_=lambda_)

    xs: list[list[np.ndarray[(dim), np.float64]]] = []
    ys: list[list[float]] = []
    pop: list = []

    # TODO change point_asks to generations
    point_asks = range(1, budget+1)
    if pbar:
        point_asks = tqdm(point_asks, leave=False)
    for point_ask in point_asks:
        # Retrieve a single new candidate solution
        x: np.ndarray[(dim,1), np.float64] = cma.ask()
        pop.append(x)

        if len(pop) == cma.parameters.lambda_:
            xs.append([])
            ys.append([])

            for x in pop:
                x_squeezed: np.ndarray[(dim), np.float64] = x.squeeze()
                xs[-1].append(x_squeezed)
                if final_target_hit(problem):
                    ys[-1].append([np.nan])
                    continue
                # Evaluate the objective function
                y: float = problem(x_squeezed)
                # Update the algorithm with the objective function value
                cma.tell(x, y)
                ys[-1].append([y])
            pop = []
        if final_target_hit(problem):
            if pbar:
                point_asks.container.close()
            break

    idx = np.unique([len(x) for x in xs], return_index=True)[1]
    xs_res = [np.array(xs[start:end]) for start, end in itertools.zip_longest(idx, idx[1:])]
    ys_res = [np.array(ys[start:end]) for start, end in itertools.zip_longest(idx, idx[1:])]
    return xs_res, ys_res, cma

def get_train_test(gen_test, x, y):
    """Generation for gen_test is counted from 0"""
    # np.argmax(np.cumsum([len(x_) for x_ in x]) >= 76) # maybe handy
    assert gen_test >= 0, "currently the gen_test must be >= 0"
    gen_added = 0
    x_train, y_train = [], []
    x_test, y_test = np.array([]), np.array([])
    last_idx = None
    for x_, y_ in zip(x, y):
        x_dim = x_.shape[-1]
        y_dim = y_.shape[-1]

        gen_curr = x_.shape[0]
        if gen_added + gen_curr > gen_test:
            last_idx = gen_test - gen_added
            x_test = x_[last_idx].reshape(-1, x_dim)
            y_test = y_[last_idx].reshape(-1, y_dim)

        x_train.append(x_[:last_idx].reshape(-1, x_dim))
        y_train.append(y_[:last_idx].reshape(-1, y_dim))
        if last_idx is not None:
            break
        gen_added += gen_curr

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    return (x_train, y_train), (x_test, y_test)

def get_max_distance_idx(x_train, x_test, std_multiple=10):
    """TODO r^A_max, example with SEED=3, FUNCTION=2"""
    if len(x_test) < 2:
        raise ValueError(f"x_test must have at least 2 elements, had: {len(x_test)}")
    x_test_mean = x_test.mean(axis=0)
    x_test_std = x_test.std(axis=0)
    x_lower = x_test_mean - std_multiple*x_test_std
    x_upper = x_test_mean + std_multiple*x_test_std
    mask = ((x_lower < x_train) & (x_train < x_upper)).all(axis=1)
    return np.flatnonzero(mask)

def get_n_max_idx(x_train, x_test, *, n_max):
    n = x_train.shape[0]
    return np.arange(n) if n <= n_max else (
        np.linalg.norm(np.repeat(x_train[:, None, :], x_test.shape[0], axis=1) - x_test, axis=-1)
        .argsort(axis=0).argsort(axis=0) # ranks
        .sum(axis=-1)
        .argpartition(n_max-1)[:n_max]
    )

identity = lambda x: x

def make_scaler(points):
    if points.size > 0:
        mean = points.mean(axis=0)
        std = points.std(axis=0)
        std[std == 0] = 1 # when std==0, (-mean) makes any value (==0)
    else:
        mean = 0
        std = 1

    def scale(points, mean=mean, std=std):
        return (points - mean) / std

    def scale_back(points, mean=mean, std=std):
        return points*std + mean

    return scale, scale_back, (mean, std)

class NeuralNetwork(nn.Module):
    def __init__(self, activation, *, width=128, dropout_p=.0, dim):
        super(NeuralNetwork,self).__init__()
        self.width = width
        self.dropout_p=dropout_p
        self.activation = activation

        # self.n_0 = nn.BatchNorm1d(dim)
        self.l_1 = nn.Linear(dim,self.width)
        self.drop = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else None
        #self.l_2 = nn.Linear(self.width,self.width)
        self.l__1 = nn.Linear(self.width, 1)
        self.transforms = [
            # self.n_0,
            self.l_1,
        ] + (
            [self.drop] if self.dropout_p > 0 else []
        ) + [
            activation,
            self.l__1,
        ]

    def forward(self,x):
        for t in self.transforms:
            x = t(x)
        return x

    @property
    def name(self):
        return '-'.join([
            ''.join([str(tmp)[0] for tmp in self.transforms if isinstance(tmp, nn.Module)]),
            str(self.width),
            self.activation.__name__,
        ] + ([f"drop{self.dropout_p}"] if self.dropout_p > 0 else []))

def train_network(network, x, y, *, plot, epochs=1000, mse_stop=-np.inf, lr=0.001, pbar=True):
    best_loss = np.inf
    best_model = network.state_dict()

    if plot:
        (fig, ax, dh) = plot

    # lr = 3e-4
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    loss = np.inf
    losses = []

    x = torch.Tensor(x).to(DEVICE)
    y = torch.Tensor(y).to(DEVICE)

    iter = range(1, epochs+1)
    if pbar:
        iter = tqdm(iter, leave=False)
    for epoch in iter:
        network.train()
        optimizer.zero_grad()
        loss_fn(y, network(x)).backward()
        optimizer.step()

        network.eval()
        with torch.no_grad():
            loss = loss_fn(y, network(x)).item()
        if pbar:
            iter.set_postfix({'loss': loss, 'net': network.name})
        losses.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(network.state_dict())
        if loss < mse_stop:
            if pbar:
                iter.container.close()
            break

        if epoch%200 == 0 and plot:
            ax.clear()
            ax.semilogy(losses, ',')
            dh.update(fig)
    del iter # especially for tqdm

    network.load_state_dict(best_model)
    return losses

def train_and_predict_ensemble(x_train, y_train, x_test, *, dim, epochs, lr, mse_stop, width, plot=None, pbar=None):
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
    nets = {f.__name__: NeuralNetwork(f, width=width, dropout_p=.0, dim=dim).to(DEVICE) for f in afs}

    losses = {}
    for name, net in nets.items():
        losses[name] = train_network(net, x_train, y_train,
                                     epochs=epochs,
                                     mse_stop=mse_stop,
                                     lr=lr,
                                     plot=plot,
                                     pbar=pbar
                                     )
    x_test_tsr = torch.Tensor(x_test).to(DEVICE)
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
    return pred_mean, pred_std

class NN():
    """https://github.com/YanasGH/RAFs/blob/main/main_experiments/rafs.py#L15-L91"""
    def __init__(self, x_dim, y_dim, hidden_size, init_stddev_1_w, init_stddev_1_b,
                init_stddev_2_w, n, learning_rate, ens):

        # setting up as for a usual NN
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.hidden_size = hidden_size
        self.n = n
        self.learning_rate = learning_rate

        # set up NN
        self.inputs = tf.placeholder(tf.float64, [None, x_dim], name='inputs')
        self.y_target = tf.placeholder(tf.float64, [None, y_dim], name='target')

        activation_fns = [tensorflow.keras.activations.selu, tf.nn.tanh, tensorflow.keras.activations.gelu, tensorflow.keras.activations.softsign, tf.math.erf, tf.nn.swish, tensorflow.keras.activations.linear]

        if ens <= len(activation_fns)-1:
            self.layer_1_w = tf.layers.Dense(hidden_size,
            activation = activation_fns[ens],
            kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_w),
            bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_b))
        else:
            af_ind = randint(0,3)
            self.layer_1_w = tf.layers.Dense(hidden_size,
            activation= activation_fns[af_ind],
            kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_w),
            bias_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_1_b))

        self.layer_1 = self.layer_1_w.apply(self.inputs)

        self.output_w = tf.layers.Dense(y_dim,
            activation=None, use_bias=False,
            kernel_initializer=tf.random_normal_initializer(mean=0.,stddev=init_stddev_2_w))

        self.output = self.output_w.apply(self.layer_1)

        # set up loss and optimiser - this is modified later with anchoring regularisation
        self.opt_method = tf.train.AdamOptimizer(self.learning_rate)
        self.mse_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] * tf.reduce_sum(tf.square(self.y_target - self.output))
        self.loss_ = 1/tf.shape(self.inputs, out_type=tf.int64)[0] * tf.reduce_sum(tf.square(self.y_target - self.output))
        self.optimizer = self.opt_method.minimize(self.loss_)
        return


    def get_weights(self, sess):
        '''method to return current params'''

        ops = [self.layer_1_w.kernel, self.layer_1_w.bias, self.output_w.kernel]
        w1, b1, w2 = sess.run(ops)

        return w1, b1, w2


    def anchor(self, sess, lambda_):   #lambda_anchor
        '''regularise around initial parameters'''

        w1, b1, w2 = self.get_weights(sess)

        # get initial params
        self.w1_init, self.b1_init, self.w2_init = w1, b1, w2

        loss = lambda_[0]*tf.reduce_sum(tf.square(self.w1_init - self.layer_1_w.kernel))
        loss += lambda_[1]*tf.reduce_sum(tf.square(self.b1_init - self.layer_1_w.bias))
        loss += lambda_[2]*tf.reduce_sum(tf.square(self.w2_init - self.output_w.kernel))

        # combine with original loss
        self.loss_ = self.loss_ + 1/tf.shape(self.inputs, out_type=tf.int64)[0] * loss
        self.optimizer = self.opt_method.minimize(self.loss_)
        return

    def predict(self, x, sess):
        '''predict method'''

        feed = {self.inputs: x}
        y_pred = sess.run(self.output, feed_dict=feed)
        return y_pred

def raf(X_train, y_train, X_val, *, log=False):
    """https://github.com/YanasGH/RAFs/blob/main/main_experiments/rafs.py#L94-L157"""
    # hyperparameters
    n = X_train.shape[0]
    x_dim = X_train.shape[1]
    y_dim = y_train.shape[1]

    n_ensembles = 5
    hidden_size = 100
    init_stddev_1_w =  np.sqrt(10)
    init_stddev_1_b = init_stddev_1_w # set these equal
    init_stddev_2_w = 1.0/np.sqrt(hidden_size) # normal scaling
    data_noise = 0 # 0.01 #estimated noise variance, feel free to experiment with different values # TODO
    lambda_anchor = data_noise/(np.array([init_stddev_1_w,init_stddev_1_b,init_stddev_2_w])**2)

    n_epochs = 1000
    learning_rate = 0.01

    NNs=[]
    y_prior=[]
    tf.reset_default_graph()
    sess = tf.Session()

    # loop to initialise all ensemble members, get priors
    for ens in range(0,n_ensembles):
        NNs.append(NN(x_dim, y_dim, hidden_size,
                    init_stddev_1_w, init_stddev_1_b, init_stddev_2_w, n, learning_rate, ens))

        # initialise only unitialized variables - stops overwriting ensembles already created
        global_vars = tf.global_variables()
        is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            sess.run(tf.variables_initializer(not_initialized_vars))

        # do regularisation now that we've created initialisations
        NNs[ens].anchor(sess, lambda_anchor)  #Do that if you want to minimize the anchored loss

        # save their priors
        y_prior.append(NNs[ens].predict(X_val, sess))

    for ens in range(0,n_ensembles):

        feed_b = {}
        feed_b[NNs[ens].inputs] = X_train
        feed_b[NNs[ens].y_target] = y_train
        if log: print('\nNN:',ens)

        ep_ = 0
        while ep_ < n_epochs:
            ep_ += 1
            blank = sess.run(NNs[ens].optimizer, feed_dict=feed_b)
            if ep_ % (n_epochs/5) == 0:
                loss_mse = sess.run(NNs[ens].mse_, feed_dict=feed_b)
                loss_anch = sess.run(NNs[ens].loss_, feed_dict=feed_b)
                if log: print('epoch:', ep_, ', mse_', np.round(loss_mse*1e3,3), ', loss_anch', np.round(loss_anch*1e3,3))
                # the anchored loss is minimized, but it's useful to keep an eye on mse too

    # run predictions
    y_pred=[]
    for ens in range(0,n_ensembles):
        y_pred.append(NNs[ens].predict(X_val, sess))

    """Display results:"""

    method_means = np.mean(np.array(y_pred)[:,:,:], axis=0)
    method_stds = np.sqrt(np.square(np.std(np.array(y_pred)[:,:,:],axis=0, ddof=1)) + data_noise)
    return method_means, method_stds

def pre_and_post_process_data(surrogate, *, n_max, subset=True, scale_x=True, scale_y=True):
    def surrogate_(x_train, y_train, x_test, *args, **kwargs):
        if subset:
            subset_n_max = get_n_max_idx(x_train, x_test, n_max=n_max)
            subset_dist = get_max_distance_idx(x_train, x_test)
            subset_idx = np.intersect1d(subset_n_max, subset_dist)
        else:
            subset_idx = slice(None)
        x_train_subset = x_train[subset_idx]
        y_train_subset = y_train[subset_idx]

        if scale_x:
            scale_x_, scale_x_back, scale_x_mean_std = make_scaler(x_train_subset)
        else:
            scale_x_, scale_x_back, scale_x_mean_std = identity, identity, (0,1)

        if scale_y:
            scale_y_, scale_y_back, (scale_y_mean, scale_y_std) = make_scaler(y_train_subset)
        else:
            scale_y_, scale_y_back, (scale_y_mean, scale_y_std) = identity, identity, (0,1)

        x_train_subset_scaled = scale_x_(x_train_subset)
        x_test_scaled = scale_x_(x_test)
        y_train_subset_scaled = scale_y_(y_train_subset)

        pred_mean, pred_std = surrogate(x_train_subset_scaled, y_train_subset_scaled, x_test_scaled, *args, **kwargs)

        return scale_y_back(pred_mean), pred_std*scale_y_std
    return surrogate_

def predict_zeros(x_train, y_train, x_test):
    zeros = np.zeros((x_test.shape[0], y_train.shape[-1]))
    return zeros, zeros

def UCB(mean, std, beta):
    """* <https://modal-python.readthedocs.io/en/latest/_modules/modAL/acquisition.html>
    * <https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf>"""
    return mean - beta*std

def PI(mean, std, max_val, tradeoff):
    """https://modal-python.readthedocs.io/en/latest/_modules/modAL/acquisition.html"""
    return ndtr((mean - max_val - tradeoff)/std)

def EI(mean, std, max_val, tradeoff):
    """https://modal-python.readthedocs.io/en/latest/_modules/modAL/acquisition.html"""
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff)*ndtr(z) + std*stats.norm.pdf(z)

def select_eval(mean, std, curr_best, *, acquisition, eval_ratio=0.8):
    n = np.ceil(mean.shape[0]*eval_ratio).astype(int)
    eval_mask = mean.squeeze() < curr_best
    if eval_mask.sum() < n:
        crit = acquisition(mean, std)
        eval_idx = crit.squeeze().argpartition(n-1, axis=0)[:n]
        # eval_mask := np.zeros(pred_mean_eaf_.shape[0], dtype=bool)
        eval_mask[eval_idx] = True
    return eval_mask, ~eval_mask

def format_criterion(crit):
    s = str(crit)
    if (m := re.fullmatch(r'<function (\w+) at \w+>', s)) is not None:
        return m.groups()[0]
    elif (m := re.match(r'.*acquisition=.*\(<function (\w+).*>, (.*)\), (.*)\)', s)) is not None:
        return '-'.join(m.groups()).replace('=', '-').replace('_', '-')
    else:
        return s

def evaluate_all_criterion(mean, std=None, curr_best=None):
    n = mean.shape[0]
    return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

def seek_minimum(es, *, problem, surrogate, criterion, budget):
    dim = get_dimension(problem)
    x_archive = np.empty((0,dim))
    y_archive = np.empty((0,1))

    def is_end(problem, budget):
        return final_target_hit(problem) or get_evaluations(problem) >= budget

    while not is_end(problem, budget):
        points = es.ask()

        if y_archive.size > 0:
            pred_mean, pred_std = surrogate(x_archive, y_archive, points)
            eval_mask, surr_mask = criterion(pred_mean, pred_std, current_best_y(problem))
        else:
            pred_mean, pred_std = predict_zeros(x_archive, y_archive, points)
            eval_mask, surr_mask = evaluate_all_criterion(points)
        assert np.logical_xor(eval_mask, surr_mask).all(), f"criterion outputs must be exclusive {eval_mask = }, {surr_mask}"

        points_eval = points[eval_mask]
        values_eval = np.array([problem(p) for p in points_eval if not is_end(problem, budget)])[:, None]
        x_archive = np.concatenate((x_archive, points_eval))
        y_archive = np.concatenate((y_archive, values_eval))
        if is_end(problem, budget):
            break
        pred_mean[eval_mask] = values_eval
        # es.tell(points_eval, values_eval)
        es.tell(points, pred_mean)
    return x_archive, y_archive

def cmaes_safe(problem, *, budget, log=False, surr='EAF', lambda_=12, eval_ratio=0.05):
    dim = get_dimension(problem)
    # cmaes = WrappedModcma(dim=dim)
    cmaes = WrappedCma(x0=np.zeros(dim), lambda_=lambda_)
    n_max = 20*dim

    if surr.upper() == 'EAF':
        surrogate = pre_and_post_process_data(partial(train_and_predict_ensemble,
                                                   dim=dim, epochs=1000,
                                                   plot=None, pbar=None,
                                                   lr=0.01, mse_stop=-np.inf,
                                                   width=128),
                                              n_max=n_max)
    elif surr.upper() == 'RAF':
        surrogate = pre_and_post_process_data(raf, n_max=n_max)
    else:
        raise ValueError(f"Wrong argument: {surr =}")

    seconds, (x, y) = time_first(seek_minimum)(cmaes,
                                               problem=problem,
                                               surrogate=surrogate,
                                               criterion=partial(select_eval,
                                                                 acquisition=partial(UCB, beta=1),
                                                                 eval_ratio=eval_ratio),
                                               budget=budget)
    if log:
        print(f"{get_evaluations(problem) = }")
        print(f"{budget = }")
        print(f"{final_target_hit(problem) = }")
        print(f"{progress(problem) =:.8%}")
        print(f"{seconds = }")
        print(f"{eval_ratio = }")

def show_2d(x_train, y_train, x_test):
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.scatter(x_test[:, 0], x_test[:, 1], color='tab:red', marker='x')
    plt.colorbar()

def show_2d_interactive(x_train, y_train, x_test, *, points=None):
    idx_train = np.arange(x_train.shape[0])
    fig = go.Figure(data=[go.Scatter(
        x=x_train[:, 0],
        y=x_train[:, 1],
        text=idx_train,
        name='train',
        mode='markers',
        marker=dict(
            color=idx_train,
            colorscale='Viridis',
            # opacity=0.8
        )
    ), go.Scatter(
        x=x_test[:, 0],
        y=x_test[:, 1],
        text=np.arange(y_train.shape[0]),
        name='test',
        mode='markers',
    )])
    if points is not None:
        fig.add_trace(go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            text=np.arange(points.shape[0]),
            name='points',
            mode='markers',
            # marker_symbol='x-thin',
        ))
    fig.show()

def show_3d(x_train, y_train, x_test=None, y_test=None):
    idx_train = np.arange(x_train.shape[0])
    fig = go.Figure(data=[go.Scatter3d(
        x=x_train[:, 0],
        y=x_train[:, 1],
        z=y_train.squeeze(), #flatten()
        text=idx_train,
        name='train',
        mode='markers',
        marker=dict(
            color=idx_train,
            colorscale='Viridis',
            # opacity=0.8
        )
    )])
    if x_test is not None and y_test is not None:
        fig.add_trace(go.Scatter3d(
            x=x_test[:, 0],
            y=x_test[:, 1],
            z=y_test.squeeze(),
            text=np.arange(y_test.shape[0]),
            name='test',
            mode='markers',
        ))
    fig.show()

def plot_predictions(pred_mean, pred_std, pred_std_scaled=None, y_test=None, idx_order=None, ax=None, color='tab:blue', label=None):
    if ax is None:
        fig, ax = plt.subplots()

    if idx_order is None:
        if y_test is not None:
            idx_order = y_test.argsort(axis=0).flatten()
        else:
            idx_order = slice(None)

    pred_mean = pred_mean[idx_order]
    pred_std = pred_std[idx_order]

    ax.plot(pred_mean, 'o', color=color, label=label)
    ax.vlines(x=np.arange(pred_mean.shape[0]), ymin=pred_mean-2*pred_std, ymax=pred_mean+2*pred_std, ls=':', lw=2, color=color) # colors='teal'

    if y_test is not None:
        ax.plot(y_test[idx_order], 'xr')

    if pred_std_scaled is not None:
        ax2 = ax.twinx()
        ax2.plot(pred_std_scaled[idx_order], 'o', color='tab:orange')
        ax2.axhline(0.125, linestyle='--', color='tab:pink') # ideal at the beggining
        ax2.axhline(0.075, linestyle='--', color='tab:brown')
        ax2.axhline(0.05, linestyle='--', color='tab:grey')
        ax2.axhline(0.025, color='tab:orange') # ideal at the end

def report(*, dim, fun, lambda_, width, lr, epochs, stop, criterion, budget, points="?", secs, problem, es, seed,):
    return " ".join([
        f"{dim}D",
        f"{fun}-fun",
        f"{lambda_:>2}-init-pop",
        *([] if all(a is None for a in (width, lr, epochs, stop)) else [
            f"{width}-width",
            f"{lr}-lr",
            f"{epochs}-epochs",
            f"{stop}-stop",
        ]),
        f"{format_criterion(criterion):<26}", # "lt-0.05" # todo strip function name
        f"{get_evaluations(problem)}-evals",
        f"{budget:>4}-budget",
        f"{points}-points",
        f"{secs:>4.0f}s",
        f"{progress(problem):.12%}", # .14
        f"{str(final_target_hit(problem)):>5}-solved",
        f"{es.restarts}-restarts",
        f"{seed}-seed",
    ])

def time_first(f):
    def f_(*args, **kwargs):
        t_0 = time.perf_counter()
        res = f(*args, **kwargs)
        return (time.perf_counter() - t_0, res)
    return f_

def set_seed(seed: int = 42) -> None:
    """https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # tf.random.set_seed(seed) # module 'tensorflow._api.v2.compat.v1.random' has no attribute 'set_seed'
    # tensorflow.random.set_seed(seed)
    # tf.random.set_random_seed(seed)
    # tensorflow.experimental.numpy.random.seed(seed) # module 'tensorflow._api.v2.compat.v1.experimental' has no attribute 'numpy'
    # tf.set_random_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

# TODO dataclass
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
SEED = (
    3  # rastrigin 3 restarts
    # 5
    # 37 # sphere all x[0] == 5
)
FUNCTION = 1
INSTANCE = 1
DIMENSION = 2
BUDGET = 250
EVAL_RATIO = 0.05
LAMBDA = (
    (4 + np.floor(3 * np.log(DIMENSION))).astype(int) # None
    # (8 + np.floor(6 * np.log(DIMENSION))).astype(int)
)
#------------------------------------------------------------------------------
if __name__ == "__main__":
    set_seed(SEED)
    problem = get_problem_ioh(FUNCTION, instance=INSTANCE, dimension=DIMENSION)
    cmaes_safe(problem, budget=250)
_='''
def fig_ax_dh():
    fig, ax = plt.subplots(layout="constrained")
    dh = display(fig, display_id=True)
    return (fig, ax, dh)

def format_plot(x=None, y=None):
  # plt.grid(False)
  ax = plt.gca()
  if x is not None:
    plt.xlabel(x, fontsize=20)
  if y is not None:
    plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()

legend = partial(plt.legend, fontsize=10)

# ntk
def plot_fn(train, test, *fs):
  train_xs, train_ys = train

  plt.plot(train_xs, train_ys, 'ro', markersize=10, label='train')

  if test != None:
    test_xs, test_ys = test
    plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')

    for f in fs:
      plt.plot(test_xs, f(test_xs), '-', linewidth=3)

  plt.xlim([-jnp.pi, jnp.pi])
  plt.ylim([-1.5, 1.5])

  format_plot('$x$', '$f$')

def plot3d(x, y, z, ax):
    x, y, z = map(np.array, (x,y,z))
    z = z.reshape(x.shape)
    # Set up plot

    ls = LightSource(270, 45)
    # To use a custom hillshading mode, override the built-in shading and pass
    # in the rgb colors of the shaded surface calculated from "shade".
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                        linewidth=0, antialiased=False, shade=False)
    return ax

def ecdf(value, points):
    return (points < value).mean()

def show_ecdf(x, xlim=None, vline=None):
    ins = np.sort(np.array(x).flatten())
    n = ins.shape[0]
    vals = np.array(range(1, n+1))/float(n)
    plt.plot(ins, vals)
    if vline is not None:
        plt.axvline(vline, color='tab:orange')
    plt.xlabel('pred std')
    if xlim is not None:
        plt.xlim(xlim)
    plt.ylabel('% of preds')


def measure_cmaes_evals(attempts, *, log=False, lambda_, dim, instances=None):
    evals_by_f = {}
    for fun in range(1, 24+1):
        evals_by_f[fun] = {}
        evals = []
        if instances is None:
            instances = [*range(1,5+1),*range(101,110+1)]
        for inst in instances:
            problem = ioh.get_problem(fun, instance=inst, dimension=dim, problem_class=ioh.ProblemClass.REAL)
            for attempt in range(attempts):
                _, ys, *_ = solve_cmaes_get_data(problem, pbar=False, lambda_=lambda_)
                evals.append(ys.shape[0])
            evals_by_f[fun][inst] = np.mean(evals)
            evals = []
        l = list(evals_by_f[fun].values())
        if log:
            print(fun, np.mean(l), np.std(l))
    return evals_by_f

def split_train_test(x, y, split_idx, test_size_max):
    train_x, test_x = np.split(x, [split_idx])
    train_y, test_y = np.split(y, [split_idx])
    test_x, test_y = test_x[:test_size_max], test_y[:test_size_max]
    return (train_x, train_y), (test_x, test_y)

def closest_n_idx(center, points, n):
    return np.linalg.norm(points - center, axis=1).argpartition(n-1)[:n]

def closest_n_idx_union(*, centers, points, n):
    if n >= (n_points := points.shape[0]):
        return np.arange(n_points)
    # np.unique(np.apply_along_axis(lambda t: closest_n_idx(t, train_x), 1, train_x_new)) # slower
    return np.unique(np.concatenate([closest_n_idx(center, points, n=n) for center in centers]))

# TODO logging
def choose_train_points(x_archive, y_archive, x_test, log=False):
    inds = train_points_near_test(x_test, x_archive)
    if inds.shape[0] == 0:
        raise ValueError("inds shape == 0")
    if log:
        print(f"{x_archive.shape = }")
        print(f"{y_archive.shape = }")
        print(f"{x_test.shape = }")
        print(f"{inds.shape = }")
    train_xs_subset = x_archive[inds, :]
    train_ys_subset = y_archive[inds, :]
    return train_xs_subset, train_ys_subset

def train_and_predict_ntk(x_train, y_train, x_test):
    W_std, b_std = (1.5, 0.05) # original
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1, W_std=W_std, b_std=b_std), stax.Erf(),
        stax.Dense(1, W_std=W_std, b_std=b_std)
    )
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnames='get')
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                        y_train, diag_reg=1e-4
                                                        )

    test_ntk_mean, test_ntk_covariance = predict_fn(x_test=x_test, get='ntk',
                                        compute_cov=True)
    test_ntk_mean = test_ntk_mean.squeeze() # jnp.reshape(test_ntk_mean, (-1,))
    return test_ntk_mean, test_ntk_covariance

def surrogate_ntk(x_archive, y_archive, x_test, subset=True):
    if subset:
        x_train, y_train = choose_train_points(x_archive, y_archive, x_test)
    else:
        x_train, y_train = x_archive, y_archive[:, jnp.newaxis]

    x_scale, x_scale_back = make_scaler(x_train)
    y_scale, y_scale_back = make_scaler(y_train)

    x_train_scaled = x_scale(x_train)
    y_train_scaled = y_scale(y_train)
    x_test_scaled = x_scale(x_test)
    y_pred_scaled, y_cov_scaled = train_and_predict_ntk(x_train_scaled, y_train_scaled, x_test_scaled)

    y_pred = y_scale_back(y_pred_scaled)
    y_std_scaled = jnp.sqrt(jnp.diag(y_cov_scaled))
    y_std = y_std_scaled*y_scale_back.std
    return y_pred, y_std, y_std_scaled

def surrogate_ensemble(x_archive, y_archive, x_test, *, epochs, plot, subset=True, pbar=False, width, lr, dim):
    if subset:
        x_train, y_train = choose_train_points(x_archive, y_archive, x_test)
    else:
        x_train, y_train = x_archive, y_archive

    x_scale, x_scale_back = make_scaler(x_train)
    y_scale, y_scale_back = make_scaler(y_train)

    x_train_scaled = x_scale(x_train)
    y_train_scaled = y_scale(y_train)
    x_test_scaled = x_scale(x_test)

    # mse_stop = y_train_scaled[closest_n_idx_union(centers=x_test_scaled, points=x_train_scaled, n=10)].var() / 100 # .std()
    mse_stop = -np.inf

    y_pred_scaled, y_std_scaled = train_and_predict_ensemble(x_train_scaled, y_train_scaled, x_test_scaled, dim=dim,
                                                             width=width, lr=lr, epochs=epochs, mse_stop=mse_stop,
                                                             plot=plot, pbar=pbar)

    y_pred = y_scale_back(y_pred_scaled)
    y_std = y_std_scaled*y_scale_back.std
    return y_pred, y_std, y_std_scaled

# TODO: generations are from 0
def cmaes_with_surrogate_ensemble(problem, *, budget, dim, epochs, plot=None, width, lr):
    problem.reset()
    cma = AskTellCMAES(d=dim,
                       budget=sys.maxsize,
                       # bound_correction='COTN',
                       bound_correction="saturate",
                       active=True)

    ys_preds = []
    ys_stds = []
    ys_stds_scaled = []

    x_archive: list[np.ndarray[(dim), np.float64]] = []
    y_archive: list[float] = []
    population: list[np.ndarray[(dim,1), np.float64]] = []
    for points in tqdm(itertools.count(1)):
        population.append(cma.ask())
        if len(population) == P_SIZE:
            if len(x_archive) > 0:
                xs: np.ndarray[(points, dim), np.float64] = np.array(x_archive)
                ys: np.ndarray[(points, 1), np.float64] = np.array(y_archive)[:, np.newaxis]
                pop: np.ndarray[(P_SIZE, dim)] = np.array(population).squeeze()
                ys_pred, ys_std, ys_std_scaled = surrogate_ensemble(xs, ys, pop, dim=dim, epochs=epochs, width=width, lr=lr,
                                                                    plot=plot, pbar=False)

                ys_preds.extend(ys_pred)
                ys_stds.extend(ys_std)
                ys_stds_scaled.extend(ys_std_scaled)

                std_max = ys_std_scaled.max()
                # std_min = ys_std_scaled.min()
                # std_threshold = 0.025
                std_threshold = 0.05

                y_curr_min = min(y_archive)
                for x, y_pred, y_std in zip(population, ys_pred, ys_std_scaled):
                    # if True:
                    # if y_pred < y_curr_min or y_std != std_min:
                    if y_pred < y_curr_min or y_std > std_threshold or y_std == std_max:
                        x_squeezed: np.ndarray[(dim), np.float64] = x.squeeze()
                        y: float = problem(x_squeezed) # Evaluate the objective function
                        x_archive.append(x_squeezed)
                        y_archive.append(y)
                        if problem.state.evaluations >= budget or final_target_hit(problem):
                            return points, x_archive, y_archive, ys_preds, ys_stds, ys_stds_scaled
                        cma.tell(x, y)
                    else:
                        cma.tell(x, y_pred)
            else:
                for x in population:
                    # Evaluate the objective function
                    x_squeezed: np.ndarray[(dim), np.float64] = x.squeeze()
                    y: float = problem(x_squeezed)
                    x_archive.append(x_squeezed)
                    y_archive.append(y)
                    cma.tell(x, y)
                    if final_target_hit(problem):
                        return points, x_archive, y_archive, ys_preds, ys_stds, ys_stds_scaled
            population = []
    return points, x_archive, y_archive, ys_preds, ys_stds, ys_stds_scaled

# Commented out IPython magic to ensure Python compatibility.
# %pdb on
# pdb on is trickey for a lot of cells, because the first failing destroys all


np.set_printoptions(precision=14, floatmode='fixed')
np.get_printoptions()
jax.config.update("jax_enable_x64", True)

# NTK
sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

"""~5h
```
 1    225.5266666666666    9.872687352264101
 2    529.6733333333334   19.329165067902494
 3  21495.486666666668  5489.426193433659
 4  69544.27333333333  25642.421220663928
 5     53.186666666666675 12.380784394464763
 6    500.52666666666664  23.355812029461855
 7   7448.826666666667  6758.978090457824
 8    619.8066666666667   97.22509598292453
 9    544.3866666666667   96.04706392643602
10    515.76              27.87194048979487
11    530.8266666666667   25.337678574714687
12   1269.8866666666668  530.7444204343765
13    694.3933333333333   37.302644529428335
14    557.8666666666667   15.720248796447915
15  22383.853333333333  8122.104943003109
16   4390.64            1721.2886102374969
17  34580.69333333333  10526.338108729402
18 155991.02666666667  40302.90204929775
19  19857.353333333333  5695.134911819229
20  34328.23333333333   9460.440773358407
21   9475.09333333333   4539.300304153591
22  12358.806666666667 10257.268874345755
23   2436.366666666667  1139.609410085559
24 321100.7133333333   30312.460368916858
```
local_restart="IPOP"
```
1 272.65333333333336 10.062727706188268
2 685.5400000000001 16.626757551208428
3 18013.57333333333 20860.457476078725
4 344366.04 33265.596279916586
5 42.67333333333333 8.21530820413263
6 606.1533333333334 57.60825018075874
7 605.88 275.04673590743323
8 678.26 75.61556982526812
9 642.1733333333335 73.04553344014647
10 733.2666666666667 162.86385589060436
11 839.2933333333333 378.3601467150343
12 1911.7 1444.3218713292408
13 984.34 86.02539780010707
14 752.9933333333333 29.06497242768729
15 16530.02 30824.109171927958
16 2041.9866666666667 1653.434835674579
17 3563.553333333333 2895.8977127117055
18 8406.1 7385.476351867901
19 6311.193333333333 2493.7323001120676
20 8603.186666666666 2322.511114825694
21 23771.066666666666 43377.60453009005
22 39344.24 64219.90086495826
23 90566.08666666666 33742.6256082099
24 372643.6066666667 21033.902918557193
time.perf_counter() - t_0 = 7001.2428089370005
CPU times: user 1h 55min 34s, sys: 15.2 s, total: 1h 55min 49s
Wall time: 1h 56min 41s
```
"""

# 1 239.0 0.0
# 2 450.0 0.0
# 3 7681.333333333333 0.0
# 4 74092.0 0.0
# 5 27.666666666666668 0.0
# 6 506.6666666666667 0.0
# 7 1528.3333333333333 0.0
# 8 509.6666666666667 0.0
# 9 556.3333333333334 0.0
# 10 508.0 0.0
# 11 526.3333333333334 0.0
# 12 532.3333333333334 0.0
# 13 718.3333333333334 0.0
# 14 551.0 0.0
# 15 71291.0 0.0
# 16 4681.333333333333 0.0
# 17 25443.333333333332 0.0
# 18 325805.6666666667 0.0
# 19 22734.0 0.0
# 20 13429.0 0.0
# 21 14750.333333333334 0.0
# 22 4207.666666666667 0.0
# 23 4249.333333333333 0.0
# 24 309394.0 0.0
# CPU times: user 5min 34s, sys: 1.58 s, total: 5min 35s
# Wall time: 5min 38s

# 1 5662.0 0.0
# 2 147535.66666666666 0.0
# 3 113532.33333333333 0.0
# 4 371769.0 0.0
# 5 46.0 0.0
# 6 29143.0 0.0
# 7 669.6666666666666 0.0
# 8 142929.33333333334 0.0
# 9 26533.666666666668 0.0
# 10 82654.0 0.0
# 11 66864.0 0.0
# 12 59409.666666666664 0.0
# 13 241425.0 0.0
# 14 115069.66666666667 0.0
# 15 60130.333333333336 0.0
# 16 21806.333333333332 0.0
# 17 135820.33333333334 0.0
# 18 399996.0 0.0
# 19 47033.333333333336 0.0
# 20 297352.3333333333 0.0
# 21 2874.3333333333335 0.0 *
# 22 1251.6666666666667 0.0 *
# 23 334682.0 0.0
# 24 399996.0 0.0
# CPU times: user 13min 25s, sys: 2.94 s, total: 13min 28s
# Wall time: 13min 34s



def print_time(f):
    def f_new(*args, **kwargs):
        t_0 = time.perf_counter()
        res = f(*args, **kwargs)
        print(f"{time.perf_counter() - t_0 = }")
        return res
    return f_new

# Commented out IPython magic to ensure Python compatibility.
# %%time
# dim = 2
# lambda_ = None
# # lambda_ = (4 + np.floor(3 * np.log(dim))).astype(int)
# # lambda_ = (8 + np.floor(6 * np.log(dim))).astype(int)
# evals_by_f = print_time(measure_cmaes_evals)(10, log=True, lambda_=lambda_, instances=None, dim=dim)

# PROBLEM CHOICE

# IDX:    0  1  2   3   4   5
DIM    = (2, 3, 5, 10, 20, 40)[(IDX:=0)]
P_SIZE = (6, 7, 8, 10, 12, 15)[IDX] # LAMBDA
# 1 "Sphere"
# 2 "Ellipsoid"
# 3 "Rastrigin"
# 4 "BuecheRastrigin"
# 5 "LinearSlope"
# 6 "AttractiveSector"
# 8 "Rosenbrock"
FUNCTION = 3
INST = 1
BUDGET = int(2*1e5*DIM) # Hansen

# In order to instantiate a problem instance, we can do the following:
problem = ioh.get_problem(
    FUNCTION,
    instance=INST,
    dimension=DIM,
    problem_class=ioh.ProblemClass.REAL
)
display(problem)
display(problem.meta_data.name)
display(problem.state)
display(problem.bounds)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # CMAES WITH ENSEMBLE SURROGATE
#
# # ST: 0 - None, all cmaes
# # ST: 1 - not new min, lower than 0.025, not current biggest var
#
# # D  FUN   WIDTH      LR      EPOCHS       STOP             SELECT   POINTS     TIME   EVALS      PROGRESS           SOLVED
# # 2D 1-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05  222-points   273s  90-evals  99.99999999286628% True-solved
# # 2D 1-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05  270-points   345s  97-evals  99.99999999946183% True-solved
# # 2D 1-fun 1024-width 0.06-lr  5000-epochs std-div-100-stop not      270-points   658s 270-evals  99.99999999561237% True-solved
# # 2D 1-fun 1024-width 0.06-lr  5000-epochs std-div-100-stop min-std  246-points   567s 214-evals  99.99999999354185% True-solved
# # 2D 1-fun 1024-width 0.06-lr    10-epochs var-div-100-stop lt-0.025 217-points    35s
# # 2D 1-fun 1024-width 0.06-lr   100-epochs var-div-100-stop lt-0.025 234-points    98s 229-evals
# # 2D 1-fun 1024-width 0.06-lr   300-epochs var-div-100-stop lt-0.025 204-points   346s 183-evals
# # 2D 1-fun 1024-width 0.06-lr   500-epochs var-div-100-stop lt-0.025 246-points   368s 141-evals                     True-solved
# # 2D 1-fun 1024-width 0.06-lr  1000-epochs var-div-100-stop lt-0.025            15730s 788-evals
# # 2D 1-fun 1024-width 0.06-lr  1000-epochs var-div-100-stop lt-0.025 250-points  1005s 177-evals
# # 2D 1-fun 1024-width 0.06-lr 10000-epochs var-div-100-stop lt-0.025 250-points   811s 130-evals  99.99933023509241%
# # 2D 1-fun 1024-width 0.06-lr 10000-epochs var-div-100-stop lt-0.025 320-points  1626s 147-evals  99.99999963239304%
# # 2D 1-fun 1024-width 0.06-lr 10000-epochs var-div-100-stop lt-0.025 234-points   683s 145-evals  99.99999999074134% True-solved
# # 2D 1-fun 1024-width 0.06-lr 20000-epochs var-div-100-stop lt-0.025 486-points  2700s 233-evals  99.99999999648664% True-solved
# # 2D 1-fun 1024-width 0.06-lr 20000-epochs var-div-100-stop lt-0.025 174-points   827s 161-evals  99.99999999601176% True-solved
# # 2D 1-fun 1024-width 0.06-lr 20000-epochs var-div-100-stop lt-0.025 198-points  1262s 173-evals  99.99999998902387% True-solved
# # 2D 1-fun 1024-width 0.06-lr 20000-epochs var-div-100-stop lt-0.025 258-points  1023s 163-evals  99.99999998869376% True-solved
# # 2D 2-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05  480-points   599s 165-evals 100.00000000242710% True-solved
# # 2D 2-fun 1024-width 0.06-lr 20000-epochs var-div-100-stop lt-0.025 288-points  8801s    -evals                     True-solved
# # 2D 2-fun 1024-width 0.06-lr 10000-epochs var-div-100-stop lt-0.025 396-points  2667s 246-evals 100.00000000261657% True-solved
# # 2D 2-fun 1024-width 0.06-lr 20000-epochs var-div-100-stop lt-0.025 672-points  7939s 419-evals 100.00000000108153% True-solved
# # 2D 5-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05   54-points    65s  25-evals 100.00000000000000% True-solved
# # 2D 6-fun 1024-width 0.06-lr   100-epochs        -inf-stop lt-0.05  582-points   105s 574-evals  99.99999998335494% True-solved
# # 2D 6-fun 1024-width 0.06-lr   300-epochs        -inf-stop lt-0.05  666-points   825s 575-evals  99.99999997226449% True-solved
# # 2D 6-fun 1024-width 0.06-lr  1000-epochs        -inf-stop lt-0.05  666-points  1210s 501-evals  99.99999999996373% True-solved
# # 2D 6-fun 1024-width 0.06-lr  5000-epochs var-div-100-stop lt-0.025 546-points  3757s 513-evals  99.99999997705498% True-solved
# # 2D 6-fun 1024-width 0.06-lr 10000-epochs var-div-100-stop lt-0.025 582-points 10514s 417-evals  99.99999998400942% True-solved
# # 2D 6-fun 1024-width 0.06-lr 10000-epochs var-div-100-stop min-std  456-points  9339s 421-evals  99.99999997306645% True-solved
# # 2D 6-fun 1024-width 0.06-lr  1000-epochs        -inf-stop min-std  444-points  2278s 413-evals  99.99999997349117% True-solved
# # 2D 6-fun 1024-width 0.06-lr  5000-epochs        -inf-stop lt-0.05  588-points  4545s 345-evals  99.99999997406836% True-solved
# # 2D 6-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05  606-points   805s 361-evals  99.99999997726978% True-solved
# # 2D 8-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05 1000-points  1266s 367-evals  99.99999998622594% False-solved
# # 2D 8-fun  128-width 0.01-lr  1000-epochs        -inf-stop lt-0.05 1086-points  1299s 398-evals  99.99999999576680% True-solved
#
# # fig, ax, dh = fig_ax_dh()
#
# secs, (points, xs, ys, y_pred, y_std, y_std_scaled) = time_first(cmaes_with_surrogate_ensemble)(
#     problem, dim=DIM, budget=250*DIM, epochs=(epochs := 1000), width=(width := 128), lr=(lr := 0.01),
#     # plot=(fig,ax,dh),
# )
#
# # ax.remove()
# # dh.update(fig)
# # plt.close(fig)
#
# print(f"{DIM}D {FUNCTION}-fun {width}-width {lr}-lr {epochs}-epochs -inf-stop lt-0.05 {points}-points {secs:.0f}s {problem.state.evaluations}-evals {problem_progress(problem):.14%} {final_target_hit(problem)}-solved")

print(f"{DIM}D {FUNCTION}-fun {width}-width {lr}-lr {epochs}-epochs -inf-stop lt-0.05 {points}-points {t_after-t_before:.0f}s {problem.state.evaluations}-evals {problem_progress(problem):.14%} {final_target_hit(problem)}-solved")

show_3d(np.array(xs), np.array(ys))

_ = np.array(y_std_scaled).squeeze()
show_ecdf(_)
print(f"{np.quantile(_, [0.9]) = }")
print(f"{ecdf(points=_, value=0.05) = }")

plt.plot(_)
plt.axhline(0.1)
plt.axhline(0.05)
plt.ylim(0, 0.2)

set_seed(2)
problem.reset()
cma = AskTellCMAES(d=2,
                   budget=sys.maxsize, # None and float('inf') does not work
                   # bound_correction='COTN',
                   bound_correction="saturate",
                   # lb=problem.bounds.lb,
                   # ub=problem.bounds.ub,
                   lambda_=None,
                   local_restart='IPOP',
                   # active=True
)
for sample in range(9809):
    x = cma.ask()
    y = problem(x.squeeze())
    if final_target_hit(problem):
        break
    cma.tell(x, y)

# x = cma.ask()
# y = problem(x.squeeze())
# breakpoint()
# cma.tell(x, y)

problem.state.evaluations

cma.ask_queue

x_next = cma.ask()

x_next

# SOLVE CMAES GET DATA
# TODO KEEP EPOCH DIMENSION

# ellipsoid 460-640
# lambda_ = (4 + np.floor(3 * np.log(DIM))).astype(int)
# lambda_ = (8 + np.floor(6 * np.log(DIM))).astype(int)
# set_seed(2)
lambda_ = None
xs, ys, cma = solve_cmaes_get_data(problem, lambda_=lambda_)

jump_inds = np.flatnonzero(np.diff(ys) > 30)
plt.plot(jump_inds, ys[jump_inds+1], 'or')
plt.plot(ys)
# plt.xlim(0, 10000)
print(f"{ys.shape = }")
print(f"{xs.min(axis=0) = }")
print(f"{xs.max(axis=0) = }")

ys = ys[:, np.newaxis]

6*95
6*95 + 12*(200-95)

print(f"{cma.parameters.lambda_ = }")
cma.parameters.restarts

print(cma.parameters.local_restart)

xs[(s := 9805): s+50]

ys[(s := 9805): s+50]

print(f"{cma.parameters.restarts = }")
print(f"{cma.parameters.last_restart = }")

# show_3d(np.array(xs), np.array(ys))

# surrogate_with_all_data_available
# Predictions for every epoch with all data up to that epoch available.
# TRAIN SURROGATE FOR EACH CMAES EPOCH - DIAGNOSTICS AFTER RUN SIMULATION

#     1663s
#     1237s
#     1937s
#     3042s
#     1066s 158-xs.shape
# 1-f  981s 270-ys.shape 1000-epochs
fig, ax = plt.subplots(layout="constrained")
disp = display(fig, display_id=True)
t_before = time.perf_counter()

y_pred_list = []
y_std_list = []
y_std_scaled_list = []

d,m = divmod(xs.shape[0], P_SIZE)
epoch_last = d + int(bool(m))

for epoch in tqdm(range(1, epoch_last)):
    sep = epoch * P_SIZE
    if sep >= xs.shape[0]:
        print("sep >= xs.shape[0]")
        break
    (train_x, train_y), (test_x, test_y) = split_train_test(xs, ys, sep, P_SIZE)
    if len(ys) == 0:
        print("len(ys) == 0")
        break
    test_y_pred, test_y_std, test_y_std_scaled = surrogate_ensemble(train_x, train_y, test_x, epochs=1000, plot=(fig, ax, disp), pbar=True)
    y_pred_list.extend(test_y_pred)
    y_std_list.extend(test_y_std)
    y_std_scaled_list.extend(test_y_std_scaled)

t_after = time.perf_counter()
ax.remove()
disp.update(fig)
plt.close(fig)
print(f"{t_after - t_before = }s")

y_std_scaled_arr = np.array(y_std_scaled_list).flatten()

plt.plot(y_std_scaled_arr)
plt.axhline(0.025, color='tab:orange')

print(f"{y_std_scaled_arr.min()  = }")
print(f"{y_std_scaled_arr.mean() = }")
print(f"{y_std_scaled_arr.max()  = }")

std_threshold = 0.025
print(f"{ecdf(std_threshold, y_std_scaled_arr) = }")
show_ecdf(y_std_scaled_arr, vline=std_threshold)

# T4 6-f 9-gen 10000-epochs -> 215-secs
# cpu 168
fig, ax = plt.subplots(layout="constrained")
disp = display(fig, display_id=True)
t_before = time.perf_counter()

test_y_pred, test_y_std, test_y_std_scaled = surrogate_ensemble(train_x, train_y, test_x,
                                                                epochs=10000, plot=(fig, ax, disp), pbar=True)

t_after = time.perf_counter()
print(f"{t_after - t_before =}")
ax.remove()
disp.update(fig)
plt.close(fig)

plot_predictions(test_y_pred, test_y_std, test_y_std_scaled, test_y)

train_x_subset, train_y_subset = choose_train_points(train_x, train_y, test_x)
print(f"{train_x.shape = }")
print(f"{train_y.shape = }")
print(f"{test_x.shape = }")
print(f"{train_x_subset.shape = }")
print(f"{train_y_subset.shape = }")

scale_x, scale_x_back = make_scaler(train_x)
train_x_scaled = scale_x(train_x)
test_x_scaled = scale_x(test_x)

scale_y, scale_y_back = make_scaler(train_y)
train_y_scaled = scale_y(train_y)

show_train_test_scaled()

nets = {f.__name__: NeuralNetwork(f, width=1024, dropout_p=.0, dim=DIM).to(device) for f in afs}
n_ = next(iter(nets.values()))
print(n_.name)
print(summary(n_))
del(n_)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # 6-f 10-epoch 10000-epochs -> 215-secs
#
# # CPU
# # 80.19142952499999
# # 70.81149720600001
# # 81.66691862699986
# # 69.96062193800003
# # 83.23485007499994
#
# # T4
# # 64.56545328599998
# # 59.345030338000015
# # 63.690081222
# # 62.15715136299991
# # 66.65955229200006
#
# # V100
# # 75.61252859799993
# # 66.80958531300007
# # 72.97188624699993
# # 72.12785687099995
# # 74.33493806999991
#
# # A100
# # 53.01978427100005
# # 45.43619919700001
# # 53.815523147999954
# # 48.03101792299998
# # 55.738864262999925
#
# # TODO !!!!!!!!!!!!!!! n=10
# mse_stop = train_y_scaled[closest_n_idx_union(centers=test_x_scaled, points=train_x_scaled, n=2)].var() / 100
# # mse_stop = -np.inf
# print(f"{mse_stop = }")
#
# fig, ax, dh = fig_ax_dh()
#
# losses = {}
# for name, net in tqdm(nets.items(), leave=False):
#     t_before = time.perf_counter()
#     losses[name] = train_network(net, train_x_scaled, train_y_scaled,
#                                  epochs=10000,
#                                  mse_stop=mse_stop,
#                                  lr=0.06,
#                                  plot=(fig, ax, dh),
#                                  #pbar=False
#                                  )
#     print(f"{time.perf_counter() - t_before =}")
#
# # clear_output(wait=True)
#
# for name, loss in losses.items():
#     ax.semilogy(loss, ',', label=name)
# handles, labels = ax.get_legend_handles_labels()
# patches = []
# for handle, label in zip(handles, labels):
#     patches.append(mpatches.Patch(color=handle.get_color(), label=label))
# ax.legend(handles=patches)
# dh.update(fig)
# plt.close()
# # fig.show()
#
# last = lambda x: x[-1]
# for name, loss in sorted({name: min(loss) for name, loss in losses.items()}.items(), key=lambda i: i[1]):
#     print(f"{name:<10} {loss}")

print(f"{'target:':<8} {mse_stop:.0e}")
print('-'*14)
for name, loss in sorted({name: min(loss) for name, loss in losses.items()}.items(), key=lambda i: i[1]):
    print(f"{name:<8} {loss:.0e}")

y_preds_scaled = {}
with torch.no_grad():
    for name, net in nets.items():
        y_preds_scaled[name] = net.eval()(torch.Tensor(scale_x(test_x))).numpy()

y_pred_arr_scaled = np.array(list(y_preds_scaled.values()))
pred_mean_scaled = y_pred_arr_scaled.mean(axis=0)
pred_std_scaled = y_pred_arr_scaled.std(axis=0)
test_mean = scale_y_back(pred_mean_scaled)
test_std = pred_std_scaled * scale_y_back.std

y_preds = {name: scale_y_back(y_) for name, y_ in y_preds_scaled.items()}
y_pred_arr = np.array(list(y_preds.values()))
pred_mean = y_pred_arr.mean(axis=0)
pred_std = y_pred_arr.std(axis=0)

train_y[closest_n_idx_union(centers=test_x_scaled, points=train_x_scaled, n=2)].var()

test_y.var()

test_mean.var()

plot_predictions(test_mean, test_std, pred_std_scaled, test_y)

for name, y_ in y_preds.items():
    plt.plot(y_, 'o', label=name)
plt.plot(test_y, 'xr')
plt.legend()

train_test = np.concatenate((train_x, test_x))
x0_min, x1_min = train_test.min(axis=0)
x0_max, x1_max = train_test.max(axis=0)

x0_shape = 100
x1_shape = 100
x0 = np.linspace(x0_min, x0_max, x0_shape)
x1 = np.linspace(x1_min, x1_max, x1_shape)

# x0 = np.linspace(-10, 10, x0_shape)
# x1 = np.linspace(-10, 10, x1_shape)

x_exp = np.dstack(np.meshgrid(x0, x1)).reshape(-1, DIM)
x_exp.reshape(x1_shape, x0_shape, DIM)

print(f"{x0.shape = }")
print(f"{x1.shape = }")

y_exps = {}
with torch.no_grad():
    for name, net in nets.items():
        y_exps[name] = net.eval()(torch.Tensor(scale_x(x_exp))).numpy()

y_exps_arr = np.array(list(y_exps.values()))
mean = y_exps_arr.mean(axis=0)
std = y_exps_arr.std(axis=0)

go.Figure(data=[
    go.Surface(name='mean', x=x0, y=x1, z=mean.reshape(x1_shape, x0_shape)),
    # go.Surface(name='+2std', x=x0, y=x1, z=(mean+2*std).reshape(x1_shape, x0_shape), opacity=0.5),
    # go.Surface(name='-2std', x=x0, y=x1, z=(mean-2*std).reshape(x1_shape, x0_shape), opacity=0.5),
    # go.Surface(name='relu', x=x0, y=x1, z=y_exps['relu'].reshape(x1_shape, x0_shape), opacity=0.7),
    # go.Surface(name='hardtanh', x=x0, y=x1, z=y_exps['hardtanh'].reshape(x1_shape, x0_shape), opacity=0.7),
    # go.Surface(name='erf',  x=x0, y=x1, z=y_exp_erf.reshape(x1_shape, x0_shape), opacity=0.9),
    go.Scatter3d(
        x=train_x[:, 0],
        y=train_x[:, 1],
        z=scale_y(train_y).flatten(),
        mode='markers',
        name='train'
    ),
    go.Scatter3d(
        x=test_x[:, 0],
        y=test_x[:, 1],
        z=scale_y(test_y).flatten(),
        text=np.arange(test_y.shape[0]),
        mode='markers',
        name='test'
    ),
]).show()

"""```python
# dim=2, f=1 (sphere), id=1, train-epoch=20000 epoch=486 evals=233
# std quantiles:
{0.1: 0.006636996241286397,
 0.2: 0.009417081065475941,
 0.5: 0.0194522924721241,
 0.525: 0.020781742641702294,
 0.55: 0.021732239425182348,
 0.5750000000000001: 0.02338252291083336,
 0.6000000000000001: 0.02590760476887227,
 0.6250000000000001: 0.027419377816841024,
 0.6500000000000001: 0.02889278251677752,
 0.6750000000000002: 0.029854719107970603,
 0.7000000000000002: 0.0314062748104334,
 0.7250000000000002: 0.03329867618158465,
 0.7500000000000002: 0.036335021257400554,
 0.7750000000000002: 0.03907364709302791,
 0.8000000000000003: 0.04231830239295961,
 0.8250000000000003: 0.04589901762083175,
 0.8500000000000003: 0.05202448442578323,
 0.8750000000000003: 0.05489300843328246,
 0.9000000000000004: 0.06337844878435142,
 0.9250000000000004: 0.07301782853901388,
 0.9500000000000004: 0.08721720688045087,
 0.9750000000000004: 0.12108497861772852,
 1: 0.24633292853832245}
```
"""

# https://stackoverflow.com/questions/10640759/how-to-get-the-cumulative-distribution-function-with-numpy
y_std_scaled_arr = np.array(y_std_scaled).flatten()
p_0_025 = (y_std_scaled_arr < 0.025).mean()
print(f"{p_0_025 = }")
# epoch - ((epoch - 6) * 0.45) # minimum of bb evals

ins = np.sort(y_std_scaled_arr)
n = ins.shape[0]
vals = np.array(range(n))/float(n)
plt.plot(ins, vals)
# plt.xlim(0, 0.5)
plt.axvline(0.025, color='tab:orange')
plt.axhline(p_0_025, color='tab:red', linestyle='--')
plt.xlabel('pred std')
_ = plt.ylabel('% of preds')

# quantiles = [0.1, 0.2, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
quantiles = [0.1, 0.2, *np.arange(0.5, 1, 0.025), 1]
qs = np.quantile(y_std_scaled, quantiles)
dict(zip(quantiles, qs))

_ = plt.hist(np.array(y_std_scaled).squeeze(), bins=100)

stds = np.array(y_std_scaled).flatten()

w = 1*P_SIZE
mva = np.concatenate([
    [stds[0]] + [stds[:i].mean() for i in range(2, w)],
    np.convolve(stds, np.ones(w), 'valid') / w
])
# print(f"{stds[:5] = }")
# print(f"{mva = }")
print(f"{mva.shape = }")
print(f"{stds.shape = }")
print(f"{(np.convolve(stds, np.ones(w), 'valid') / w).shape =}")

plt.plot(stds, ',', color='tab:red')
plt.plot(mva)
plt.axhline(0.025, color='tab:orange')

# -6, -5
# np.array(y_std_scaled).flatten().shape

# size -> seconds
#  5000 -> 9
#  6000 -> 14
#  7000 -> 15
#  8000 -> 19
#  9000 -> 27
# 10000 -> 34
# 11000 -> 45
# 12000 -> 64
# 13000 -> 63
# 14000 -> 73
# 15000 -> 82
pred, std, std_sc = surrogate(train_x[-15000:], train_y[-15000:], test_x, False)

train = (jnp.array(train_x), jnp.array(train_y[:, jnp.newaxis]))
test = (jnp.array(test_x), jnp.array(test_y[:, jnp.newaxis]))

if (subset:=False):
    idcs = train_points_near_test(test_x, train_x)
    train = (train_x[idcs], train_y[idcs])

init_fn, apply_fn, kernel_fn = stax.serial(
    # stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(1024, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(1, W_std=1.5, b_std=0.05)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get')

learning_rate = 0.1
training_steps = 10000

opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
opt_update = jit(opt_update)

loss = jit(lambda params, x, y: 0.5 * jnp.mean((apply_fn(params, x) - y) ** 2))
grad_loss = jit(lambda state, x, y: grad(loss)(get_params(state), x, y))

# 4096width, 500steps in 20m
# 1024width, 1200steps in 18:30m
train_losses = []
test_losses = []

key = random.PRNGKey(10)
key, net_key = random.split(key)
_, params = init_fn(net_key, (-1, 2))
opt_state = opt_init(params)

for i in trange(training_steps):
  opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

  train_losses += [loss(get_params(opt_state), *train)]
  test_losses += [loss(get_params(opt_state), *test)]

# plt.subplot(1, 2, 1)
#
# plt.loglog(ts, ntk_train_loss_mean, linewidth=3)
# plt.loglog(ts, ntk_test_loss_mean, linewidth=3)
#
plt.loglog(train_losses, 'k-', linewidth=2)
plt.loglog(test_losses, 'k-', linewidth=2)
#
# format_plot('Step', 'Loss')
# legend(['Infinite Train', 'Infinite Test', 'Finite'])
#
# plt.subplot(1, 2, 2)
#
# plot_fn(train, None)
#
# plt.plot(test_xs, ntk_mean, 'b-', linewidth=3)
# plt.fill_between(
#     jnp.reshape(test_xs, (-1)),
#     ntk_mean - 2 * ntk_std,
#     ntk_mean +  2 * ntk_std,
#     color='blue', alpha=0.2)
#
# plt.plot(test_xs, apply_fn(get_params(opt_state), test_xs), 'k-', linewidth=2)
#
# legend(
#     ['Train', 'Infinite Network', 'Finite Network'],
#     loc='upper left')
#
# finalize_plot((1.5, 0.6))

# 81.749X
print(test_y)
apply_fn(get_params(opt_state), test_x)

def train_network(key):
  train_losses = []
  test_losses = []

  _, params = init_fn(key, (-1, 2)) # DIM
  opt_state = opt_init(params)

  for i in range(training_steps):
    train_losses += [jnp.reshape(loss(get_params(opt_state), *train), (1,))]
    test_losses += [jnp.reshape(loss(get_params(opt_state), *test), (1,))]
    opt_state = opt_update(i, grad_loss(opt_state, *train), opt_state)

  train_losses = jnp.concatenate(train_losses)
  test_losses = jnp.concatenate(test_losses)
  return get_params(opt_state), train_losses, test_losses

params, train_loss, test_loss = train_network(key)

# 1 -> 91
# 2 -> 102 sec
# 3 -> 123
# 5 -> 157
ensemble_size = 5
ensemble_key = random.split(key, ensemble_size)
params, train_loss, test_loss = vmap(train_network)(ensemble_key)

plt.loglog(test_loss.mean(axis=0))
plt.loglog(train_loss.mean(axis=0))

ensemble_fx = vmap(apply_fn, (0, None))(params, test_x)

mean_fx = jnp.reshape(jnp.mean(ensemble_fx, axis=0), (-1,))
std_fx = jnp.reshape(jnp.std(ensemble_fx, axis=0), (-1,))

plt.plot(mean_fx - 2 * std_fx, 'k--', label='_nolegend_')
plt.plot(mean_fx + 2 * std_fx, 'k--', label='_nolegend_')
plt.plot(mean_fx, linewidth=2, color='black')

def cmaes_with_surrogate(problem, budget=int(2*1e5*DIM)):
    problem.reset()
    cma = AskTellCMAES(DIM, active=True)

    ys_preds = []
    ys_stds = []
    ys_stds_scaled = []

    x_archive: list[np.ndarray[(DIM), np.float64]] = []
    y_archive: list[float] = []
    population: list[np.ndarray[(DIM,1), np.float64]] = []
    for epoch in tqdm(range(1, budget+1)):
        population.append(cma.ask())
        if len(population) == P_SIZE:
            if len(x_archive) > 0:
                xs: jnp.ndarray[(epoch, DIM), np.float64] = jnp.array(x_archive)
                ys: jnp.ndarray[(epoch), np.float64] = jnp.array(y_archive)
                pop: jnp.ndarray[(P_SIZE, DIM)] = jnp.array(population).squeeze()
                ys_pred, ys_std, ys_std_scaled = surrogate(xs, ys, pop)
                ys_preds.extend(ys_pred)
                ys_stds.extend(ys_std)
                ys_stds_scaled.extend(ys_std_scaled)

                y_curr_min = min(y_archive)
                for x, y_pred, y_std in zip(population, ys_pred, ys_std_scaled):
                    # Evaluate the objective function
                    if y_pred < y_curr_min or y_std > 0.025:
                        x_squeezed: np.ndarray[(DIM), np.float64] = x.squeeze()
                        y: float = problem(x_squeezed)
                        x_archive.append(x_squeezed)
                        y_archive.append(y)
                        cma.tell(x, y)
                        if problem.optimum.y + 1e-8 >= problem.state.current_best.y:
                            return x_archive, y_archive, ys_preds, ys_stds, ys_stds_scaled
                    else:
                        cma.tell(x, y_pred)
            else:
                for x in population:
                    # Evaluate the objective function
                    x_squeezed: np.ndarray[(DIM), np.float64] = x.squeeze()
                    y: float = problem(x_squeezed)
                    x_archive.append(x_squeezed)
                    y_archive.append(y)
                    cma.tell(x, y)
                    if problem.optimum.y + 1e-8 >= problem.state.current_best.y:
                        return
            population = []

xs, ys, y_pred, y_std, y_std_scaled = cmaes_with_surrogate(problem)
print()
print(f"{problem.state.evaluations = }")
print(f"{problem.optimum.y / problem.state.current_best.y = }")

len(xs)
len(ys)
len(y_pred)
len(y_std)

epoch += 1

preds = jnp.array(y_pred[(epoch-1)*P_SIZE:epoch*P_SIZE])
stds = jnp.array(y_std[(epoch-1)*P_SIZE:epoch*P_SIZE])
stds_scaled = jnp.array(y_std_scaled[(epoch-1)*P_SIZE:epoch*P_SIZE])
y_true = jnp.array(ys[epoch*P_SIZE:(epoch+1)*P_SIZE])

fig, ax1 = plt.subplots()

ax1.plot(preds, label='pred')
ax1.plot(preds - 2*stds, "x", label='pred -2std')
ax1.plot(preds + 2*stds, "x", label='pred +2std')
ax1.plot(y_true, label='true')
ax1.grid(False)

ax2 = ax1.twinx()

ax2.plot(stds_scaled, 'o', label='std')
ax2.set_ylabel('std')

fig.legend()

# STD threshold
# 0.3
# 0.25
# 0.12
# 0.07
# 0.05
# 0.025

# plt.hist()
np.percentile(jnp.array(y_std_scaled), [10, 50, 60, 75, 90])

# STD vs Train Size
fig, ax1 = plt.subplots()
ax1.plot(
    jnp.array(y_std_scaled)[24:]
)

ax2 = ax1.twinx()
ax2.plot(
    np.array([[len(i)]*6 for i in global_idcs]).flatten()[24:],
    color='orange'
)

def scale(ys):
    """Heuristic scale, so the differences between y values are meaningful for the whole run."""
    #return np.log((ys - (ys_min := ys.min() * 0.999999)) / (ys.max() - ys_min))
    return (ys - (ys_min := ys.min())) ** (1/7)

# plt.scatter(xs[:,0], xs[:,1], c=ys)
plt.scatter(xs[:,0], xs[:,1], c=scale(ys))
# # plt.scatter(xs[:,0], xs[:,1], c=scale(ys), norm=matplotlib.colors.LogNorm())
# plt.scatter(xs[:,0], xs[:,1], c=ys, norm=matplotlib.colors.LogNorm())
plt.colorbar()

print(f"{xs.shape = }")
print(f"{ys.shape = }")
print(f"{ys[-2]=}")
print(f"{ys[-1]=}")

plt.plot(ys)
np.flatnonzero(ys > -300)

def simulate_run(xs, ys, epoch):
    print(f"{epoch=}")
    sep = P_SIZE*epoch   # N_min = 40*D # xs.shape[0] - p_size
    assert(sep -1 < xs.shape[0])

    # size = sep # N_max = None
    # train_xs = jnp.array(xs[max(sep-size, 0):sep])
    # train_ys = jnp.array(ys[max(sep-size, 0):sep, np.newaxis])
    train_xs = jnp.array(xs[:sep])
    train_ys = jnp.array(ys[:sep])
    test_xs = jnp.array(xs[sep:sep+P_SIZE])
    test_ys = jnp.array(ys[sep:sep+P_SIZE, np.newaxis])
    return train_xs, train_ys, test_xs, test_ys

train_xs, train_ys, test_xs, test_ys = simulate_run(xs, ys, epoch=600)
print(f"{train_xs.shape = }")
print(f"{test_xs.shape = }")

idcs = train_points_near_test(test_xs, train_xs)
print(f"{idcs.shape = }")
print(f"{idcs = }")
train_xs_subset = train_xs[idcs, :]
train_ys_subset = train_ys[idcs, jnp.newaxis]


train_xs_mean = train_xs_subset.mean(axis=0)
print(f"{train_xs_mean=}")
train_xs_std = train_xs_subset.std(axis=0)
print(f"{train_xs_std=}")

train_ys_mean = train_ys_subset.mean().item()
print(f"{train_ys_mean=}")
train_ys_std = train_ys_subset.std().item()
print(f"{train_ys_std=}")


train_xs_scaled = (train_xs_subset - train_xs_mean) / train_xs_std
train_ys_scaled = (train_ys_subset - train_ys_mean) / train_ys_std

test_xs_scaled = (test_xs - train_xs_mean) / train_xs_std
print(f"{test_xs_scaled.std(axis=0)=}")
test_ys_scaled = (test_ys - train_ys_mean) / train_ys_std
print(f"{test_ys_scaled = }")

# fig, ax = plt.subplots()
plt.scatter(train_xs_scaled[:,0], train_xs_scaled[:,1], c=train_ys_scaled)
plt.colorbar()
plt.scatter(test_xs_scaled[:,0], test_xs_scaled[:,1], color='blue')
for idx, x in enumerate(test_xs_scaled):
    plt.gca().annotate(idx, x)

# debug
# train_ys.shape
idcs
print(f"{train_ys[idcs].min() = }")
print(f"{train_ys[idcs].mean()= }")
print(f"{train_ys[idcs].max() = }")
train_ys[idcs]

W_std, b_std = (1.5, 0.05) # original

init_fn, apply_fn, kernel_fn = stax.serial(
    # stax.Dense(1, W_std=W_std, b_std=b_std), stax.Erf(),
    # stax.Dense(512, W_std=W_std, b_std=b_std), stax.Erf(),
    stax.Dense(1, W_std=W_std, b_std=b_std),
    stax.Erf(),
    stax.Dense(1, W_std=W_std, b_std=b_std)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get')

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_xs_scaled,
                                                      train_ys_scaled, diag_reg=1e-4
                                                      )

train_ntk_mean, train_ntk_covariance = predict_fn(x_test=train_xs_scaled, get='ntk',
                                      compute_cov=True)

train_ntk_mean = jnp.reshape(train_ntk_mean, (-1,))
train_ntk_var = jnp.diag(train_ntk_covariance)
train_ntk_std = jnp.sqrt(jnp.diag(train_ntk_covariance))

# display(train_ys_scaled.squeeze())
# display(ntk_mean)
# display(ntk_var)
# display(ntk_std)

fig, ax1 = plt.subplots()
ax1.plot(train_ntk_mean, label="mean pred")
ax1.plot(train_ys_scaled, color='tab:red', label='train y scaled')
fig.legend()

test_ntk_mean, test_ntk_covariance = predict_fn(x_test=test_xs_scaled, get='ntk',
                                      compute_cov=True)

test_ntk_mean = jnp.reshape(test_ntk_mean, (-1,))
test_ntk_var = jnp.diag(test_ntk_covariance)
test_ntk_std = jnp.sqrt(jnp.diag(test_ntk_covariance))

display(test_ys_scaled.squeeze())
display(test_ntk_mean)
display(test_ntk_var)
display(test_ntk_std)

train_ys_min = train_ys_scaled.min()
not_new_min = (test_ntk_mean > train_ys_min).nonzero()

print(f"{not_new_min = }")
print(test_ys[not_new_min].flatten().argsort())
print(test_ntk_mean[not_new_min].argsort())
print(stats.spearmanr(test_ys[not_new_min].flatten(), test_ntk_mean[not_new_min]).statistic)

fig, ax1 = plt.subplots()

ax1.plot(test_ntk_mean, label="mean pred")
ax1.plot(test_ntk_mean - 2*test_ntk_std, 'x') # tmp
ax1.plot(test_ntk_mean + 2*test_ntk_std, 'x') # tmp
ax1.plot(test_ys_scaled, color='tab:red', label='true y scaled')
ax1.axhline(train_ys_min, linestyle='--', color='tab:red', label='train min')

ax2 = ax1.twinx()

ax2.plot(test_ntk_std, 'o:', label='std')
ax2.set_ylabel('std')

fig.legend()

if TRANSFORMED_BACK := False:
    fig, ax = plt.subplots()
    ntk_test_ys_back = test_ntk_mean*train_ys_std+train_ys_mean
    ax.plot(ntk_test_ys_back, label="mean pred")
    ax.plot(ntk_test_ys_back - 2*test_ntk_std*train_ys_std, 'x')
    ax.plot(ntk_test_ys_back + 2*test_ntk_std*train_ys_std, 'x')
    ax.plot(test_ys, color='tab:red', label='true y')

def surrogate(train_x, train_y, test_x):
    W_std, b_std = (1.5, 0.05) # original
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(1, W_std=W_std, b_std=b_std), stax.Erf(),
        #stax.Dense(1, W_std=W_std, b_std=b_std), stax.Erf(),
        stax.Dense(1, W_std=W_std, b_std=b_std)
    )
    apply_fn = jit(apply_fn)
    kernel_fn = jit(kernel_fn, static_argnames='get')

    train_xs = jnp.array(train_x).squeeze()
    train_xs_mean = train_xs.mean(axis=0)
    train_xs_std = train_xs.std(axis=0)
    train_xs_scaled = (train_xs - train_xs_mean) / train_xs_std

    train_ys = jnp.array(train_y)
    train_ys_mean = train_ys.mean()
    train_ys_std = train_ys.std()
    train_ys_scaled = (train_ys - train_ys_mean) / train_ys_std
    train_ys_scaled = train_ys_scaled[:, jnp.newaxis]


    test_xs = jnp.array(test_x).squeeze()
    test_xs_scaled = (test_xs - train_xs_mean) / train_xs_std

    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_xs_scaled,
                                                        train_ys_scaled, diag_reg=1e-4
                                                        )
    ntk_mean, ntk_covariance = predict_fn(x_test=test_xs_scaled, get='ntk',
                                      compute_cov=True)

    ntk_mean = jnp.reshape(ntk_mean, (-1,)) * train_ys_std + train_ys_mean
    ntk_var = jnp.diag(ntk_covariance)

    return ntk_mean, ntk_var

# mean, var = surrogate(train_x,train_y, points)
# mean

# alternating
problem.reset()
cma = AskTellCMAES(dim, active=True)
train_x = []
train_y = []
iter = 0
points = []
mean = [0]*p_size
var = [0]*p_size

progress = lambda: problem.optimum.y / problem.state.current_best.y
while iter < 1000:
    # if (not hasattr(cma, "ask_queue")) or (len(cma.ask_queue) > 0):
    if len(points) < p_size:
        points.append(cma.ask())
        iter += 1
    else:
        # if surrogate_eval := False:
        if (surrogate_eval := (dim*5 <= len(train_x)) and (iter % (3*p_size) == 0)):
            mean, var = surrogate(train_x[-50:], train_y[-50:], points)

        if (real_eval := not surrogate_eval):
            train_x.extend(points)
        for (x, y_surr, v) in zip(points, mean, var):
            if surrogate_eval:
                cma.tell(x, y_surr)
            else:
                y = problem(x.reshape(dim))
                train_y.append(y)
                cma.tell(x, y)
        points = []

        #if problem.state.optimum_found:
        if problem.optimum.y + 1e-8 >= problem.state.current_best.y:
            break
        print(progress(), iter)
print(progress(), iter)
problem.state.evaluations

len(train_x)

problem.state.evaluations

x1, x2 = (
    np.array(xs)
        .transpose((1,0,2))
    )
y1 = np.array(ys).reshape((-1, 1))

x_cnt, y_cnt = 5, 6
x = np.linspace(problem.bounds.lb[0], problem.bounds.ub[0], x_cnt)
y = np.linspace(problem.bounds.lb[1], problem.bounds.ub[1], y_cnt)
display(x_cnt, y_cnt)

x, y = np.meshgrid(x, y)
# 2,3,4

points = np.array([x, y]).transpose((1, 2, 0)).reshape((-1, 2))
z = np.array(problem(points)).reshape((y_cnt, x_cnt))

# x, y, z

ax = plt.axes(projection='3d')
ax.scatter(x1, x2, y1, c='red')#, marker=m)

ax = plt.axes(projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0
                       #,facecolors=rgb, antialiased=False, shade=False
                       )
ax.scatter(x1, x2, y1, c='red')#, marker=m)

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# X, y = make_regression(n_samples=200, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=1000).fit(x_train, y_train)
# regr.predict(X_test[:2])
# regr.score(X_test, y_test)

regr.predict(points)

z = np.array(regr.predict(points)).reshape((y_cnt, x_cnt))

x.shape, y.shape, z.shape

ax = plt.axes(projection='3d')

surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0)

x_train = np.array(xs).reshape((-1, 2))
y_train = np.array(ys)#.reshape((-1, 1))
x_train.shape, y_train.shape

x_train.shape

from flax import linen as nn  # Linen API

class NN(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=500)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x

ones = lambda: jnp.ones((1, 2)) # jnp.ones((1, 28, 28, 1))
net = NN()
print(net.tabulate(jax.random.key(0)
                  ,ones()
                  ,compute_flops=True, compute_vjp_flops=True
                  ))

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate, momentum):
  """Creates an initial `TrainState`."""
  params = module.init(rng, ones())['params'] # initialize parameters by passing a template image
  tx = optax.sgd(learning_rate, momentum)
  return TrainState.create(
      apply_fn=module.apply, params=params, tx=tx,
      metrics=Metrics.empty())

@jax.jit
def train_step(state, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, batch['image'])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
    return loss
  grad_fn = jax.grad(loss_fn)
  grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state

@jax.jit
def compute_metrics(*, state, batch):
  logits = state.apply_fn({'params': state.params}, batch['image'])
  loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch['label']).mean()
  metric_updates = state.metrics.single_from_model_output(
    logits=logits, labels=batch['label'], loss=loss)
  metrics = state.metrics.merge(metric_updates)
  state = state.replace(metrics=metrics)
  return state

y_train.dtype

y_train.shape
x_train.shape

def get_datasets():
    # mnist = {
    #     'train': torchvision.datasets.MNIST('./data', train=True, download=True),
    #     'test': torchvision.datasets.MNIST('./data', train=False, download=True)
    # }

    ds = {}

    for split in ['train', 'test']:
        ds[split] = {
            'image': x_train, # mnist[split].data.numpy(),
            'label': y_train, # mnist[split].targets.numpy()
        }

        # cast from np to jnp and rescale the pixel values from [0,255] to [0,1]
        ds[split]['image'] = jnp.float32(ds[split]['image']) #/ 255
        ds[split]['label'] = jnp.float32(ds[split]['label'])

        # torchvision returns shape (B, 28, 28).
        # hence, append the trailing channel dimension.
        # ds[split]['image'] = jnp.expand_dims(ds[split]['image'], 3)

    return ds['train'], ds['test']

num_epochs = 10
batch_size = 32

train_ds, test_ds = get_datasets() # num_epochs, batch_size)

init_rng = jax.random.key(0)

learning_rate = 0.01
momentum = 0.9



# fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
# ax = plot3d(x, y, z, ax)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

#.optimum_found
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

train_points = 5
test_points = 50
noise_scale = 1e-1

target_fn = problem

key = random.PRNGKey(10)
key, x_key, y_key = random.split(key, 3)

train_xs = random.uniform(x_key, (train_points, 2), minval=-jnp.pi, maxval=jnp.pi)
problem(train_xs)

minval = problem.bounds.lb[0]
# minval = -jnp.pi
maxval = problem.bounds.ub[0]
# maxval = jnp.pi

train_xs = random.uniform(x_key, (train_points, 2), minval=minval, maxval=maxval)

train_ys = jnp.array(target_fn(train_xs)).reshape((train_points, 1))
train_ys += noise_scale * random.normal(y_key, (train_points, 1))
display(train := (train_xs, train_ys))

first = jnp.linspace(problem.bounds.lb[0], problem.bounds.ub[0], test_points)
second = jnp.linspace(problem.bounds.lb[1], problem.bounds.ub[1], test_points)
xa_plot, xb_plot = jnp.meshgrid(first,second)
xa_plot.shape, xb_plot.shape
# test_xs = jnp.array([xa_plot, xb_plot]).transpose((1,2,0)).reshape(-1, 2)
#
# test_ys = jnp.array(target_fn(test_xs)).reshape((-1, 1))
# (test := (test_xs, test_ys))

plot3d(xa_plot, xb_plot, test_ys)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(512, W_std=1.5, b_std=0.05), stax.Erf(),
    stax.Dense(1, W_std=1.5, b_std=0.05)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get')

key, net_key = random.split(key)
input_shape = (-1, 2)
_, params = init_fn(net_key, input_shape)
prior_ys = apply_fn(params, test_xs)

plot3d(xa_plot, xb_plot, prior_ys.reshape(test_points, test_points))



# https://matplotlib.org/stable/gallery/mplot3d/custom_shaded_3d_surface.html#sphx-glr-gallery-mplot3d-custom-shaded-3d-surface-py

# Load and format data
dem = cbook.get_sample_data('jacksboro_fault_dem.npz', np_load=True)
z = dem['elevation']
nrows, ncols = z.shape
x = np.linspace(dem['xmin'], dem['xmax'], ncols)
y = np.linspace(dem['ymin'], dem['ymax'], nrows)

dem['xmin'], dem['xmax'], dem['ymin'], dem['ymax']

z.shape

x.shape

x, y = np.meshgrid(x, y)

x.shape, y.shape

region = np.s_[5:50, 5:50]
x, y, z = x[region], y[region], z[region]



plot3d(x, y, z)

problem.bounds.lb[0], problem.bounds.ub[0]



x0 = np.random.uniform(problem.bounds.lb, problem.bounds.ub)
display(x0)

X = np.random.uniform(problem.bounds.lb, problem.bounds.ub, size=(5, problem.meta_data.n_variables))
X

problem.reset()





plot3d(x, y, z)





(x1 := cma.ask())

(x2 := cma.ask())

(x3 := cma.ask())

x1
# problem(x1)



xs = [cma.ask() for _ in (4,5,6)]
cma.ask_queue

cma.ask()
cma.ask_queue

problem.

cma = AskTellCMAES(2, budget=10, active=True, lb=problem.bounds.lb, ub=problem.bounds.ub)

problem.bounds.lb

(x1 := cma.ask())

cma.tell(x1, np.array([np.nan]))

cma.ask_queue

problem.bounds.lb

(x2 := cma.ask())

problem(x2.reshape()

import cma
import cma.fitness_models

fun = cma.ff.rosen
dimension = 40
es = cma.CMAEvolutionStrategy ( dimension * [0.1], 0.1 )
surrogate = cma.fitness_models.SurrogatePopulation ( fun )
while not es.stop():
    X = es.ask () # sample a new population
    print(X)
    break
    F = surrogate ( X ) # see Algorithm 1
    es.tell (X , F ) # update sample distribution
    es.inject ([ surrogate.model.xopt ])
    es.disp () # just checking what 's going on
len(X)



key = random.PRNGKey(10)

train_points = 5
test_points = 50
noise_scale = 1e-1

target_fn = lambda x: jnp.sin(x)
#---
key, x_key, y_key = random.split(key, 3)

train_xs = random.uniform(x_key, (train_points, 1), minval=-jnp.pi, maxval=jnp.pi)

train_ys = target_fn(train_xs)
train_ys += noise_scale * random.normal(y_key, (train_points, 1))
train = (train_xs, train_ys)
#---
test_xs = jnp.linspace(-jnp.pi, jnp.pi, test_points)
test_xs = jnp.reshape(test_xs, (test_points, 1))

test_ys = target_fn(test_xs)
test = (test_xs, test_ys)

plot_fn(train, test)
legend(loc='upper left')
finalize_plot((0.85, 0.6))

W_std, b_std = (1.5, 0.05)

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512, W_std=W_std, b_std=b_std), stax.Erf(),
    stax.Dense(512, W_std=W_std, b_std=b_std), stax.Erf(),
    stax.Dense(1, W_std=W_std, b_std=b_std)
)

apply_fn = jit(apply_fn)
kernel_fn = jit(kernel_fn, static_argnames='get')

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, train_xs,
                                                      train_ys, diag_reg=1e-4)

ntk_mean, ntk_covariance = predict_fn(x_test=test_xs, get='ntk',
                                      compute_cov=True)

ntk_mean = jnp.reshape(ntk_mean, (-1,))
ntk_std = jnp.sqrt(jnp.diag(ntk_covariance))

plot_fn(train, test)

# plt.plot(test_xs, nngp_mean, 'r-', linewidth=3)
# plt.fill_between(
#     jnp.reshape(test_xs, (-1)),
#     nngp_mean - 2 * nngp_std,
#     nngp_mean +  2 * nngp_std,
#     color='red', alpha=0.2)


plt.plot(test_xs, ntk_mean, 'b-', linewidth=3)
plt.fill_between(
    jnp.reshape(test_xs, (-1)),
    ntk_mean - 2 * ntk_std,
    ntk_mean +  2 * ntk_std,
    color='blue', alpha=0.2)

plt.xlim([-jnp.pi, jnp.pi])
plt.ylim([-1.5, 1.5])

legend(['Train', 'f(x)', #'Bayesian Inference',
        'Gradient Descent'],
       loc='upper left')

finalize_plot((0.85, 0.6))

# Commented out IPython magic to ensure Python compatibility.
# %%shell
# ## The cocoex does not work in colab, because installing it in the shell
# ## does not install it to the python.
# # git clone https://github.com/numbbo/coco.git
# # cd coco
# # python do.py run-python
# # python -c 'import cocoex; print("ok")'

_ = """
import os
os.chdir('coco')
print(os.getcwd())

import do
do.RELEASE = os.getenv('COCO_RELEASE', 'false') == 'true'
do.main(['run-python'])

import cocoex
"""

# W_std, b_std = (0.15, 0.005)
# W_std, b_std = (0.75, 0.025)
# W_std, b_std = (1.5, 0.05) # original
# W_std, b_std = (4.5, 0.15)
# W_std, b_std = (9, 0.30) # unstable
# W_std, b_std = (15, 0.5)
# W_std, b_std = (30, 1)

# W_std, b_std = (1, 1)
# W_std, b_std = (1, 10)
# W_std, b_std = (5, 50)
# W_std, b_std = (5, 100)
# W_std, b_std = (10, 100)
#W_std, b_std = (1, 10)
#W_std, b_std = (10, 100)
# W_std, b_std = (5, 50)
# W_std, b_std = (5, 45)
# W_std, b_std = (4, 45) # best
# W_std, b_std = (4, 40) # best
# W_std, b_std = (4, 25)
# W_std, b_std = (2.5, 25)
# W_std, b_std = (4, 25)
# W_std, b_std = (4, 30) # best
# W_std, b_std = (5, 30) # baad
# W_std, b_std = (4, 20) # best 5. iter
# W_std, b_std = (4, 15)
# W_std, b_std = (3, 10)
# W_std, b_std = (3, 8) # best 5. iter until 22. iter
# W_std, b_std = (3, 7)
# W_std, b_std = (1, 1)
# W_std, b_std = (0.1, 0.1)
# W_std, b_std = (0.01, 0.01)
# W_std, b_std = (0.005, 0.012)
# W_std, b_std = (0.005, 0.07)
# W_std, b_std = (0.01, 0.10)
# W_std, b_std = (0.01, 0.1)
# W_std, b_std = (0.1, 0.1)
# W_std, b_std = (0.05, 0.1)
# W_std, b_std = (0.025, 0.1)
# W_std, b_std = (0.025, 0.15) # kinda best for 25 iter. last 60 samples
# W_std, b_std = (0.025, 0.2)
# W_std, b_std = (0.025, 0.3)
# W_std, b_std = (0.025, 0.5) # promissing 0.0005
# W_std, b_std = (0.025, 0.75)
# W_std, b_std = (0.025, 1)
# W_std, b_std = (0.025, 2)
# W_std, b_std = (0.025, 10)
# W_std, b_std = (0.025, 100)
# --- centered ---
# W_std, b_std = (10, 0.05) # best iter 1
# W_std, b_std = (10, 10) # best iter 3
# W_std, b_std = (4, 20)
# W_std, b_std = (5, 15) # good after iter 3
# W_std, b_std = (8, 20)
# W_std, b_std = (3, 8) # iter 10 good
# W_std, b_std = (1, 8)
# W_std, b_std = (1, 5)
# W_std, b_std = (0.5, 2) # best iter 10
# W_std, b_std = (0.5, 3) # best iter 10
# W_std, b_std = (1, 2) # best iter 10
# W_std, b_std = (1.5, 3) # best iter 10
# W_std, b_std = (5, 3)

# (1,1) for std 16
# W_std, b_std = (0.5, 0.5) # for std 12, 2 it (10, 15)
# W_std, b_std = (1, 1) # for std 0.34 it 20
'''
