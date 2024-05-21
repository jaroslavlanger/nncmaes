import sys
from enum import Enum, auto
from statistics import mean
from math import isfinite

class State(Enum):
    CMA = auto()
    NN = auto()
    TRAIN = auto()
    SUBSET = auto()
    SUMMARY = auto()
S = State


N = 5
subset_sizes = []
tau_pop_list = []
tau_train_list = []
train_tau_mean = [[]]*N
train_tau_first = [[]]*N
train_tau_last = [[]]*N

# lines = sys.stdin.read().split('\n')[:-1]
with open(sys.argv[1]) as f: lines = f.readlines()
state = S.CMA
while lines:
    match state:
        case S.CMA:
            line, *lines = lines
            assert line.startswith('(')
            state = S.SUBSET if lines[0].startswith('subset') else S.NN
        case S.NN:
            if lines[0].startswith('('):
                state = S.CMA
                continue
            if len(lines[0]) > 2 and lines[0][2] == 'D':
                state = S.SUMMARY
                continue
            sep = 7
            length = N*sep
            nn, lines = lines[:length], lines[length:]
            assert nn[0].strip() == '', nn
            assert nn[-1].startswith('epoch: 1000')
            for i in range(N):
                curr, nn = nn[:sep], nn[sep:]
                train_tau_mean[i].append(mean(float(l.split('=')[1]) for l in curr[2:]))
                train_tau_last[i].append(float(curr[-1].split("=")[1]))
                train_tau_first[i].append(float(curr[2].split("=")[1]))
            state = S.TRAIN
        case S.TRAIN:
            tau_train, tau_archive, row, tau_pop, *lines = lines
            assert tau_train.startswith('tau-train'), tau_train
            assert tau_archive.startswith('tau-archive'), tau_archive
            assert row.startswith('eval'), row
            assert tau_pop.startswith('tau-population'), tau_pop
            tau_train_list.append(float(tau_train.split('=')[1]))
            tau_pop_list.append(float(tau_pop.split('=')[1]))
            state = S.SUBSET if lines[0].startswith('subset') else S.NN
        case S.SUBSET:
            subset_size, *lines = lines
            subset_sizes.append(int(subset_size.split('=')[1]))
            state = S.NN
        case S.SUMMARY:
            summary, *lines = lines
            print(summary[:99])

print(f"| {mean(subset_sizes) = }")
print(f"| {mean(tau_train_list) = :.2f}")
print(f"| {mean(filter(isfinite, tau_pop_list)) = :.2f}")
print(f"| {[mean(l) for l in train_tau_mean] = }")
print(f"| {[mean(l) for l in train_tau_first] = }")
print(f"| {[mean(l) for l in train_tau_last] = }")
