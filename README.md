
### Overview

This is the repo for [arXiv:2308.07152](https://arxiv.org/abs/2308.07152). 

- `lib/` is the directory for source codes of the stabilizer scheme and the Linearity Attack.
- `scripts.py` is the source code for generating data, stored in `data/`, which will be used for plotting figures in our paper by `proc_data.py`. The figures are stored in `fig/`.
- `challenge/` contains a challenge instance, stored in `challenge_H.txt`. It requires 10000 samples (bit strings), which will be checked with `lib.hypothesis.hypothesis_test` and the hidden secret. 


### Environment

The specific version of necessary python packages used are as follows.

| Library  | Version |
|----------|---------|
| NumPy    | 1.23.4  |
| galois   | 0.1.1   |