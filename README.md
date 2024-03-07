
### Overview

This is the repo for [arXiv:2308.07152](https://arxiv.org/abs/2308.07152). 

- `lib/` is the directory for source codes of the stabilizer scheme, the Linearity Attack and the new attacks from [deobfuscate-iqp](https://github.com/goliath-klein/deobfuscate-iqp/tree/4bbbc3f0e059dc521d5e2aab5a162fc67fc94fe0), which is linked to the paper [Secret extraction attacks against obfuscated IQP circuits](https://arxiv.org/abs/2312.10156).
- `scripts.py` is the source code for generating data, stored in `data/`, which will be used for plotting figures in our paper by `proc_data.py`. The figures are stored in `fig/`.
- `challenge/` contains a challenge instance, stored in `challenge_H.txt`. It requires 10000 samples (bit strings), which will be checked with `lib.hypothesis.hypothesis_test` and the hidden secret. 


### Environment

The specific version of necessary python packages used are as follows.

| Library  | Version |
|----------|---------|
| NumPy    | 1.25.2  |
| galois   | 0.3.7   |