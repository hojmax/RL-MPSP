## 🏄‍♂️ Usage

My report can be found [here](https://github.com/hojmax/RL-MPSP/blob/main/Speeding%20up%20stowage%20planning%20with%20deep%20reinforcement%20learning.pdf).

### Compiling C
The Makefile has two commands, one for compiling the C code on an ARM machine (e.g. M1 and M2 Macs), and one for compiling on an x86 machine. To compile on an ARM machine, run `make arm`. To compile on an x86 machine, run `make non-arm`.

### Local

For local use, simply run the `main.py` file.

### Colab

When using colab, add a cell to the top of the notebook with the following content:

```python
!git clone https://github.com/hojmax/RL-MPSP.git
%cd RL-MPSP
!pip install -r requirements.txt --quiet
```

If you also want to use the benchmarking data you should run the following command as well:

```python
!git clone https://[git_token]@github.com/hojmax/rl-mpsp-benchmark.git
```

The `git_token` should be replaced by your personal access token, and is required since the repo is private. You can generate a token by going to:

Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic)

In the final block you add:

```python
!python main.py [n_processes] [wandb_api_key] [wandb_note] [should_log_wandb]
```

## Benchmarking

The benchmarking data is obtained from the authors of the 2019 paper "Solution Strategies for a Multiport Container Ship Stowage Problem". They are available from the author (consuelo.parreno@uv.es) upon request.
