## 🏄‍♂️ Usage

### Local

For local use, simply run the `main.py` file.

### Colab

When using colab, add a cell to the top of the notebook with the following content:

```python
!git pull
%cd
!git clone https://github.com/hojmax/RL-MPSP.git
%cd RL-MPSP
!pip install -r requirements.txt --quiet
```

The `!git pull` means that you only have to execute 'Restart, and run all' for changes to take effect. If you also want to use the benchmarking data you should run the following command as well:

```python
!git clone https://[git_token]@github.com/hojmax/rl-mpsp-benchmark.git
```

The `git_token` should be replaced by your personal access token, and is required since the repo is private. You can generate a token by going to:

Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic)

In the final block you add:

```python
!python main.py [n_processes] [wandb_api_key] [wandb_note] [should_log_wandb]
```

### Hendrix Cluster

The first time you enter the cluster you should download the repos and install the required packages:

```bash
git clone https://github.com/hojmax/RL-MPSP.git && cd ./RL-MPSP && git clone https://[git_token]@github.com/hojmax/rl-mpsp-benchmark.git && module load python/3.9.9 && pip install -r requirements.txt
```

You can then request resources (4 hours in this case):

```bash
srun -p gpu --pty --time=04:00:00 --gres gpu:1 bash
```

And run the python script:

```bash
!python main.py [n_processes] [wandb_api_key] [wandb_note] [should_log_wandb] 0
```

The zero at the makes sure that the script does not try to show a progress bar, since this is not supported on the cluster.

See [Hendrix documentation](https://diku-dk.github.io/wiki/slurm-cluster) for more information.

## 🏋️ Weights & Biases

You will need an API key when connecting. This can be found in your settings.

You can access [our W&B team here](https://wandb.ai/rl-msps).

## Benchmarking

The benchmarking data is obtained from the authors of the 2019 paper "Solution Strategies for a Multiport Container Ship Stowage Problem". They are available from the author (consuelo.parreno@uv.es) upon request.
