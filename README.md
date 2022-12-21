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

The ```git_token``` should be replaced by your personal access token, and is required since the repo is private. You can generate a token by going to:

Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic)

In the final block you add:

```python
!python main.py [n_processes] [wandb_api_key] [wandb_note]
```

## 🏋️ Weights & Biases

You will need an API key when connecting. This can be found in your settings.

You can access [our W&B team here](https://wandb.ai/rl-msps).

## Benchmarking

The benchmarking data is obtained from the authors of the 2019 paper "Solution Strategies for a Multiport Container Ship Stowage Problem". They are available from the author (consuelo.parreno@uv.es) upon request.