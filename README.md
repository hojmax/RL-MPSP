## 🏄‍♂️ Usage

### Local

For local use, simply run the `main.ipynb` notebook.

### Colab

When using colab, upload the `main.ipynb` notebook. Afterwards, add a cell to the top of the notebook with the following content:

```python
!git clone https://github.com/hojmax/RL-MPSP.git
%cd /content/RL-MPSP
!pip install -r requirements.txt
```

When pushing to the repository, you need to run `!git pull` and restart the runtime for the changes to take effect.

If you also want to use the benchmarking data you should run the following command as well:
```python
!git clone https://git_token@github.com/hojmax/rl-mpsp-benchmark.git
```

The ```git_token``` should be replaced by your personal access token, and is required since the repo is private. You can generate a token by going to:

Settings -> Developer Settings -> Personal Access Tokens -> Tokens (classic)


## 🏋️ Weights & Biases

You will need an API key when connecting. This can be found in your settings.

You can access [our W&B team here](https://wandb.ai/rl-msps).

## Benchmarking

The benchmarking data is obtained from the authors of the 2019 paper "Solution Strategies for a Multiport Container Ship Stowage Problem". They are available from the author (consuelo.parreno@uv.es) upon request.