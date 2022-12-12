import pandas as pd
import numpy as np
import os


def get_benchmarking_data(path):
    """Go through all files in the directory and return a list of dictionaries with:
     N, R, C, seed, transportation_matrix, paper_result"""

    output = []
    df = pd.read_excel(
        os.path.join(path, "paper_results.xlsx"),
    )

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                lines = f.readlines()
                N = int(lines[0].split(": ")[1])
                R = int(lines[1].split(": ")[1])
                C = int(lines[2].split(": ")[1])
                seed = int(lines[3].split(": ")[1])

                paper_result = df[
                    (df['N'] == N) &
                    (df['R'] == R) &
                    (df['C'] == C) &
                    (df['seed'] == seed)
                ]['res'].values

                assert len(paper_result) == 1
                paper_result = paper_result[0]

                output.append({
                    "N": N,
                    "R": R,
                    "C": C,
                    "seed": seed,
                    "transportation_matrix": np.loadtxt(lines[4:], dtype=int),
                    "paper_result": paper_result
                })

    return output
