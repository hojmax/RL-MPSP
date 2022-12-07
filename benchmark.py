import os
import numpy as np


def get_benchmarking_data(path):
    """Go through all files in the directory and return a list of dictionaries with:
     N, R, C, seed, transportation_matrix"""
    output = []

    for file in os.listdir(path):
        if file.endswith(".txt"):
            with open(os.path.join(path, file), "r") as f:
                lines = f.readlines()
                output.append({
                    "N": int(lines[0].split(": ")[1]),
                    "R": int(lines[1].split(": ")[1]),
                    "C": int(lines[2].split(": ")[1]),
                    "seed": int(lines[3].split(": ")[1]),
                    "transportation_matrix": np.loadtxt(lines[4:], dtype=int)
                })

    return output
