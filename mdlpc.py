import pandas as pd
import numpy as np

# Load your dataset into a pandas dataframe
df = pd.read_csv("your_dataset.csv")

# Define the continuous variable that you want to discretize
continuous_var = "age"

# calculate the optimal number of intervals using the MDLPC criterion


def calculate_optimal_intervals(data):
    # assuming data is a numpy array
    n = len(data)
    # calculate the entropy of the data
    data_values = set(data)
    freq_dist = {}
    for value in data_values:
        freq_dist[value] = len([x for x in data if x == value])/n
    entropy = sum([-1*p*np.log2(p) for p in freq_dist.values()])
    # calculate the minimum description length
    mdl = n*entropy
    # initialize the number of intervals to 1
    k = 1
    # initialize the minimum description length
    mdl_min = mdl
    for k in range(2, int(np.sqrt(n))):
        intervals = np.linspace(min(data), max(data), k+1)
        interval_freqs = []
        for i in range(len(intervals)-1):
            interval_freqs.append(
                len([x for x in data if x >= intervals[i] and x < intervals[i+1]]))
        interval_entropies = []
        for freqs in interval_freqs:
            interval_entropies.append(sum([-1*p*np.log2(p) for p in freqs/n]))
        mdl = sum(interval_entropies) + \
            sum(np.array(interval_freqs)/n*np.log2(n))
        if mdl < mdl_min:
            mdl_min = mdl
            k_optimal = k
    return k_optimal


n_intervals = calculate_optimal_intervals(df[continuous_var])

# Discretize the continuous variable into the optimal number of intervals
df[continuous_var] = pd.cut(
    df[continuous_var], n_intervals, labels=range(n_intervals))
