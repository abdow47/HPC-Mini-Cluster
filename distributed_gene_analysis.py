
from mpi4py import MPI
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Rank 0 loads the dataset and broadcasts it to all other ranks
if rank == 0:
    # Assuming leukemia_expression.csv is in the same directory or accessible
    df = pd.read_csv("leukemia_expression.csv")
    data = df.iloc[:, :-1].values
    target = df.iloc[:, -1].values
else:
    data = None
    target = None

data = comm.bcast(data, root=0)
target = comm.bcast(target, root=0)

# Divide data into chunks for each process
chunk_size = len(data) // size
start = rank * chunk_size
end = start + chunk_size if rank != size - 1 else len(data)
local_data = data[start:end]
local_target = target[start:end]

# Train a RandomForestClassifier on local data
clf = RandomForestClassifier(n_estimators=100)
clf.fit(local_data, local_target)

# Gather local scores from all processes to rank 0
local_score = clf.score(local_data, local_target)
scores = comm.gather(local_score, root=0)

# Rank 0 prints the results
if rank == 0:
    print("Bioinformatics scores from all nodes:", scores)
    print("Average score:", np.mean(scores))


