import os
import torch
import csv
from glob import glob


os.mkdir("data")

for file in os.listdir("data_raw"):
    print(file)
    data = []

    with open(f"data_raw/{file}") as f:
        reader = csv.reader(f)

        for row in reader:
            data.append([float(row[0]), float(row[1])])
    
    data = torch.tensor(data)
    torch.save(data, f"data/{file[:-4]}.pt")
