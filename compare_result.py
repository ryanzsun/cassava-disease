import pandas as pd
import numpy as np

lmh = pd.read_csv("./train_650_svm_submission-1.csv")
sz = pd.read_csv("./sample_submission_file_2.csv")

# print(lmh)
# print(sz)
count = 0
for index, row in lmh.iterrows():
    if row["Category"]==sz.loc[sz["Id"] == row["Id"]]["Category"].item():
        count +=1
    # else:
    #     print(row["Category"], sz.loc[sz["Id"] == row["Id"]]["Category"].item())
print(count, 3773-count)