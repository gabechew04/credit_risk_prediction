import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns

data = pd.read_csv("datasets/master.csv")

def group_grade(grade):
    grouping = {char.upper() : 0 for char in "ABCD"} + {char.upper() : 1 for char in "EFG"}
    return grade.replace(grouping)

data_copy = data.copy()
data_copy["grade"] = group_grade(data["grade"])

data_copy