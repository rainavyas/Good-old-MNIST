import torch
import csv

# Read in Data
y = []
X = []

with open('train.csv', newline='') as csvfile:
    myreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    counter = 0

    for row in myreader:
        # ignore header of file
        if counter == 0:
            counter +=1
            continue
        row_vals = row[0].split(',')
        row_vals = [int(val) for val in row_vals]
        X.append(row_vals[1:])
        y.append(row_vals[0])


print(len(X))
print(len(y))


# Separate into Training and Dev sets (80:20 ratio)
training_size = 0.8*len(X)
X_train = X[:training_size]
X_dev = X[training_size:]
y_train = y[:training_size]
y_dev = y[training_size:]

# Convert to tensors
X_train = torch.FloatTensor(X_train)
X_dev = torch.FloatTensor(X_dev)
y_train = torch.FloatTensor(y_train)
y_dev = torch.FloatTensor(y_dev)
