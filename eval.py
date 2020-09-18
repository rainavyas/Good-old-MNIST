import torch
import csv
from models import Simple_DNN, Basic_two_layer, Simple_CNN

IMG_DIM = 28

# Read in Data
X = []

with open('test.csv', newline='') as csvfile:
    myreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

    counter = 0

    for row in myreader:
        # ignore header of file
        if counter == 0:
            counter +=1
            continue
        row_vals = row[0].split(',')
        row_vals = [int(val) for val in row_vals]
        X.append(row_vals)

X = torch.FloatTensor(X)

# Reshape into image
X = torch.reshape(X, (-1, IMG_DIM, IMG_DIM))

# Load model
model_path = "Simple_CNN.pt"
model = torch.load(model_path)
model.eval()

y_probs = model.forward(X)
y = torch.argmax(y_probs, dim = 1)
y = y.tolist()

# Write to csv file
file_name = "Simple_CNN_test_output.csv"
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header
    writer.writerow(['ImageId', 'Label'])
    for id, pred in enumerate(y):
        writer.writerow([id+1 , pred])
