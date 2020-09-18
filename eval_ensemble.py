import torch
import csv
from models import Simple_DNN, Basic_two_layer, Simple_CNN

IMG_DIM = 28
num_seeds = 5

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

# Load models and evaluate
y_all_seeds = []

for seed in range(1,num_seeds+1):
    print("On seed "+str(seed))
    model_path = "Simple_CNN_seed"+str(seed)+".pt"
    model = torch.load(model_path)
    model.eval()

    y_probs = model.forward(X)

    # Normalise the probabilities
    sf = torch.nn.Softmax(dim=1)
    y_normalised = sf(y_probs)

    y_all_seeds.append(y_normalised)

# Average the ensemble predictions
y_overall = torch.stack(y_all_seeds, dim=0).sum(dim=0)
y = torch.argmax(y_overall, dim=1)
y = y.tolist()

# Write to csv file
file_name = "Simple_CNN_seed_ensemble_test_output.csv"
with open(file_name, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Write header
    writer.writerow(['ImageId', 'Label'])
    for id, pred in enumerate(y):
        writer.writerow([id+1 , pred])
