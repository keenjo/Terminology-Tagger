import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split, Subset
import copy
from tqdm import tqdm
from preprocessing import TermsDataset

directory = 'terminology-project-2022/' # Path to the annotated terminology project data
vocab_path = '[ENTER VOCAB PATH]'

#Creating train, test, and dev splits
train_data = TermsDataset(directory, vocab_path, 'train', one_hot=True)
test_data = TermsDataset(directory, vocab_path, 'test', one_hot=True)
dev_data = TermsDataset(directory, vocab_path, 'dev', one_hot=True)

# Getting label statistics for Random Sampler (to have a better distribution of classes in each batch)
# Splitting the data into two lists of data and labels
train_split = list(zip(*train_data))
test_split = list(zip(*test_data))
dev_split = list(zip(*dev_data))

# Converting labels to integers
train_labels = train_split[1]
train_labels = [int(label) for label in train_labels]
test_labels = test_split[1]
test_labels = [int(label) for label in test_labels]
dev_labels = dev_split[1]
dev_labels = [int(label) for label in dev_labels]

# Getting the counts of all of the labels in each dataset split
train_labels_unique, train_count = np.unique(train_labels, return_counts=True)
test_labels_unique, test_count = np.unique(test_labels, return_counts=True)
dev_labels_unique, dev_count = np.unique(dev_labels, return_counts=True)

# Getting the weight for each label in each split
train_class_weights = [sum(train_count)/c for c in train_count]
test_class_weights = [sum(test_count)/c for c in test_count]
dev_class_weights = [sum(dev_count)/c for c in dev_count]

# Adding the label weights to a list for each split
train_weights = [train_class_weights[e] for e in train_labels]
test_weights = [test_class_weights[e] for e in test_labels]
dev_weights = [dev_class_weights[e] for e in dev_labels]

# Creating the Weighted Random Samplers
train_sampler = WeightedRandomSampler(train_weights, len(train_labels))
test_sampler = WeightedRandomSampler(test_weights, len(test_labels))
dev_sampler = WeightedRandomSampler(dev_weights, len(dev_labels))

# Loading Data into DataLoader
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=5)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=5)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=5)
batch_data, batch_name = next(iter(train_dataloader))

# Define model input size and final number of classes
input_size = batch_data.shape[1]
num_classes = 3

# MLP Classification model
class MLPClassif(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, act_fn):
        super(MLPClassif, self).__init__()

        self.input_layer = nn.Sequential(nn.Linear(input_size, hidden_size),
                                         act_fn,
                                         nn.BatchNorm1d(num_features=hidden_size))
        self.hidden_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                          act_fn,
                                          nn.BatchNorm1d(num_features=hidden_size))
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        y = self.input_layer(x)
        z = self.hidden_layer(y)
        out = self.output_layer(z)

        return out


# Training function
def training_mlp_classifier(model, train_dataloader, val_dataloader, num_epochs, loss_fn, learning_rate, verbose=True):
    # Make a copy of the model (avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)

    # Set the model in 'training' mode
    model_tr.train()

    # Define the optimizer
    optimizer = torch.optim.Adam(model_tr.parameters(), lr=learning_rate)

    # Initialize a list to record the training loss over epochs
    loss_all_epochs = []
    accuracy_all_epochs = []
    val_loss_all_epochs = []

    # Training loop
    for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        loss_current_epoch = 0

        # Iterate over batches using the dataloader
        for batch_index, (text, labels) in tqdm(enumerate(train_dataloader), desc=f'Training Loop Epoch {epoch+1}'):
            y_pred = model_tr.forward(text)
            loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_current_epoch += loss.item()

        # Checking validation accuracy at each epoch
        accuracy, val_loss = eval_mlp_classifier(model_tr, val_dataloader, loss_fn)

        # Early stopping implementation
        if accuracy_all_epochs != []:
            if accuracy > max(accuracy_all_epochs):
                torch.save(model_tr.state_dict(), 'models/model_MLP_classif.pt')
                print(f'-----> Old Best Accuracy: {max(accuracy_all_epochs)}')
                print(f'-----> Current Best Accuracy: {accuracy}')

        accuracy_all_epochs.append(accuracy)
        val_loss_all_epochs.append(val_loss)

        # At the end of each epoch, record and display the loss over all batches
        loss_all_epochs.append(loss_current_epoch / (batch_index + 1))
        if verbose:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss_current_epoch / (batch_index + 1)))

    return model_tr, loss_all_epochs, accuracy_all_epochs, val_loss_all_epochs


# Evaluation function
def eval_mlp_classifier(model, eval_dataloader, loss_fn):
    # Set the model in evaluation mode
    model.eval()

    with torch.no_grad():
        correct_labels = 0
        total_labels = 0
        total_loss = 0

        for text, labels in eval_dataloader:
            # Get the predicted labels
            y_predicted = model(text)
            loss = loss_fn(y_predicted, labels)

            total_loss += loss.item()
            # To get the predicted labels, we need to get the max over all possible classes
            _, label_predicted = torch.max(y_predicted.data, 1)

            # Compute accuracy: count the total number of samples, and the correct labels (compare the true and predicted labels)
            total_labels += labels.size(0)
            #print(f'Preds: {label_predicted}')
            #print(labels)
            correct_labels += (label_predicted == labels).sum().item()

    accuracy = 100 * correct_labels / total_labels
    final_loss = total_loss / total_labels

    return accuracy, final_loss


# initialization (to ensure reproducibility)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)


# Instantiating the model
model = MLPClassif(input_size, 512, num_classes, nn.ReLU())
torch.manual_seed(0)
model.apply(init_weights)

# Training parameters
num_epochs= 10
lr = 0.01
loss_fn = nn.CrossEntropyLoss()

# Run training Loop
model_tr, loss_all_epochs, accuracy_all_epochs, val_loss_all_epochs = training_mlp_classifier(model, train_dataloader, dev_dataloader,
                                                                                             num_epochs, loss_fn, lr)

# Save model
#torch.save(model_tr.state_dict(), 'model_mlp_classif_trained.pt')

# Model evaluation
test_accuracy = eval_mlp_classifier(model_tr, test_dataloader, loss_fn)

print(f'All training loss values: {loss_all_epochs}')
print(f'All validation loss values: {val_loss_all_epochs}')
print(f'Best validation Accuracy: {max(accuracy_all_epochs):.3f}%')
print(f'Test Accuracy: {test_accuracy:.3f}%')
