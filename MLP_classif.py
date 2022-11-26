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

# Enter the directory to the dataset
directory = 'terminology-project-2022/' # Path to the annotated terminology project data

# Model hyperparameters
batch_size = 3 # size of the training batches
num_epochs = 15 # number of training epochs
lr = 0.01 # learning rate
loss_fn = nn.CrossEntropyLoss() # loss function
hidden_size = 32 # size of the hidden layer in the MLP

#Creating train, test, and dev splits
train_data = TermsDataset(directory, 'train', one_hot=True)
test_data = TermsDataset(directory, 'test', one_hot=True)
dev_data = TermsDataset(directory, 'dev', one_hot=True)

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
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)
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
    # Make a copy of the model (to avoid changing the model outside this function)
    model_tr = copy.deepcopy(model)

    # Get model ready to train
    model_tr.train()

    # Define the optimizer
    optimizer = torch.optim.SGD(model_tr.parameters(), lr=learning_rate)

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

        # Checking validation accuracy, accuracy for each tag, and validation loss at each epoch
        accuracy, B_acc, I_acc, O_acc, val_loss = eval_mlp_classifier(model_tr, val_dataloader, loss_fn)

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
        I_total = 0
        I_correct = 0
        O_total = 0
        O_correct = 0
        B_total = 0
        B_correct = 0

        for text, labels in eval_dataloader:
            # Get the predicted labels
            y_predicted = model(text)
            loss = loss_fn(y_predicted, labels)

            total_loss += loss.item()
            # To get the predicted labels, we need to get the max over all possible classes
            _, label_predicted = torch.max(y_predicted.data, 1)

            # Compute accuracy overall accuracy for each token
            total_labels += labels.size(0)
            correct_labels += (label_predicted == labels).sum().item()
            # Loop to get the B, I, and O individual accuracies
            for index, lab in enumerate(labels):
                if lab == 0:
                    B_total += 1
                    if lab == label_predicted[index]:
                        B_correct += 1
                elif lab == 1:
                    I_total += 1
                    if lab == label_predicted[index]:
                        I_correct += 1
                elif lab == 2:
                    O_total += 1
                    if lab == label_predicted[index]:
                        O_correct += 1

    accuracy = 100 * correct_labels / total_labels
    final_loss = total_loss / total_labels
    B_acc = 100 * B_correct / B_total
    I_acc = 100 * I_correct / I_total
    O_acc = 100 * O_correct / O_total

    return accuracy, B_acc, I_acc, O_acc, final_loss


# Initialization (to be able to reproduce experiments)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0.01)


# Instantiating the model
model = MLPClassif(input_size, hidden_size, num_classes, nn.ReLU())
torch.manual_seed(0)
model.apply(init_weights)

# Run training Loop
model_tr, loss_all_epochs, accuracy_all_epochs, val_loss_all_epochs = training_mlp_classifier(model, train_dataloader, dev_dataloader,
                                                                                             num_epochs, loss_fn, lr)
# Save model
#torch.save(model_tr.state_dict(), 'model_mlp_classif_trained.pt')

# Model evaluation
test_accuracy, B_acc, I_acc, O_acc, test_loss = eval_mlp_classifier(model_tr, test_dataloader, loss_fn)

print(f'All training loss values: {loss_all_epochs}')
print(f'All validation loss values: {val_loss_all_epochs}')
print(f'Best validation Accuracy: {max(accuracy_all_epochs):.3f}%')
print(f'Test Accuracy: {test_accuracy:.3f}%')
print(f'B Accuracy: {B_acc:.3f}%')
print(f'I Accuracy: {I_acc:.3f}%')
print(f'O Accuracy: {O_acc:.3f}%')