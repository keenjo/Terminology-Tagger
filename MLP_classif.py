import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import DataLoader, random_split, Subset
import copy
from preprocessing import TermsDataset

directory = 'terminology-project-2022/' # Path to the annotated terminology project data
vocab_path = '[ENTER VOCAB PATH]'

#Creating train, test, and dev splits
train_data = TermsDataset(directory, vocab_path, 'train', one_hot=True)
test_data = TermsDataset(directory, vocab_path, 'test', one_hot=True)
dev_data = TermsDataset(directory, vocab_path, 'dev', one_hot=True)

# Loading Data into DataLoader
train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=5, shuffle=True)
dev_dataloader = DataLoader(dev_data, batch_size=5, shuffle=True)
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
                                         nn.Dropout(0.2))
        self.hidden_layer = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                          act_fn,
                                          nn.Dropout(0.2))
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, output_size),
                                          nn.Softmax())

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

    # Training loop
    for epoch in range(num_epochs):
        # Initialize the training loss for the current epoch
        loss_current_epoch = 0

        # Iterate over batches using the dataloader
        for batch_index, (text, labels) in enumerate(train_dataloader):
            y_pred = model_tr.forward(text)
            loss = loss_fn(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_current_epoch += loss.item()

        # Checking validation accuracy at each epoch
        accuracy = eval_mlp_classifier(model_tr, val_dataloader)

        # Early stopping implementation
        if accuracy_all_epochs != []:
            if accuracy > max(accuracy_all_epochs):
                # torch.save(model_tr.state_dict(), 'models/model_opt.pt')
                print(f'-----> Old Best Accuracy: {max(accuracy_all_epochs)}')
                print(f'-----> Current Best Accuracy: {accuracy}')

        accuracy_all_epochs.append(accuracy)

        # At the end of each epoch, record and display the loss over all batches
        loss_all_epochs.append(loss_current_epoch / (batch_index + 1))
        if verbose:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss_current_epoch / (batch_index + 1)))

    return model_tr, loss_all_epochs, accuracy_all_epochs


# Evaluation function
def eval_mlp_classifier(model, eval_dataloader):
    # Set the model in evaluation mode
    model.eval()

    with torch.no_grad():
        correct_labels = 0
        total_labels = 0

        for text, labels in eval_dataloader:
            # Get the predicted labels
            y_predicted = model(text)

            # To get the predicted labels, we need to get the max over all possible classes
            # !! May not need to do this since we are using softmax in the last layer
            _, label_predicted = torch.max(y_predicted.data, 1)

            # Compute accuracy: count the total number of samples, and the correct labels (compare the true and predicted labels)
            total_labels += labels.size(0)
            print(f'Preds: {label_predicted}')
            print(labels)
            correct_labels += (label_predicted == labels).sum().item()
    accuracy = 100 * correct_labels / total_labels

    return accuracy


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
model_tr, loss_all_epochs, accuracy_all_epochs = training_mlp_classifier(model, train_dataloader, dev_dataloader, num_epochs, loss_fn, lr)

# Save model
#torch.save(model_tr.state_dict(), 'model_mlp_classif_trained.pt')

# Model evaluation
test_accuracy = eval_mlp_classifier(model_tr, test_dataloader)

print(loss_all_epochs)
print(accuracy_all_epochs)
print(test_accuracy)
