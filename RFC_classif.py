from preprocessing import TermsDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

directory = 'terminology-project-2022/' # Path to the annotated terminology project data

print('Loading data...')
train_data = TermsDataset(directory, 'train', one_hot=True)
test_data = TermsDataset(directory, 'test', one_hot=True)
dev_data = TermsDataset(directory, 'dev', one_hot=True)

# Unzipping the data
train_split = list(zip(*train_data))
test_split = list(zip(*test_data))
dev_split = list(zip(*dev_data))

# Converting the train data from tensors to numpy arrays
train_data, train_labels = train_split
train_data = [np.array(x) for x in train_data]
train_labels = [np.array(x) for x in train_labels]

# Converting the train data from tensors to numpy arrays
test_data, test_labels = test_split
test_data = [np.array(x) for x in test_data]
test_labels = [np.array(x) for x in test_labels]

print('Training Random Forest Classifier...')
rfc_classifier = RandomForestClassifier()
rfc_classifier.fit(train_data, train_labels)

# Evaluating the Random Forest Classifier on the test data
rfc_scores = rfc_classifier.score(test_data, test_labels)
preds = rfc_classifier.predict(test_data)
precision, recall, f1, support = precision_recall_fscore_support(test_labels, preds, labels=[0,1,2])

# Printing the overall accuracy and the precision, recall, and F1-score results
print(f'Overall Test Accuracy: {rfc_scores}')
for index, (P,R,F) in enumerate(zip(precision, recall, f1)):
    if index == 0:
        print('B')
    elif index == 1:
        print('I')
    else:
        print('O')
    print(f'Precision: {P}')
    print(f'Recall: {R}')
    print(f'F1: {F}\n')
