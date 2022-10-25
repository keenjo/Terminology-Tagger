import spacy
import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

nlp = spacy.load('en_core_web_sm')

directory = '.../terminology-project-2022' # Path to the annotated terminology project data
data_path = '/.../' # Path where you want the preprocessed data to be saved

# Lists for each of the dataset splits: train, dev, test
train = []
dev = []
test = []
splits = [train, dev, test]

# List to iterate through the dataset split files for preprocessing
splits_name = ['train', 'dev', 'test']

# Dictionary which will contain the inputs for each dataset split
data_total = {'train': [],
              'dev': [],
              'test': []}

# Dictionary which will contain the annotated tags for each dataset split
combined_tags = {'train': [],
                'dev': [],
                'test': []}


# Collecting all of the final annotation files
for root, dirs, files in os.walk(directory):
    for file in sorted(files):
        if file.endswith('.final'):
            if 'train' in root:
                train.append(os.path.join(root, file))
            elif 'dev' in root:
                dev.append(os.path.join(root, file))
            elif 'test' in root:
                test.append(os.path.join(root, file))


# Preprocessing loop
for split_index, split in enumerate(splits):
    split_tags = []
    for fpath in tqdm(split):
        #print(fpath)
        with open(fpath, 'r+') as f:
            text = f.readlines()

        words = [] # all of the words for a given document
        doc_tags = [] # all of the term tags for a given document
        POS_list = [] # all of the word POS tags for a given document

        # Splitting the txt file into the word and tag lists
        for line in text:
            #print(line)
            if line != '\n' and line != '':
                try:
                    word, tag = line.split('\t')
                except ValueError:
                    word, tag = line.split(' ')
                tag = ''.join(tag.split())
                if word != '' and tag != '':
                    words.append(word)
                    doc_tags.append(tag)
                elif word == '' and tag == '':
                    continue
                elif word != '' and tag == '':
                    words.append(word)
                    doc_tags.append('O')

        # Getting the POS tags of all of the words
        doc = nlp(' '.join(words))
        for token in doc:
            POS_list.append(token.pos_)

        # Appending each input word into the inputs dictionary
        for index, word in enumerate(words):
            # Inputs dictionary: contains all input information for a given split
            inputs = {}
            # Adding all of the information to the inputs dictionary
            inputs['Main word'] = word
            inputs['Main POS'] = POS_list[index]
            inputs['is_first'] = 'True' if index == 0 else 'False'
            inputs['is_capitalized'] = 'True' if word.lower() != word else 'False'
            #inputs['Word Tag'] = labels[index]
            inputs['Preceding word'] = '<bos>' if index == 0 else words[index - 1]
            inputs['Preceding POS'] = '---' if index == 0 else POS_list[index - 1]
            inputs['Preceding Tag'] = '---' if index == 0 else doc_tags[index - 1]
            inputs['Following word'] = '<eos>' if index == len(words) - 1 else words[index + 1]
            inputs['Following POS'] = '---' if index == len(words) - 1 else POS_list[index + 1]
            inputs['Following Tag'] = '---' if index == len(words) - 1 else doc_tags[index + 1]
            data_total[splits_name[split_index]].append(inputs)

        # Adding the tags from one document to list of tags for data split (train, dev, or test)
        for tag in doc_tags:
            split_tags.append(tag)
    # Adding the tags from data split to a combined_tags dictionary
    for tag in split_tags:
        combined_tags[splits_name[split_index]].append(tag)


# Vectorizing the data
# Creating a vocabulary for all of the data
vocab = []
# Used separate lists here just to organize things a bit more
just_words = [] # List of all unique words
just_tags = [] # List of all unique term tags
just_pos = [] # List of all unique POS tags
just_bool = ['False', 'True']

for split in data_total:
    for input in data_total[split]:
        for key in input:
            vocab.append(input[key])
            if 'word' in key.lower():
                just_words.append(input[key])
            elif 'pos' in key.lower():
                just_pos.append(input[key])
            elif 'tag' in key.lower():
                just_tags.append(input[key])

just_words = list(set(just_words))
just_pos = list(set(just_pos))
just_tags = list(set(just_tags))
# Converting the vocabulary to token:integer dictionary
vocab = just_bool + just_tags + just_pos + just_words
tok_int = {}
for index, item in enumerate(vocab):
    tok_int[item] = index + 1

# Updating the data and tag dictionaries with the vocabulary integer values
for split in data_total:
    for input in data_total[split]:
        for key in input:
            input[key] = tok_int[input[key]]

for split in combined_tags:
    for index, tag in enumerate(combined_tags[split]):
        combined_tags[split][index] = tok_int[tag]

int_tok = {v: k for k, v in tok_int.items()}

# Saving the data (inputs, tags, vocabulary) as json files
with open(f'{data_path}/inputs.json', 'w+') as file:
    json.dump(data_total, file)

with open(f'{data_path}/tags.json', 'w+') as file:
    json.dump(combined_tags, file)

with open(f'{data_path}/vocab_tok-int.json', 'w+') as file:
    json.dump(tok_int, file)

with open(f'{data_path}/vocab_int-tok.json', 'w+') as file:
    json.dump(int_tok, file)

print(f'All inputs, tags, and vocabulary saved to {data_path}')


'''
Vectors as one hot encoding
- I didn't save these values in a file like the other data. I just figured we could copy this loop into the script 
  where we create the model since the one hot encodings would create a super big file. 
'''
total_arrs = [] # 3D list of all of the final arrays

for split in data_total:
    split_arrs = [] # List of arrays for an entire dataset split
    for input in data_total[split]:
        arrs_list = [] # List of arrays for one document
        for key in input:
            if 'word' in key.lower():
                arr = np.zeros(len(just_words))
                arr[just_words.index(int_tok[input[key]])] = 1
                arrs_list.append(arr)
            if 'tag' in key.lower():
                arr = np.zeros(len(just_tags))
                arr[just_tags.index(int_tok[input[key]])] = 1
                arrs_list.append(arr)
            if 'pos' in key.lower():
                arr = np.zeros(len(just_pos))
                arr[just_pos.index(int_tok[input[key]])] = 1
                arrs_list.append(arr)
            if 'is' in key.lower():
                arr = np.zeros(len(just_bool))
                arr[just_bool.index(int_tok[input[key]])] = 1
                arrs_list.append(arr)
        split_arrs.append(np.hstack(arrs_list).reshape(9, 906)) # Append the arrays from one document to list of arrays for a dataset split & reshape
    total_arrs.append(split_arrs) # Append all arrays for a given dataset split to a list that will contain all of the arrays

# Testing to make sure outputs are organized as planned
print(len(total_arrs)) # This should be 3 for the 3 dataset splits (train, dev, test)
print(total_arrs[0][0].shape)
print(total_arrs[0][0])


# ************************************************************
# ************************************************************

# Testing the data on a Decision Tree Classifier
#classifier = Pipeline([
    #('vectorizer', DictVectorizer()),
    #('classifier', DecisionTreeClassifier())
#])
'''
vectorizer = DictVectorizer()
train_vect = vectorizer.fit_transform(data_total['train'])
test_vect = vectorizer.fit_transform(data_total['test'])
print(train_vect.shape)
print(test_vect.shape)
#reformed_data = vectorizer.inverse_transform(train_vect)

classifier = DecisionTreeClassifier()

classifier.fit(train_vect, combined_tags['train'])

res_test = classifier.score(test_vect, combined_tags['test'])
print(f'Accuracy test: {res_test}')
'''