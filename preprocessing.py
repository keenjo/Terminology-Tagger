import spacy
import numpy as np
import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

nlp = spacy.load('en_core_web_sm')

directory = '.../terminology-project-2022' # Path to the annotated terminology project data
vocab_path = '/.../' # Path where you want the vocabularies to be saved (so data can be decoded later if we want)

class TermsDataset(Dataset):

    def __init__(self, directory, split, one_hot=False):
        self.directory = directory # Directory where data is stored
        self.split = split # Dataset split -> train, dev, or test
        self.one_hot = one_hot # Choice of whether or not to do one-hot encoding

        # Preparing/Organizing the data
        self.data, self.tags, self.words_total, self.tags_total, self.POS_total = self.prep_data()
        # Creating the vocabulary
        self.tok_int, self.int_tok, self.just_words, self.just_tags, self.just_pos, self.just_bool = self.create_vocab()
        # Encoding the data
        if self.one_hot == False:
            '''
            Choice of data encoding:
            - Non one-hot: as N dimensional vector where N is the number of features
            - One-hot: each feature encoded in binary with N dimensions, where N is the number of options for each feature
                - All of the feature vectors are then combined and reshaped into one vector
            '''
            self.encoded_data, self.encoded_tags = self.encode_data()
        else:
            self.encoded_data, self.encoded_tags = self.one_hot_encode_data()

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, index):
        return self.encoded_data[index], self.encoded_tags[index]

    def prep_data(self):
        '''
        Function to read the data from the files and organized it with additional features
        - Returns:
            * Important for creation of the data inputs and hypothesized outputs
            - data_final: list of data for the chosen data split
            - tags_final: list of hypothesized term tags for the chosen data split
            * Important for the creation of the vocabulary in the create_vocab() fxn
                - must include data from all data splits so the encoding/decoding values are identical between splits
            - words_total: all of the words for all of the data splits combined
            - tags_total: all of the term tags for all of the data splits combined
            - POS_total: all of the POS tags for all of the data splits combined
        '''
        data_paths = [] # All of the data paths for all of the data splits combined
        data_final = [] # All of the data for the chosen data split [train, dev, test]
        tags_final = [] # All of the term tags for the chosen data split [train, dev, test]

        words_total = [] # All of the words for all of the data splits combined
        tags_total = [] # All of the term tags for all of the data splits combined
        POS_total = [] # All of the POS tags for all of the data splits combined

        # Collecting all of the final annotation files
        for root, dirs, files in os.walk(self.directory):
            for file in sorted(files):
                if file.endswith('.final'):
                    #if self.split in root:
                    data_paths.append(os.path.join(root, file))

        # Preprocessing loop
        for fpath in data_paths:
            #print(fpath)
            with open(fpath, 'r+') as f:
                text = f.readlines()

            words = [] # all of the words for a given document
            doc_tags = [] # all of the term tags for a given document
            POS_list = [] # all of the word POS tags for a given document

            # Splitting the txt file into the word and tag lists
            for line in text:
                if line == '\n':
                    words.append('<EOS>')
                    doc_tags.append('O')
                elif line != '\n' and line != '':
                    try:
                        word, tag = line.split('\t')
                    except ValueError:
                        word, tag = line.split()
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

            for word in words:
                words_total.append(word)
            for tag in doc_tags:
                tags_total.append(tag)
            for pos in POS_list:
                POS_total.append(pos)

            # Creating a dictionary of features for each input (ONLY FOR THE CHOSEN DATA SPLIT)
            # - we can experiment with these features and definitely add more as we think of them
            # - ** if we change the features, the create_vocab() and one_hot_encode_data() fxns will need to be updated accordingly
            if self.split in fpath: # Making sure to only organize data for the chosen data split
                for index, word in enumerate(words):
                    if word != '<EOS>':
                        # Inputs dictionary: contains all input information for a given document
                        inputs = {}
                        # Adding all of the information to the inputs dictionary
                        inputs['Main word'] = word
                        inputs['Main POS'] = POS_list[index]
                        inputs['is_first'] = 'True' if index == 0 or words[index - 1] == '<EOS>' else 'False'
                        inputs['is_capitalized'] = 'True' if word.lower() != word else 'False'
                        inputs['Preceding word'] = '<BOS>' if index == 0 or words[index - 1] == '<EOS>' else words[index - 1]
                        inputs['Preceding POS'] = 'X' if index == 0 or words[index - 1] == '<EOS>' else POS_list[index - 1]
                        inputs['Preceding Tag'] = 'O' if index == 0 or words[index - 1] == '<EOS>' else doc_tags[index - 1]
                        inputs['Following word'] = '<EOS>' if index == len(words) - 1 else words[index + 1]
                        inputs['Following POS'] = 'X' if index == len(words) - 1 else POS_list[index + 1]
                        inputs['Following Tag'] = 'O' if index == len(words) - 1 else doc_tags[index + 1]
                        # Appending one input to the data list
                        data_final.append(inputs)

                # Adding the tags from one document to list of tags for data split (train, dev, or test)
                for tag in doc_tags:
                    tags_final.append(tag)

        return data_final, tags_final, words_total, tags_total, POS_total

    def create_vocab(self):
        '''
        Creating a vocabulary for all of the data
        - Returns:
            - tok_int: dictionary mapping tokens to integers
            - int_tok: inverse of tok_int mapping integers to tokens
            - just_words: List of all of the unique words
            - just_tags: List of all of the unique Term tags
            - just_pos: List of all of the unique POS tags
            - just_bool: List of all of the unique boolean values [True, False]
        '''

        vocab = [] # total vocab of train, test and split sets
        # Used separate lists here just to organize things a bit more
        just_words = self.words_total # List of all unique words
        just_tags = self.tags_total # List of all unique term tags
        just_pos = self.POS_total # List of all unique POS tags
        just_bool = ['False', 'True']


        for input in self.data:
            for key in input:
                vocab.append(input[key])
                if 'word' in key.lower():
                    just_words.append(input[key])
                elif 'pos' in key.lower():
                    just_pos.append(input[key])
                elif 'tag' in key.lower():
                    just_tags.append(input[key])

        just_bool = ['False', 'True']
        just_words = sorted(list(set(just_words)))
        just_tags = sorted(list(set(just_tags)))
        just_POS = sorted(list(set(just_pos)))

        # Converting the vocabulary to token:integer dictionary
        vocab = just_bool + just_tags + just_POS + just_words
        tok_int = {'<PAD>': 9000, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        for index, item in enumerate(vocab):
            if item not in tok_int.keys():
                tok_int[item] = index + 4 # + 4 so we avoid values repeating with the defaults in tok_int just above

        # Updating the data and tag dictionaries with the vocabulary integer values
        for input in self.data:
            for key in input:
                input[key] = tok_int[input[key]]

        for index, tag in enumerate(self.tags):
            self.tags[index] = tok_int[tag]

        int_tok = {v: k for k, v in tok_int.items()}

        # Saving the int_tok and tok_int vocabularies so we can decode the data later if we want to
        with open(f'{vocab_path}/vocab_tok-int.json', 'w+') as file:
            json.dump(tok_int, file)

        with open(f'{vocab_path}/vocab_int-tok.json', 'w+') as file:
            json.dump(int_tok, file)

        return tok_int, int_tok, just_words, just_tags, just_POS, just_bool

    def encode_data(self):
        '''
        Function to encode the data as a vector which is the size of the number of features
        - Returns: encoded data (inputs) and encoded term tags (hypothesis outputs)
        '''
        data_encoded = []
        tags_encoded = []

        for input in self.data:
            single_input = []
            for item in input:
                single_input.append(input[item])
            data_encoded.append(torch.tensor(single_input))

        for tag in self.tags:
            tags_encoded.append(torch.tensor(tag))

        return data_encoded, tags_encoded

    def one_hot_encode_data(self):
        '''
        Function to one-hot encode the data
        - Returns: encoded data (inputs) and encoded term tags (hypothesis outputs)
        '''
        data_arrs = []  # List of all of the final arrays

        for input in self.data:
            arrs_list = []  # List of arrays for one document
            for key in input:
                if 'word' in key.lower():
                    arr = torch.zeros(len(self.just_words))
                    arr[self.just_words.index(self.int_tok[input[key]])] = 1
                    arrs_list.append(arr)
                if 'tag' in key.lower():
                    arr = torch.zeros(len(self.just_tags))
                    arr[self.just_tags.index(self.int_tok[input[key]])] = 1
                    arrs_list.append(arr)
                if 'pos' in key.lower():
                    arr = torch.zeros(len(self.just_pos))
                    arr[self.just_pos.index(self.int_tok[input[key]])] = 1
                    arrs_list.append(arr)
                if 'is' in key.lower():
                    arr = torch.zeros(len(self.just_bool))
                    arr[self.just_bool.index(self.int_tok[input[key]])] = 1
                    arrs_list.append(arr)
            # May need to change reshape value if gitlab dataset gets updated with more inputs
            data_arrs.append(torch.hstack(arrs_list).reshape(5, -1))  # Append the arrays from one document to list of arrays for the dataset split & reshape

        # Transform the hypothesis tags into numpy arrays
        tag_arrs = []
        for index, tag in enumerate(self.tags):
            tag_arr = torch.zeros(len(self.just_tags), dtype=float)
            tag_arr[self.just_tags.index(self.int_tok[tag])] = 1
            tag_arrs.append(tag_arr)

        return data_arrs, tag_arrs


split = 'train'
dataset = TermsDataset(directory, split, one_hot=True)
print(f'{len(dataset)} inputs in {split} dataset')
input_tensor, tag_tensor = dataset[0]
print(input_tensor)
print(tag_tensor)

# Testing to make sure data works with DataLoader
'''
text_dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
batch_data, batch_name = next(iter(text_dataloader))
print(batch_data.shape)
print(batch_name.shape)
print(batch_name)
'''