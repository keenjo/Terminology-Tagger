import spacy
import numpy as np
import os
import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

# Loading the spacy English mode
nlp = spacy.load('en_core_web_sm')

# Path to the annotated terminology project data
directory = 'terminology-project-2022/'

class TermsDataset(Dataset):

    def __init__(self, directory, split, one_hot=False):
        self.directory = directory # Directory where data is stored
        self.split = split # Dataset split -> train, dev, or test
        self.one_hot = one_hot # Choice of whether or not to do one-hot encoding
        # Uncommented this b/c we only saved the vocab early on to check that it was correct
        #self.vocab_path = vocab_path # Directory where vocab will be saved

        # Preparing/Organizing the data
        self.data, self.tags, self.words_total, self.tags_total, self.POS_total = self.prep_data()
        # Creating the vocabulary
        self.word_int, self.tag_int, self.pos_int, self.bool_int = self.create_vocab()
        # Encoding the data
        if self.one_hot == False:
            '''
            Choice of data encoding:
            - Non one-hot: as N dimensional vector where N is the number of features
            - One-hot: each feature encoded in binary with N dimensions, where N is the number of options for each feature
                - All of the feature vectors are then combined into one vector
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
                    data_paths.append(os.path.join(root, file))

        # Preprocessing loop
        for fpath in data_paths:
            with open(fpath, 'r+', encoding='utf8') as f:
                text = f.readlines()

            words = [] # all of the words for a given document
            doc_tags = [] # all of the term tags for a given document
            POS_list = [] # all of the word POS tags for a given document

            # Splitting the txt file into the word and tag lists
            for line in text:
                if line == '\n':
                    # Add annotation word the new lines as we will add <BOS> and <EOS> tokens for these in our feature dictionaries
                    words.append('#')
                    doc_tags.append('O')
                elif line != '\n' and line != '':
                    # Added try except below to deal with inconsistencies with separators between words and annotations
                    try:
                        word, tag = line.split('\t')
                    except ValueError:
                        word, tag = line.split()
                    tag = ''.join(tag.split())
                    if word != '' and tag != '':
                        words.append(word)
                        # Added these statements below to deal with inconsistencies in annotations
                        if 'B' in tag.upper():
                            doc_tags.append('B')
                        elif 'I' in tag.upper():
                            doc_tags.append('I')
                        elif 'O' in tag.upper() or '0' in tag:
                            doc_tags.append('O')
                    elif word == '' and tag == '':
                        continue
                    # If there is a missing tag for a word we just annotate the 'O' tag
                    elif word != '' and tag == '':
                        words.append(word)
                        doc_tags.append('O')

            # Getting the POS tags of all of the words
            doc = nlp(' '.join(words))
            for index, token in enumerate(doc):
                if str(token) != '#':
                    POS_list.append(token.pos_)
                    # Lemmatization of the words
                    if str(token.lemma_) != words[index].lower():
                        words[index] = str(token.lemma_)
                else:
                    # If a '#' token is found (which will be converted to <BOS> or <EOS> later we assign 'X' as the POS tag
                    POS_list.append('X')

            # Change # to <EOS>
            # Originally used # rather than <EOS> to avoid messing with the tokenization when finding POS just above
            for index, word in enumerate(words):
                if word == '#':
                    words[index] = '<EOS>'
            for word in words:
                words_total.append(word)
            for tag in doc_tags:
                tags_total.append(tag)
            for pos in POS_list:
                POS_total.append(pos)

            # Creating a dictionary of features for each input (ONLY FOR THE CHOSEN DATA SPLIT)
            if self.split in fpath: # Making sure to only organize data for the chosen data split
                for index, word in enumerate(words):
                    if word != '<EOS>':
                        # Inputs dictionary: contains all input information for a given document
                        inputs = {}
                        # Adding all of the information to the inputs dictionary
                        inputs['Main word'] = word
                        inputs['Main POS'] = POS_list[index]
                        inputs['is_first'] = True if index == 0 or words[index - 1] == '<EOS>' else False
                        inputs['is_capitalized_first'] = True if word[0].lower() != word[0] else False
                        inputs['is_capitalized_within'] = True if word[1:].lower() != word [1:] else False
                        inputs['Preceding word'] = '<BOS>' if index == 0 or words[index - 1] == '<EOS>' else words[index - 1]
                        inputs['Preceding POS'] = 'X' if index == 0 or words[index - 1] == '<EOS>' else POS_list[index - 1]
                        inputs['Following word'] = '<EOS>' if index == len(words) - 1 else words[index + 1]
                        inputs['Following POS'] = 'X' if index == len(words) - 1 else POS_list[index + 1]
                        # Appending one input to the data list
                        data_final.append(inputs)
                        tags_final.append(doc_tags[index])

        return data_final, tags_final, words_total, tags_total, POS_total

    def create_vocab(self):
        '''
        Creating a vocabulary for each feature of the data
        - Returns:
            - word_int: dictionary mapping possible words from full annotated dataset to integers
            - tag_int: dictionary mapping possible term tags (I,O,B) from full annotated dataset to integers
            - pos_int: dictionary mapping possible POS tags from feature dictionary to integers
            - bool_int: dictionary mapping possible boolean values from feature dictionary to integers
        '''

        # Used separate lists here just to organize things a bit more
        just_words = sorted(list(set(self.words_total))) # List of all unique words
        just_tags = sorted(list(set(self.tags_total))) # List of all unique term tags
        just_pos = sorted(list(set(self.POS_total))) # List of all unique POS tags
        just_bool = [False, True]

        # Converting the vocabulary to token:integer dictionary
        word_int = {'<UNK>': -10, '<BOS>': 1, '<EOS>': 2}
        count = 0
        for item in just_words:
            if item not in word_int.keys():
                word_int[item] = count + 3 # + 3 so we avoid values repeating with the defaults in tok_int just above
                count += 1

        tag_int = {}
        for index, item in enumerate(just_tags):
            if item not in tag_int.keys():
                tag_int[item] = index

        pos_int = {}
        for index, item in enumerate(just_pos):
            if item not in pos_int.keys():
                pos_int[item] = index

        bool_int = {False: 0, True: 1}

        total_dicts = {'Word dict': word_int,
                       'Tag dict': tag_int,
                       'POS dict': pos_int,
                       'Bool dict': bool_int}
        # Creating an inverse integer to token mapping that could be used for decoding later if we want
        int_tok = {v: k for k, v in word_int.items()}

        # Saving the tok_int vocabularies so we can decode the data later if we want to
        #with open(f'{self.vocab_path}/vocab_tok-int.json', 'w+') as file:
            #json.dump(total_dicts, file, ensure_ascii=False)

        return word_int, tag_int, pos_int, bool_int

    def encode_data(self):
        '''
        Function to encode each input as a tensor
        - Returns: encoded data (inputs) and encoded term tag labels (expected outputs)
        '''
        data_encoded = []

        for input in self.data:
            tensor_list = []
            # For each feature we append an encoded feature to the input vector
            for key, value in input.items():
                if 'word' in key.lower():
                    val = self.word_int[value]
                    tensor_list.append(val)
                elif 'pos' in key.lower():
                    val = self.pos_int[value]
                    tensor_list.append(val)
                elif 'is' in key.lower():
                    val = self.bool_int[value]
                    tensor_list.append(val)
            data_encoded.append(torch.tensor(tensor_list, dtype=torch.float))

        tags_encoded = [torch.tensor(self.tag_int[x]) for x in self.tags]

        return data_encoded, tags_encoded

    def one_hot_encode_data(self):
        '''
        Function to one-hot encode the data
        - Returns: encoded data (inputs) and encoded term tag labels (expected outputs)
        '''
        data_one_hot = []

        for input in self.data:
            tensor_list = []
            # For each feature we append a one-hot encoded feature to the input vector
            for key, value in input.items():
                if 'word' in key.lower():
                    val = torch.nn.functional.one_hot(torch.tensor(self.word_int[value]), num_classes=len(self.word_int))
                    val.type(torch.float)
                    tensor_list.append(val)
                elif 'pos' in key.lower():
                    val = torch.nn.functional.one_hot(torch.tensor(self.pos_int[value]), num_classes=len(self.pos_int))
                    val.type(torch.float)
                    tensor_list.append(val)
                elif 'is' in key.lower():
                    val = torch.nn.functional.one_hot(torch.tensor(self.bool_int[value]), num_classes=len(self.bool_int))
                    val.type(torch.float)
                    tensor_list.append(val)
            data_one_hot.append(torch.cat(tensor_list).type(torch.float))

        tags_one_hot = [torch.tensor(self.tag_int[x]) for x in self.tags]

        return data_one_hot, tags_one_hot
