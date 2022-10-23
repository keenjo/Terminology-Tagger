import spacy

nlp = spacy.load('en_core_web_sm')

# * Update later to go through all of the *.final files in the dataset repo

fpath = 'Insert path of file you want to preprocess'

with open(fpath, 'r+') as f:
    text = f.readlines()

words = [] # all of the words
labels = [] # all of the labels
POS_list = [] # all of the word POS tags

# Splitting the txt file into the word and label lists
for line in text:
    if line != '\n':
        word, label = line.split('\t')
        words.append(word)
        labels.append(label.strip('\n'))

# Getting the POS tags of all of the words
doc = nlp(' '.join(words))
for token in doc:
    POS_list.append(token.pos_)

# Organizing each input word into a dictionary with several features
inputs = [] # List containing all of the input words along with all of their features

for index, word in enumerate(words):
    inp = {}
    inp['Word'] = word
    inp['Word POS'] = POS_list[index]
    inp['Preceding'] = '<bos' if index == 0 else words[index - 1]
    inp['Preceding POS'] = '---' if index == 0 else POS_list[index - 1]
    inp['Following'] = '<eos>' if index == len(words) - 1 else words[index + 1]
    inp['Following POS'] = '---' if index == len(words) - 1 else POS_list[index + 1]

    inputs.append(inp)
