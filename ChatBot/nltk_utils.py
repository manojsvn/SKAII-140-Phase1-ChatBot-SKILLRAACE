import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenize_text(text):
    return nltk.word_tokenize(text)

def stem_word(word):
    return porter.stem(word.lower())

def generate_bag_of_words(tokenized_sentence, vocabulary):
    stemmed_words = [stem_word(word) for word in tokenized_sentence]
    bag_of_words = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        if word in stemmed_words:
            bag_of_words[idx] = 1
    return bag_of_words
