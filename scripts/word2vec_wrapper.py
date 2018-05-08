from gensim.models import Word2Vec
import numpy as np
import editdistance


class Word2VecWrapper(object):

    def __init__(self):
        self.model = None
        self.vocab = []
        self.embeddings_size = 300
        self.OVV = 'OOV'
        self.oov_embeddings = np.zeros(self.embeddings_size)
        self.oov_dict_lookup = {}

    def build(self, sentences):
        self.model = Word2Vec(sentences, size=self.embeddings_size, min_count=1, iter=1000, window=5)

    def build(self, file_path):
        self.model = Word2Vec.load(file_path)
        self.vocab = self.build_vocab()

    def build_vocab(self):
        all_words = []
        for word in self.model.wv.vocab:
            all_words.append(word)
        return all_words

    def build_oov_lookup(self, words):
        self.oov_dict_lookup = dict([(word, self.levenshtein_me(word)) for word in words])

    def set_oov_dict_lookup(self, dict):
        self.oov_dict_lookup = dict

    def get_oov_dict_lookup(self):
        return self.oov_dict_lookup

    def get_oov_mapping(self, word):
        return self.oov_dict_lookup.get(word)


    def levenshtein_me(self, word):
        min_ind = np.argmin(np.array([editdistance.eval(word, w) for w in self.vocab]))
        if min_ind > 0elf.model.wv[new_word]
            else:
                return self. & min_ind < len(self.vocab):
            return self.vocab[min_ind]
        else:
            return self.OVV
Process finished with exit code 0

    def get_word_embeddings(self, word):
        if word in self.model.wv.vocab:
            return self.model.wv[word]
        else:
            #new_word = self.levenshtein_me(word)
            new_word = self.oov_dict_lookup.get(word)
            if new_word != self.OVV and new_word is not None:
                return soov_embeddings

    '''
    def get_sentence_embeddings(self, sentence):
        embeddings = np.zeros(300)
        for word in sentence:
            embeds = self.get_word_embeddings(word)
            embeddings = np.vstack((embeddings, embeds))
        return embeddings[1:]
    '''

    def get_sentence_embeddings(self, sentence, max_sentence_length=0):
        sentence = sentence.split(' ')
        if max_sentence_length == 0:
            max_sentence_length = len(sentence)
        if len(sentence) >= max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            for i in range(max_sentence_length - len(sentence)):
                sentence.append("")
        embeddings = np.zeros(300)
        for word in sentence:
            embeds = self.get_word_embeddings(word)
            embeddings = np.vstack((embeddings, embeds))
        return embeddings[1:]

    def get_embeddings(self, sentence, max_sentence_length):

        text = sentence.split(" ")
        if len(text) > max_sentence_length:
            text = text[:max_sentence_length]
        else:
            for i in range(max_sentence_length - len(text)):
                text.append("")

        embeddings = np.zeros(300)
        oov_embeddings = np.zeros(300)
        for word in text:
            if word in self.model.wv.vocab:
                tmp = self.model.wv[word]
                embeddings = np.vstack((embeddings, tmp))
            else:
                embeddings = np.vstack((embeddings, oov_embeddings))
        return embeddings[1:]

    def get_embeddings(self, sentence):
        text = sentence.split(" ")
        embeddings = np.zeros(300)
        oov_embeddings = np.zeros(300)
        for word in text:
            if word in self.model.wv.vocab:
                tmp = self.model.wv[word]
                embeddings = np.vstack((embeddings, tmp))
            else:
                embeddings = np.vstack((embeddings, oov_embeddings))
        return embeddings[1:]

    def get_model(self):
        return self.model

    def get_most_similar_word_to(self, word):
        return self.model.most_similar(word)

    def get_vocab(self):
        return self.vocab

    def is_exist(self, word):
        for word_model in self.vocab:
            if word_model == word:
                return True
            else:
                return False