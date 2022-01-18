import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from nltk import word_tokenize
import pandas as pd


class TextCleaning:
    def basic_clean(self, data):
        self.data = data
        # case folding
        self.data = self.data.lower()
        # remove punctuation
        self.data = re.sub('[^\w\s]', ' ', self.data)
        # remove numbers
        self.data = re.sub('\d+', '', self.data)
        # remove extra whitespace
        self.data = ' '.join(self.data.split())
        # remove emoji
        self.data = re.sub(r'[^\x00-\x7F]+', ' ', self.data)
        # remove new line
        self.data = re.sub('\n', ' ', self.data)
        return self.data

    # Stemming
    def stemmer_sastrawi(self, data):
        self.data = data
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        self.data = stemmer.stem(self.data)
        return self.data

    # # Stopword Removal
    # def stopword_removal(self, data):
    #     self.data = data
    #     factory = StopWordRemoverFactory()
    #     stopword = factory.create_stop_word_remover()
    #     self.data = stopword.remove(self.data)
    #     return self.data

    def formalize_text(self, data):
        self.data = data
        # Kamus alay
        kamus_alay_1 = pd.read_csv(
            "https://raw.githubusercontent.com/ramaprakoso/analisis-sentimen/master/kamus/kbba.txt",
            delimiter="\t",
            header=None,
            names=['slang', 'formal'])
        kamus_alay_2 = pd.read_csv(
            "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv",
            usecols=["slang", "formal"])

        kamus_alay = pd.concat([kamus_alay_1, kamus_alay_2])

        # Dictionary bahasa alay
        dict_alay = dict()
        for index, row in kamus_alay.iterrows():
            dict_alay[row['slang']] = row['formal']

        word_tokens = word_tokenize(self.data)
        result = [dict_alay.get(x, x) for x in word_tokens]
        return ' '.join(result)

    def all_preprocessing(self, data):
        self.data = data
        self.data = self.basic_clean(self.data)
        self.data = self.formalize_text(self.data)
        self.data = self.stemmer_sastrawi(self.data)
        # self.data = self.stopword_removal(self.data)
        return self.data
