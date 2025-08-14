import os, sys, nltk, re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from collections import defaultdict
from pathlib import Path
from datetime import datetime
startTime = datetime.now()

class InvertedIndexBuilder():

    def __init__(self, doc_fol, index_fol):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.index = defaultdict(list)
        self.doc_fol = doc_fol
        self.index_fol = index_fol

    def get_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        pos_map = {
            'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.VERB,
        }
        return pos_map.get(tag, wordnet.NOUN)

    def clean_document(self, document):

        document = re.sub(r'(\d+)?\.\d+', '', document)
        document = re.sub(r'(\d+),(\d+)', r'\1\2', document)

        document = re.sub('[.,]', '', document)
        document = re.sub('[^a-zA-Z0-9]', ' ', document)
        pattern = re.compile(r'\b(?:[a-z]\.){2,}', re.I)
        document = pattern.sub(lambda m: m.group().replace('.', ''), document)

        return document
       

    def clean_words(self, document):
        words = word_tokenize(document)
        
        lemmas = []
        stemmed = []
        tags = []

        for word in words: 
            tag = self.get_pos(word)
            lemmas.append(self.lemmatizer.lemmatize(word, self.get_pos(word)))
            stem = (self.stemmer.stem(word))
            stemmed.append((stem, tag))

        return stemmed

    def create_index(self):
        """Build inverted index from documents in the specified folder"""
        for filename in sorted(os.listdir(self.doc_fol)):
            doc_id = filename
            pos = 0
            with open(os.path.join(self.doc_fol, filename), 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    line = self.clean_document(line)
                    words = self.clean_words(line)

                    for word in words:
                        self.index[word].append(f"{doc_id}:{line_num}:{pos}")
                        pos += 1

        self.write_index_to_file()
            
    def write_index_to_file(self):
        
        file_path = os.path.join(self.index_fol, 'index.txt')
        with open(file_path, 'w') as f:
        
            for word in self.index:
                term = word[0]
                tag = word[1]
                f.write(f"{term},{tag}")
                for pos in self.index[word]:
                    f.write(f" {pos}")
                f.write("\n")
        
        with open(os.path.join(self.index_fol, 'path.txt'), 'w') as f:
            f.write(f"{self.doc_fol}")
        print("Indexing complete")
        print(datetime.now() - startTime)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python index.py <fol-of-documents> <fol-of-indexes>")
        exit(1)
        
    doc_fol = sys.argv[1]
    index_fol = sys.argv[2]  
    Path(index_fol).mkdir(parents=True, exist_ok=True)

    builder = InvertedIndexBuilder(doc_fol, index_fol)
    builder.create_index()
