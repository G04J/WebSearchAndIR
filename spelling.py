import sys, os, nltk, re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from collections import defaultdict, Counter
from nltk.corpus import words as wd
from nltk.metrics import edit_distance

class SpellCorrection:
    def __init__(self):
        self.all_wrds = set(wd.words())
        self.alp = 'abcdefghijklmnopqrstuvwxyz'

    def one_edit(self, word):
        
        spl = []
        for i in range(len(word) + 1):
            spl.append((word[:i], word[i:]))
        
        dele = set()
        for left, right in spl:
            if right:
                dele.add(left + right[1:])
        
        rep = set()
        for left, right in spl:
            if right:
                for c in self.alp:
                    rep.add(left + c + right[1:])
        
        ins = set()
        for left, right in spl:
            for c in self.alp:
                ins.add(left + c + right)
        
        return dele or rep or ins

    def two_edit(self, word):
        edits = set()
        for edit_1 in self.one_edit(word):
            for edit_2 in self.one_edit(edit_1):
                edits.add(edit_2)
        return edits

    def val_wds(self, words):
        valid = set()
        for word in words:
            if word in self.all_wrds:
                valid.add(word)
        return valid

    def possible_corrections(self, word):
        poss_words = []

        if word in self.all_wrds:
            poss_words.append((word, 0))
            return poss_words

        ed1 = self.val_wds(self.one_edit(word))
        if ed1:
            for word in ed1:
                poss_words.append((word, 1))

        ed2 = self.val_wds(self.two_edit(word))
        if ed2:
            for word in ed2:
                poss_words.append((word, 2))

        return poss_words
