import sys, os, nltk, re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import wordnet
from collections import defaultdict, Counter
from nltk.corpus import words as wd
from nltk.metrics import edit_distance
from spelling import SpellCorrection

class SearchQuery:
    def __init__(self, ind_fol):
        self.ind_fol = ind_fol
        self.ind_path = os.path.join(self.ind_fol, 'index.txt')
        self.invert_index = self.get_inverted_index()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.spell_checker = SpellCorrection()
        self.min_pos = {}

    def get_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        pos_map = {
            'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.VERB,
        }
        return pos_map.get(tag, wordnet.NOUN)

    def min_dis_wds(self, word1, word2, doc_id):
        min_dis = float('inf')
        corr_ord = False
        
        min_pos1 = None
        min_pos2 = None

        pos_wd1 = self.invert_index[word1][doc_id]
        pos_wd2 = self.invert_index[word2][doc_id]

        for _, line_num1, pos1 in pos_wd1:
            for _, line_num2, pos2 in pos_wd2:

                curr_dis = abs((line_num2 - line_num1) * 1000 + (pos2 - pos1)) - 1

                if curr_dis < min_dis:
                    min_dis = curr_dis

                    if line_num1 < line_num2:
                        corr_ord = True 
                    elif line_num1 == line_num2 and pos1 < pos2:
                        corr_ord = True 

                    min_pos1 = (line_num1, pos1)
                    min_pos2 = (line_num2, pos2)

        self.min_pos[(word1, doc_id)] = min_pos1
        self.min_pos[(word2, doc_id)] = min_pos2

        return min_dis, corr_ord

    def t_min_prox_dist(self, doc_id, words):

        t_dis = 0
        corr_ord = 0

        for i in range(len(words) - 1):
            min_dis, corr_ord = self.min_dis_wds(words[i], words[i + 1], doc_id)
            t_dis += min_dis
            if corr_ord:
                corr_ord += 1

        return t_dis, corr_ord

    def get_inverted_index(self):
        invert_index = defaultdict(lambda: defaultdict(list))

        with open(self.ind_path, 'r') as file:
            for line in file:
                words = line.strip().split()
                word, pos = words[0].split(',')
                word_key = (word, pos)

                for position in words[1:]:
                    doc_id, line_num, pos_num = (int(val) for val in position.split(':'))
                    invert_index[word_key][doc_id].append((doc_id, line_num, pos_num))

        return invert_index

    def process_words(self, words):

        document = ' '.join(words)
        document = re.sub(r'(\d+)?\.\d+', '', document)  
        document = re.sub(r'(\d+),(\d+)', r'\1\2', document) 
        document = re.sub('[.,]', '', document)  
        document = re.sub('[^a-zA-Z0-9]', ' ', document)
        pattern = re.compile(r'\b(?:[a-z]\.){2,}', re.I)
        document = pattern.sub(lambda m: m.group().replace('.', ''), document)

        words = nltk.word_tokenize(document)

        p_wd = []
        for word in words:
            tag = self.get_pos(word)
            lemmatized_word = self.lemmatizer.lemmatize(word.lower(), pos=tag)
            stemmed_word = self.stemmer.stem(lemmatized_word)
            p_wd.append((stemmed_word, tag))

        return p_wd

    def find_common_doc_id(self, words):
        if len(words) == 0:
            return set()

        common_doc_ids = set(self.invert_index.get(words[0], {}).keys())

        for word in words[1:]:
            if word not in self.invert_index:
                return set()
            common_doc_ids &= set(self.invert_index[word].keys())

        return common_doc_ids

    def print_min_dis_lines(self, doc_id, words):

        path_file = os.path.join(self.ind_fol, 'path.txt')
        with open(path_file, 'r') as f:
            doc_fol = f.readline().strip()
        doc_path = os.path.join(doc_fol, f"{doc_id}")


        line_numbers = set()

        for word in words:
            if (word, doc_id) in self.min_pos:
                line_num, _ = self.min_pos[(word, doc_id)]
                line_numbers.add(line_num)

        line_numbers = sorted(line_numbers)

        with open(doc_path) as f:
            content = f.readlines()
            for line_num in line_numbers:
                print(content[line_num - 1].rstrip('\n'))

    def search(self, query):
        og_words = query.strip().split()

        is_line = False
        if len(og_words) > 0:
            if og_words[0] == '>':
                is_line = True
                og_words = og_words[1:]

        words = self.process_words(og_words)
        common_docs = self.find_common_doc_id(words)

        if common_docs:
            min_dist = {doc_id: self.t_min_prox_dist(doc_id, words) for doc_id in common_docs}
            sorted_common_docs = sorted(common_docs, key=lambda x: (min_dist[x][0], -min_dist[x][1], x))

            if is_line:
                for doc_id in sorted_common_docs:
                    print(f"> {doc_id}")
                    if len(words) == 1:
                        self.print_first_line(doc_id, words[0][0])
                    else:
                        self.print_min_dis_lines(doc_id, words)
            else:
                for doc_id in sorted_common_docs:
                    print(doc_id)
        else:
            w_can = self.poss_word_candidates(og_words)
            q_can = self.poss_query_candidates(w_can)

            if q_can:
                self.search_candidate_query(q_can, is_line)
            else:
                print("Not found")

    def poss_word_candidates(self, og_words):
        w_can = []
        for word in og_words:
            corr_w = self.spell_checker.possible_corrections(word.lower())
            corr_w = [c for c in corr_w if c[1] <= 2]
            w_can.append(corr_w)
        return w_can

    def poss_query_candidates(self, w_can):
        q_can = []
        t_min_ed = float('inf')

        def gen_query(index, curr_w, t_ed):
            nonlocal t_min_ed, q_can
            if t_ed > 2:
                return
            if index == len(w_can):
                if t_ed <= t_min_ed:
                    processed_words = self.process_words(curr_w)
                    common_docs = self.find_common_doc_id(processed_words)
                    if common_docs:
                        if t_ed < t_min_ed:
                            t_min_ed = t_ed
                            q_can = [(curr_w.copy(), t_ed, common_docs)]
                        elif t_ed == t_min_ed:
                            q_can.append((curr_w.copy(), t_ed, common_docs))
                return
            for correction in w_can[index]:
                curr_w.append(correction[0])
                gen_query(index + 1, curr_w, t_ed + correction[1])
                curr_w.pop()

        gen_query(0, [], 0)
        return q_can

    def search_candidate_query(self, q_can, is_line):
        
        for corr_query in q_can:
            corr_w, _, common_docs = corr_query
            words = self.process_words(corr_w)
            min_dist = {doc_id: self.t_min_prox_dist(doc_id, words) for doc_id in common_docs}
            sorted_common_docs = sorted(common_docs, key=lambda x: (min_dist[x][0], -min_dist[x][1], x))

            if is_line:
                for doc_id in sorted_common_docs:
                    print(f"> {doc_id}")
                    if len(words) == 1:
                        self.print_first_line(doc_id, words[0][0])
                    else:
                        self.print_min_dis_lines(doc_id, words)
            else:
                for doc_id in sorted_common_docs:
                    print(doc_id)

    def print_first_line(self, doc_id, word):
        path_file = os.path.join(self.ind_fol, 'path.txt')

        with open(path_file, 'r') as f:
            doc_fol = f.readline().strip()

        doc_path = os.path.join(doc_fol, f"{doc_id}")

        with open(doc_path) as f:
            lines = f.readlines()
            for line in lines:
                if word.lower() in line.lower():
                    print(line)
                    return 
        return 

if __name__ == '__main__':
    
    if len(sys.argv) != 2:
        print("Usage: python search.py <folder-of-indexes>")
        exit(1)

    find_query = SearchQuery(sys.argv[1])

    for query in sys.stdin:
        results = find_query.search(query)
