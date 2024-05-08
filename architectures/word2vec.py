import gensim.models
import pandas as pd

word2vec_path = 'C:\\Users\\Antek\\PycharmProjects\\ArcheLinguist\\models\\word2vec\\GoogleNews-vectors-negative300.bin.gz'

class SemanticModel:
    def __init__(self, word2vec_path = 'C:\\Users\\Antek\\PycharmProjects\\ArcheLinguist\\models\\word2vec\\GoogleNews-vectors-negative300.bin.gz', word_limit = 10000):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=word_limit)
        self.word_limit = word_limit

    def prepare_data(self, data_save_path = ".\\data\\train.pkl", word_limit = None, word_list = None):
        if word_limit is None:
            word_limit = self.word_limit


        if word_list is None:
            words = self.model.index_to_key[:word_limit]

        else:
            words = word_list

        embeddings = [self.model[w] for w in words]
        df = pd.DataFrame()
        df['embeddings'] = embeddings
        df['words'] = words
        print("Saving the following df: ")
        print(df)
        df.to_pickle(data_save_path)

english_nouns_100 = [
    'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand', 'part', 'child', 'eye',
    'woman', 'place', 'work', 'week', 'case', 'point', 'government', 'company', 'number', 'group', 'problem',
    'fact', 'be', 'have', 'do', 'say', 'get', 'make', 'go', 'know', 'take', 'see', 'come', 'think', 'look',
    'want', 'give', 'use', 'find', 'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'good',
    'new', 'first', 'last', 'long', 'great', 'little', 'own', 'other', 'old', 'right', 'big', 'high', 'different',
    'small', 'large', 'next', 'early', 'young', 'important', 'few', 'public', 'bad', 'same', 'able',
    'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about', 'into', 'over', 'after', 'beneath', 'under',
    'above', 'the', 'that', 'I', 'it', 'not', 'he', 'as', 'you', 'this', 'but', 'his', 'they', 'her',
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'so', 'if', 'time', 'year', 'people',
    'way', 'day', 'man', 'thing', 'woman', 'life', 'child', 'world', 'school', 'state', 'family', 'student', 'group',
    'country', 'problem', 'hand', 'part', 'place', 'case', 'week', 'company', 'system', 'program', 'question',
    'work', 'government', 'number', 'night', 'point', 'home', 'water', 'room', 'mother', 'area', 'money', 'story',
    'fact', 'month', 'lot', 'right', 'study', 'book', 'eye', 'job', 'word', 'business', 'issue', 'side', 'kind',
    'head', 'house', 'service', 'friend', 'father', 'power', 'hour', 'game', 'line', 'end', 'member', 'law',
    'car', 'city', 'community', 'name', 'president', 'team', 'minute', 'idea', 'kid', 'body', 'information',
    'back', 'parent', 'face', 'others', 'level', 'office', 'door', 'health', 'person', 'art', 'war', 'history',
    'party', 'result', 'change', 'morning', 'reason', 'research', 'girl', 'guy', 'moment', 'air', 'teacher',
    'force', 'education'
]

english_nouns_10 = [
    'time', 'person', 'year', 'way', 'day', 'thing', 'man', 'world', 'life', 'hand'
]

# model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=10000)

# print(model.index_to_key[:100])
# print(model['ocean'])
# print(model['sea'])
# print(model.most_similar(positive='ocean'))
# print(model.similarity('ocean','sea'))
# print(model.similarity('ocean','dad'))

if __name__ == '__main__':
    sem_model = SemanticModel(word2vec_path, word_limit=10000)
    sem_model.prepare_data(data_save_path = ".\\data\\train10.pkl", word_list=english_nouns_10)