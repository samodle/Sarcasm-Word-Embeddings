import pandas as pd  # For data handling
from time import time  # To time our operations
import logging  # Setting up the loggings to monitor gensim
import gensim.downloader as api


def filter_not_in_vocab(in_str):
    if in_str in w2v_model.vocab:
        return True
    else:
        return False


# set up mechanism for timing how long the program takes to execute
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
t = time()

comments_df = pd.read_csv(r'C:\Users\Sam\Desktop\train-balanced-sarcasm.csv')
comment_count = len(comments_df)

count_correct = 0
count_incorrect = 0
count_net = 0

front_to_back = True
temp = True

for label, df_type in comments_df.groupby('label'):
    print(df_type)
    if temp:
        serious_df = df_type
        temp = False
    else:
        sarcastic_df = df_type

del serious_df['label']
del sarcastic_df['label']
print('Setup Complete: {} min'.format(round((time() - t) / 60, 2)))

df_a = serious_df.comment.str.cat().split()
df_b = sarcastic_df.comment.str.cat().split()

# Load our Word2Vec Model
w2v_model = api.load("word2vec-google-news-300")
# w2v_model = api.load("glove-wiki-gigaword-100")

# filter words not in the vocabulary
vocab_filter_b = filter(filter_not_in_vocab, df_b)
df_b = list(vocab_filter_b)

vocab_filter_a = filter(filter_not_in_vocab, df_a)
df_a = list(vocab_filter_a)

# calculate distance
distance = w2v_model.wv.n_similarity(df_a, df_b)
print(distance)

for name, val in sarcastic_df.iterrows():
    a = val.comment
    n = a.split()
    vocab_filter_n = filter(filter_not_in_vocab, n)
    df_n = list(vocab_filter_n)

    sar_dist = w2v_model.wv.n_similarity(df_n, df_a)
    ser_dist = w2v_model.wv.n_similarity(df_n, df_b)

    count_net += 1
    if ser_dist < sar_dist:
        count_incorrect += 1
    else:
        count_correct += 1

    pct_correct = (count_correct * 100 / count_net)
    print(f'% Correct: {pct_correct}')

print('fin')
