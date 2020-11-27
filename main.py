import pandas as pd  # For data handling
from time import time  # To time our operations
import logging  # Setting up the loggings to monitor gensim
import gensim.downloader as api


def filter_not_in_vocab(in_str):
    if in_str in w2v_model.vocab:
        return True
    else:
        return False

SARCASM_LABEL = 1
SERIOUS_LABEL = 0

# set up mechanism for timing how long the program takes to execute
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)
t = time()

comments_df = pd.read_csv(r'C:\Users\Sam\Desktop\train-balanced-sarcasm-Full.csv')
comment_count = len(comments_df)

sarcastic_count_correct = 0
sarcastic_count_incorrect = 0

serious_count_correct = 0
serious_count_incorrect = 0

err_count = 0

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
#distance = w2v_model.wv.n_similarity(df_a, df_b)
#print(distance)

test_df = pd.read_csv(r'C:\Users\Sam\Desktop\train-balanced-sarcasm-Test.csv')

for name, val in test_df.iterrows():
    a = val.comment
    n = a.split()
    vocab_filter_n = filter(filter_not_in_vocab, n)
    df_n = list(vocab_filter_n)

    try:
        sar_dist = w2v_model.wv.n_similarity(df_n, df_a)
        ser_dist = w2v_model.wv.n_similarity(df_n, df_b)

        # sar_distW = w2v_model.wmdistance(df_n, df_a)
        # ser_distW = w2v_model.wmdistance(df_n, df_b)

        if val.label == SARCASM_LABEL:
            if ser_dist < sar_dist:
                sarcastic_count_incorrect += 1
            else:
                sarcastic_count_correct += 1
        else:  # it's serious
            if ser_dist > sar_dist:
                serious_count_incorrect += 1
            else:
                serious_count_correct += 1

        sarcastic_pct_correct = (sarcastic_count_correct * 100 / (max(sarcastic_count_correct + sarcastic_count_incorrect, 1)))
        serious_pct_correct = (serious_count_correct * 100 / (max(serious_count_correct + serious_count_incorrect, 1)))
        print(f'%NET: {(sarcastic_count_correct+serious_count_correct)*100/(max(sarcastic_count_correct + sarcastic_count_incorrect + serious_count_correct + serious_count_incorrect, 1))}  ||  N Similarity - Sarcastic Correct: {sarcastic_pct_correct}, Serious Correct: {serious_pct_correct}')
        #print(f'%WMD - Sarcastic Correct: {pct_correct}, Serious Correct: {}')
    except:
        err_count += 1
        print(f'# Unable To Calculate: {err_count}')

print('fin')
