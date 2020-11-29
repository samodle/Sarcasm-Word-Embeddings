import pandas as pd  # For data handling
from time import time  # To time our operations
import logging  # Setting up the loggings to monitor gensim
import gensim.downloader as api


def filter_not_in_vocab(in_str):
    if in_str in w2v_model.vocab:
        return True
    else:
        return False


def percentage(part, whole):
    if whole == 0:
        whole = 1
    return round(100 * part / whole, 1)


def average(net, count):
    if count == 0:
        count = 1
    return round(net / count, 5)


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

correct_distance_net = 0
incorrect_distance_net = 0

correct_lte_4 = 0
incorrect_lte_4 = 0
correct_lte_3 = 0
incorrect_lte_3 = 0
correct_lte_pointtwo = 0
incorrect_lte_pointtwo = 0
correct_lte_pointone = 0
incorrect_lte_pointone = 0

err_count = 0
net_count = 0

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

test_df = pd.read_csv(r'C:\Users\Sam\Desktop\train-balanced-sarcasm-Test.csv')
test_df = test_df.sample(frac=1).reset_index(drop=True)

for name, val in test_df.iterrows():
    try:
        a = val.comment
        n = a.split()
        vocab_filter_n = filter(filter_not_in_vocab, n)
        df_n = list(vocab_filter_n)

        sar_dist = w2v_model.wv.n_similarity(df_n, df_a)
        ser_dist = w2v_model.wv.n_similarity(df_n, df_b)

        difference = abs(ser_dist - sar_dist)

        if val.label == SARCASM_LABEL:
            if ser_dist < sar_dist:
                sarcastic_count_incorrect += 1
                incorrect_distance_net += difference
                if difference <= 0.001:
                    incorrect_lte_pointone += 1
                elif difference <= 0.002:
                    incorrect_lte_pointtwo += 1
                elif difference <= 0.003:
                    incorrect_lte_3 += 1
                elif difference <= 0.004:
                    incorrect_lte_4 += 1
            else:
                sarcastic_count_correct += 1
                correct_distance_net += difference
                if difference <= 0.001:
                    correct_lte_pointone += 1
                elif difference <= 0.002:
                    correct_lte_pointtwo += 1
                elif difference <= 0.003:
                    correct_lte_3 += 1
                elif difference <= 0.004:
                    correct_lte_4 += 1
        else:  # it's serious
            if ser_dist > sar_dist:
                serious_count_incorrect += 1
                incorrect_distance_net += difference
                if difference <= 0.001:
                    incorrect_lte_pointone += 1
                elif difference <= 0.002:
                    incorrect_lte_pointtwo += 1
                elif difference <= 0.003:
                    incorrect_lte_3 += 1
                elif difference <= 0.004:
                    incorrect_lte_4 += 1
            else:
                serious_count_correct += 1
                correct_distance_net += difference
                if difference <= 0.001:
                    correct_lte_pointone += 1
                elif difference <= 0.002:
                    correct_lte_pointtwo += 1
                elif difference <= 0.003:
                    correct_lte_3 += 1
                elif difference <= 0.004:
                    correct_lte_4 += 1

        net_count += 1

        sarcastic_pct_correct = percentage(sarcastic_count_correct, sarcastic_count_correct + sarcastic_count_incorrect)
        serious_pct_correct = percentage(serious_count_correct, serious_count_correct + serious_count_incorrect)
        print(
            f'%NET: {percentage(sarcastic_count_correct + serious_count_correct, net_count)}%.  Count: {net_count + err_count} in {format(round((time() - t) / 60, 2))}  ||  N Similarity - Sarcastic Correct: {sarcastic_pct_correct}%, Serious Correct: {serious_pct_correct}%')
        print(f'     Avg Correct Dist: {average(correct_distance_net, serious_count_correct + sarcastic_count_correct)}, Avg Incorrect Dist: {average(incorrect_distance_net, serious_count_incorrect + sarcastic_count_incorrect)}.  Incorrect/Correct < 0.001: {incorrect_lte_pointone}/{correct_lte_pointone}, 0.001-0.002: {incorrect_lte_pointtwo}/{correct_lte_pointtwo}, 0.002-0.003: {incorrect_lte_3}/{correct_lte_3}, 0.003-0.004: {incorrect_lte_4}/{correct_lte_4}')
    except:
        err_count += 1
        print(f'# Unable To Calculate: {err_count}')

print('fin')
