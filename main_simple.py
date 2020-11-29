import math
import re
from collections import Counter

# WORD = re.compile(r"\w+")

text = "dude get out"


def get_cos_distance(text1, text2):
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    return get_cosine(vector1, vector2)


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text_input):
    words = re.compile(r"\w+").findall(text_input)
    return Counter(words)


def get_avg_score(txt, file_name):
    net_score = 0
    count = 0

    with open(file_name, 'r') as file:
        for line in file:
            sample = line.strip()
            net_score += get_cos_distance(txt, sample)
            count += 1

    return net_score / count


sarc_score = get_avg_score(text, 'sarcasm_sample.txt')
ser_score = get_avg_score(text, 'serious_sample.txt')

if sarc_score < ser_score:
    print('sarcastic')
else:
    print('serious')

