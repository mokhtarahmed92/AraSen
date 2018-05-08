import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem.arlstem import ARLSTem

positive_lex_file_path = '../data/pos_lex.txt'
negative_lex_file_path = '../data/neg_lex.txt'
stop_words_file_path = '../data/arabicStops.txt'

link_word = "لينكبرايف"
mention_word = "كلمهمنشن"
hashtag_word = "كلمهشتاج"
positive_word = "كلمهموجب"
negative_word = "كلمهسالب"

preserved_words = [link_word, mention_word, hashtag_word, positive_word, negative_word]


def remove_elongation(text):
    a = re.sub(r"ا+", "ا", text)
    b = re.sub(r"ب+", "ب", a)
    c = re.sub(r"ت+", "ت", b)
    d = re.sub(r"ث+", "ث", c)
    e = re.sub(r"ج+", "ج", d)
    f = re.sub(r"ح+", "ح", e)
    g = re.sub(r"خ+", "خ", f)
    q = re.sub(r"د+", "د", g)
    w = re.sub(r"ذ+", "ذ", q)
    r = re.sub(r"ش+", "ش", w)
    t = re.sub(r"س+", "س", r)
    y = re.sub(r"ط+", "ط", t)
    u = re.sub(r"ع+", "ع", y)
    i = re.sub(r"غ+", "غ", u)
    o = re.sub(r"م+", "م", i)
    p = re.sub(r"ه+", "ه", o)
    s = re.sub(r"و+", "و", p)
    j = re.sub(r"ي+", "ي", s)
    k = re.sub(r"ى+", "ى", j)
    l = re.sub(r"ر+", "ر", k)
    x = re.sub(r"ز+", "ز", l)
    x1 = re.sub(r"ن+", "ن", x)
    x2 = re.sub(r"ف+", "ف", x1)
    x3 = re.sub(r"ئ+", "ئ", x2)
    return x3


def normalization(text):
    a = re.sub(r"أ", "ا", text)
    b = re.sub(r"اْ", "ا", a)
    s = re.sub(r"آ", "ا", b)
    return s


def remove_number(text):
    a = re.sub(r"\d", "", text)
    return a


def remove_punctuations(text):
    a = re.sub(r"[!_|<>;:۔.``..$?]", "", text)
    return a


def read_stop_words_from_file(file_path, encoding='utf-8'):
    stop_words_file = open(file_path, encoding=encoding)
    arabic_stop_words = stop_words_file.read()
    arabic_stop_words = [w.strip() for w in arabic_stop_words.split('\n')]
    return list(set(arabic_stop_words))


def read_pos_lex(file_path, encoding='utf-8'):
    pos_lex_file = open(file_path, encoding=encoding)
    pos_lex = pos_lex_file.read()
    pos_lex = [w.strip() for w in pos_lex.split('\n')]
    return list(set(pos_lex))


def read_neg_lex(file_path, encoding='utf-8'):
    neg_lex_file = open(file_path, encoding=encoding)
    neg_lex = neg_lex_file.read()
    neg_lex = [w.strip() for w in neg_lex.split('\n')]
    return list(set(neg_lex))


def match_preserved_word(word):
    for w in preserved_words:
        if w in word:
            return True
    return False


def custom_stop_words(file_path):
    custom_stop_words = read_stop_words_from_file(file_path)
    stop_list1 = list(set(stopwords.words("Arabic")))
    all_stop_words = list(set(stop_list1 + custom_stop_words))
    return all_stop_words


def clean_str(text):
    search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
              '&quot;', '?', '؟', '!']
    replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ', ' ؟ ',
               ' ! ']

    # remove tashkeel
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel, "", text)

    # remove longation
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)

    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')

    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    # trim
    #text = text.strip()

    return text


def encode_class_labels(label):
    if label == 'positive':
        return [1, 0]
    else:
        return [0, 1]


def calculate_accuracy(predictions, test_labels):
    correct = 0
    total = len(predictions)
    for xi, x in enumerate(predictions):
        if x == test_labels[xi]:
            correct += 1
    return (correct/total)


def clean_sentence(text,
                   positive_lex=read_pos_lex(positive_lex_file_path),
                   negative_lex=read_neg_lex(negative_lex_file_path),
                   stemmer=ARLSTem(),
                   stopwords=custom_stop_words(stop_words_file_path)):
    text = clean_str(text)

    cleaned_sentence = re.sub(r"http\S+", link_word, text)  # link remove

    cleaned_sentence = re.sub(r"@\S+", mention_word, cleaned_sentence)  # mention_replace

    cleaned_sentence = re.sub(r"#\S+", hashtag_word, cleaned_sentence)  # hastag_replacer

    cleaned_sentence = re.sub(r"[A-Z]", "", cleaned_sentence)  # remove_capital

    cleaned_sentence = re.sub(r"[a-z]", "", cleaned_sentence)  # remove_small

    stemmed_sentence = stemmer.stem(cleaned_sentence)  # stemmer

    final = stemmer.norm(remove_number(remove_punctuations(normalization(
        remove_elongation(stemmed_sentence)))))  # removeelnogation #removenuber #normalize_word #remove punc

    final = word_tokenize(final)  # word_tokenize

    output = [w for w in final if not w in stopwords]

    pos_postfix = [positive_word for w in output if w in positive_lex]
    neg_postfix = [negative_word for w in output if w in negative_lex]

    # pos_words = [w for w in output if w in positive_lex]
    # neg_words = [w for w in output if w in negative_lex]
    # print(pos_words)
    # print(neg_words)

    output += pos_postfix
    output += neg_postfix

    textOnly = [w for w in output if match_preserved_word(w) == False]

    return ' '.join(output), ' '.join(textOnly)


def featurize(text, tfidf_model, feature_len=5):
    text = str(text)

    perc_link = (len(re.findall(r"لينكبرايف", text))) / feature_len
    perc_mention = (len(re.findall(r"كلمهمنشن", text))) / feature_len
    perc_hash = (len(re.findall(r"كلمهشتاج", text))) / feature_len
    perc_neg = (len(re.findall(r"كلمهسالب", text))) / feature_len
    perc_pos = (len(re.findall(r"كلمهموجب", text))) / feature_len

    if perc_link >= 1: perc_link = 0.99
    if perc_mention >= 1: perc_mention = 0.99
    if perc_hash >= 1: perc_hash = 0.99
    if perc_neg >= 1: perc_neg = 0.99
    if perc_pos >= 1: perc_pos = 0.99

    lenght = len(text)
    if lenght <= 10:
        rs = [1, 0, 0]
    elif lenght <= 20:
        rs = [0, 1, 0]
    else:
        rs = [0, 0, 1]

    features_vec = [perc_link, perc_mention, perc_hash, perc_neg, perc_pos]
    features_vec += rs

    tfidf_feature = tfidf_model.transform(np.array([text])).toarray()
    all_features = features_vec + list(tfidf_feature[0])
    #return features_vec
    return all_features
