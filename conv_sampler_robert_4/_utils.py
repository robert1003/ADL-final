import string
import random
import time
import math

def _asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, epoch, epochs):
    now = time.time()
    s = now - since
    rs = s / epoch * (epochs - epoch)
    return '%s (- %s)' % (_asMinutes(s), _asMinutes(rs))

def gen_name(stringLength=8):
    lettersAndDigits = string.ascii_letters + string.digits
    return ''.join((random.choice(lettersAndDigits) for i in range(stringLength)))

import csv
import argparse
import unicodedata
import re


def normalize_tag(tag):
    tag = unicodedata.normalize("NFKC", re.sub('＊|\*|\s+', '', tag))
    return tag

def score(ref_file, pred_file):
    with open(ref_file) as csvfile:
        reader = csv.DictReader(csvfile)
        ref_data = list(reader)

    with open(pred_file) as csvfile:
        reader = csv.DictReader(csvfile)
        pred_data = list(reader)

    f_score = 0.0
    for ref_row, pred_row in zip(ref_data, pred_data):
        refs = set(ref_row["Prediction"].split())
        preds = set(pred_row["Prediction"].split())

        p = len(refs.intersection(preds)) / len(preds) if len(preds) > 0 else 0.0
        r = len(refs.intersection(preds)) / len(refs) if len(refs) > 0 else 0.0
        f = 2*p*r / (p+r) if p + r > 0 else 0
        f_score += f

    return f_score / len(ref_data)
