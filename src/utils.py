#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : ROUGE 计算、结果处理
@Author       : Qinghe Li
@Create time  : 2021-02-16 10:07:28
@Last update  : 2021-09-13 09:45:29
"""

import glob
import json
import logging
import os

import config
import numpy as np
import pyrouge
import torch
import torch.nn.functional as F
from distinct_n import distinct_n_corpus_level
from nltk.translate.bleu_score import corpus_bleu
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from PRENR import call_fn


def cosine(x, y):
    # x(b, n, d) y(b, m, d)
    b, n, d = x.size()
    m = y.size(1)

    C = []
    for i in range(m):
        C.append(F.cosine_similarity(x, y[:, i, :].unsqueeze(1).expand(b, n, d), dim=2).view(b, n, 1))      # (b, n, 1)
    C = torch.cat(C, dim=2)         # (b, n, m)
    return C


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def print_results(question, answer, decoded_output):
    print("")
    print("Question:  %s", question)
    print("Reference answer: %s", answer)
    print("Generated answer: %s", decoded_output)
    print("")


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()

    r.model_filename_pattern = "#ID#_reference.txt"
    r.system_filename_pattern = r"(\d+)_decoded.txt"

    r.model_dir = ref_dir
    r.system_dir = dec_dir

    logging.getLogger("global").setLevel(logging.WARNING)
    rouge_results = r.convert_and_evaluate()
    rouge_dict = r.output_to_dict(rouge_results)

    result_dict = {}
    for x in ["1", "l"]:
        key = "rouge_%s_f_score" % (x)
        val = rouge_dict[key]
        result_dict[key] = val

    rouge_1 = result_dict["rouge_1_f_score"]
    rouge_l = result_dict["rouge_l_f_score"]

    return rouge_1, rouge_l


def eval_decode_result(ref_dir, dec_dir, decoded_dir):
    gene_table = PrettyTable()
    opin_table = PrettyTable()
    pare_table = PrettyTable()
    gene_table.field_names = ["Rouge-1", "Rouge-L", "Distinct-1", "Distinct-2"]
    opin_table.field_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    pare_table.field_names = ["PRENR-P", "PRENR-R", "PRENR-F"]

    decoded_filelist = glob.glob(os.path.join(decoded_dir, "*_decoded.json"))

    res_dec = []
    res_ref = []
    res_reviews = []
    res_pred = []
    res_true = []

    for f in decoded_filelist:
        with open(f, "r") as rf:
            item = json.load(rf)
            res_dec.append(item["decoded_answer"])
            res_ref.append(item["answer"])
            res_reviews.append([r.split() for r in item["all_reviews"]])
            res_pred.extend(item["predict_opinions"])
            res_true.extend(item["ground_opinions"])

    rouge_1, rouge_l = rouge_eval(ref_dir, dec_dir)
    distinct_1 = distinct_n_corpus_level(res_dec, 1)
    distinct_2 = distinct_n_corpus_level(res_dec, 2)
    gene_table.add_row((round(rouge_1, 4), round(rouge_l, 4), round(distinct_1, 4), round(distinct_2, 4)))

    acc = accuracy_score(res_true, np.argmax(res_pred, axis=1))
    pre = precision_score(res_true, np.argmax(res_pred, axis=1), average='macro')
    rec = recall_score(res_true, np.argmax(res_pred, axis=1), average='macro')
    f1 = f1_score(res_true, np.argmax(res_pred, axis=1), average='macro')
    opin_table.add_row((round(acc, 4), round(pre, 4), round(rec, 4), round(f1, 4)))

    pt_p, pt_r, pt_f1 = call_fn(res_dec, res_reviews)
    pare_table.add_row((round(pt_p, 4), round(pt_r, 4), round(pt_f1, 4)))

    print("* " * 30)
    print("* Generator Evaluate Metric：")
    print(gene_table)
    print("* Opinion Classification Evaluate Metric：")
    print(opin_table)
    print("* PRENT Evaluate Metric：")
    print(pare_table)
    print("* " * 30)


def write_dec_ref(reference_sent, decoded_words, ex_index, ref_dir, dec_dir):
    decoded_sent = " ".join(decoded_words).strip()
    decoded_sent = make_html_safe(decoded_sent)
    reference_sent = make_html_safe(reference_sent)

    ref_file = os.path.join(ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as fr:
        fr.write(reference_sent)
    with open(decoded_file, "w") as fd:
        fd.write(decoded_sent)


def write_for_eval(questions, answers, decoded_results, reviews, selected_reviews, pred_o, true_o, decode_dir):
    question_sents = [make_html_safe(question) for question in questions]
    answer_sents = [make_html_safe(answer) for answer in answers]
    decoded_sents = [make_html_safe(" ".join(decoded_words).strip()) for decoded_words in decoded_results]

    file_num = len(questions)
    for i in range(file_num):
        item = {
            "question": question_sents[i],
            "answer": answer_sents[i],
            "decoded_answer": decoded_sents[i],
            "all_reviews": reviews[i],
            "selected_reviews": selected_reviews[i],
            "predict_opinions": pred_o[i],
            "ground_opinions": true_o[i]
        }

        decoded_file = os.path.join(decode_dir, "%06d_decoded.json" % i)

        with open(decoded_file, "w") as fd:
            json.dump(item, fd)
