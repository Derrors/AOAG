#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : BertScore 结果评估
@Author       : Qinghe Li
@Create time  : 2021-02-16 10:07:28
@Last update  : 2021-09-13 09:46:53
"""

import glob
import json
import os

from bert_score import score
import torch.nn.functional as F

from prettytable import PrettyTable


def eval_decode_result(decoded_dir):
    bert_table = PrettyTable()
    bert_table.field_names = ["BertScore-P", "BertScore-R", "BertScore-F1"]

    decoded_filelist = glob.glob(os.path.join(decoded_dir, "*_decoded.json"))

    res_dec = []
    res_ref = []

    for f in decoded_filelist:
        with open(f, "r") as rf:
            item = json.load(rf)
            res_dec.append(item["decoded_answer"])
            res_ref.append(item["answer"])

    bs_p, bs_r, bs_f1 = score(res_dec, [ref[:512] for ref in res_ref], lang='en')
    bert_table.add_row((round(bs_p.mean().item(), 4), round(bs_r.mean().item(), 4), round(bs_f1.mean().item(), 4)))

    print("* Bert Score Evaluate Metric：")
    print(bert_table)
