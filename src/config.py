#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : Experimental Settings
@Author       : Qinghe Li
@Create time  : 2021-02-22 17:07:11
@Last update  : 2021-09-13 09:38:21
"""

import torch
import random

# CPU/显卡
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 随机数种子
SEED = 2021
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 训练结果文件名前缀
save_stamp = "sport"

# train 路径
model_file_path = None
train_data_path = "../data/sport/chunked/train_*"
vocab_path = "../data/sport/vocab"
glove_emb_path = "../../glove/glove.42B.300d.txt"
log_root = "../Exam/AOAR(v2)/sport_test(k=5)"

# decode path
decode_model_path = "../Exam/AOAR(v2)/sport_test(k=5)/train_sport_cov/model/model"
decode_data_path = "../data/sport/chunked/test_*"

# 模型超参数
hidden_dim = 256
emb_dim = 300
batch_size = 32
beam_size = 4
max_que_steps = 50
max_rev_steps = 50
max_dec_steps = 50
min_dec_steps = 10
vocab_size = 50000

pointer_gen = True
is_faith = True
is_coverage = True

lr = 5e-5
cov_loss_wt = 1.0
om_loss_wt = 0.2
ft_loss_wt = 0.2
eps = 1e-12
max_grad_norm = 2.0
max_epochs = 50
review_num = 10
topk = 5
