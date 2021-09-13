#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 模型训练
@Author       : Qinghe Li
@Create time  : 2021-02-22 17:18:38
@Last update  : 2021-09-13 09:43:06
"""

import os
import time

import torch
import torch.nn as nn

from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from tqdm import tqdm

import config
from data import Vocab, get_batch_data_list, get_input_from_batch, get_output_from_batch, get_init_embeddings
from model import Model
from utils import cosine


class Train(object):
    def __init__(self):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = get_batch_data_list(config.train_data_path,
                                           self.vocab,
                                           batch_size=config.batch_size,
                                           mode="train")
        time.sleep(10)

        train_dir = os.path.join(config.log_root, "train_{}".format(config.save_stamp))

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)

        self.model_dir = os.path.join(train_dir, "model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def ipot_loss(self, x, y, beta=0.01, N=20, K=1, x_mask=None):
        # x(b, l, d) y(b, l, d)
        b = x.size(0)
        l_x = x.size(1)
        l_y = y.size(1)

        sigma = (1.0 / l_y) * torch.ones((b, l_y, 1), device=y.device)              # (b, l_y, 1)
        T = torch.ones((b, l_x, l_y), device=y.device)                              # (b, l_x, l_y)
        C = cosine(x, y)                                              # (b, l_x, l_y)

        if x_mask is not None:
            C = x_mask.view(b, l_x, 1) * C

        A = torch.exp(-beta * C)                                                    # (b, l_x, l_y)

        for t in range(N):
            Q = A * T                                                               # (b, l_x, l_y)
            for k in range(K):
                delta = 1 / (l_x * Q.bmm(sigma))                                    # (b, l_x, 1)
                sigma = 1 / (l_y * Q.transpose(1, 2).bmm(delta))                    # (b, l_y, 1)
            T = (sigma * (delta * Q).transpose(1, 2)).transpose(1, 2)               # (b, l_x, l_y)

        return torch.sum(T * C)

    def save_model(self):
        """保存模型"""
        state = {
            "ao_encoder_state_dict": self.model.ao_encoder.state_dict(),
            "opinion_classifier_state_dict": self.model.opinion_classifier.state_dict(),
            "reduce_state_dict": self.model.reduce_state.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        model_save_path = os.path.join(self.model_dir, "model")
        torch.save(state, model_save_path)

    def setup_train(self, model_file_path=None):
        """模型初始化或加载、初始化迭代次数、损失、优化器"""

        # 初始化模型
        self.model = Model(model_file_path, get_init_embeddings(self.vocab._id_to_word))
        self.model.to(config.DEVICE)

        # 定义优化器
        self.optimizer = Adam(self.model.parameters(), lr=config.lr)

        self.cross_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # 如果传入的已存在的模型路径，加载模型继续训练
        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)

            if not config.is_coverage:
                self.optimizer.load_state_dict(state["optimizer"])
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.to(config.DEVICE)

    def train_one_batch(self, batch):
        """训练一个batch，返回该batch的loss"""

        que_batch, que_mask, que_lens, que_extend_vocab, que_asp, que_asp_mask, rev_batch, rev_mask, rev_lens, rev_extend_vocab, rev_asp, rev_opi, rev_asp_mask, extra_zeros, rating_batch, c_t_0, qr_cov = \
            get_input_from_batch(batch)
        dec_batch, dec_mask, max_dec_len, dec_lens, target_batch, ans_asp, ans_asp_len = \
            get_output_from_batch(batch)

        self.optimizer.zero_grad()

        h_q, q_state, q_a, topk_h_r, topk_s_r, topk_r_a, topk_r_o, topk_r_o_mask, topk_rev_batch, topk_rev_mask, topk_ratings, topk_rev_extend_vocab, idx, topk_scores = \
            self.model.ao_encoder(que_batch, que_lens, que_mask, que_asp, que_asp_mask,
                                  rev_batch, rev_lens, rev_mask, rev_asp, rev_opi, rev_asp_mask,
                                  rating_batch, rev_extend_vocab)

        p_o, o_m = self.model.opinion_classifier(topk_s_r, topk_r_o, topk_r_o_mask, topk_scores)

        s_t = self.model.reduce_state(q_state, o_m)

        if config.is_faith:
            decoded_text = []

        step_losses = []
        for di in range(min(max_dec_len, config.max_dec_steps)):
            y_t = self.model.ao_encoder.embedding_layer(dec_batch[:, di])                       # 当前step解码器的输入单词

            batch_extend_vocab = torch.cat([que_extend_vocab, topk_rev_extend_vocab.view(config.batch_size, -1)], dim=1)
            final_dist, s_t, c_t_0, alpha_qr, qr_cov_next = \
                self.model.decoder(y_t, s_t, c_t_0, o_m, h_q, que_mask, q_a,
                                   topk_h_r, topk_rev_mask, topk_r_a, topk_r_o, qr_cov,
                                   batch_extend_vocab, extra_zeros, di)

            if config.is_faith:
                # w_id = soft_argmax(final_dist).ceil().long()
                _, w_id = torch.topk(final_dist, k=1)
                _w_id = torch.zeros_like(w_id, dtype=w_id.dtype, device=w_id.device)
                decoded_text.append(torch.where(w_id < config.vocab_size, w_id, _w_id).view(config.batch_size, 1))

            target = target_batch[:, di]                                                        # 当前step解码器的目标词            # (b, )
            # final_dist 是词汇表每个单词的概率，词汇表是扩展之后的词汇表
            gold_probs = torch.gather(final_dist, 1, target.unsqueeze(1)).squeeze()             # 取出目标单词的概率
            step_loss = -torch.log(gold_probs + config.eps)                                     # 最大化gold_probs，也就是最小化step_loss

            if config.is_coverage:
                step_cov_loss = torch.sum(torch.min(alpha_qr, qr_cov), dim=1)
                step_loss = step_loss + config.cov_loss_wt * step_cov_loss
                qr_cov = qr_cov_next

            step_mask = dec_mask[:, di]
            step_loss = step_loss * step_mask
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1) / dec_lens
        mle_loss = torch.mean(sum_losses)

        if config.is_faith:
            original_result = self.model.ao_encoder.embedding_layer(topk_rev_batch.reshape(config.batch_size, -1))
            decoded_result = self.model.ao_encoder.embedding_layer(torch.cat(decoded_text, dim=1))             # (b, l_y, 300)
            original_mask = topk_rev_mask.reshape(config.batch_size, -1)

            true_asp = self.model.ao_encoder.embedding_layer(ans_asp).sum(dim=1) / ans_asp_len
            pred_asp = topk_r_a.view(config.batch_size, -1, config.emb_dim).sum(dim=1) / topk_r_o_mask.view(config.batch_size, -1).sum(dim=1, keepdim=True)

            ft_loss = self.ipot_loss(original_result, decoded_result, x_mask=original_mask) + self.mse_loss(pred_asp, true_asp)
            mle_loss = mle_loss + config.ft_loss_wt * ft_loss

        om_loss = self.cross_loss(p_o.view(-1, 5), topk_ratings.view(config.batch_size * config.topk,)) * config.om_loss_wt
        loss = mle_loss + om_loss

        assert torch.isnan(loss).sum() == 0, "error: step loss is Nan ..."

        loss.backward()

        clip_grad_norm_(self.model.parameters(), config.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def trainIters(self, epochs, model_file_path=None):
        # 训练设置
        self.setup_train(model_file_path)

        self.model.train()
        print("Starting to train the model...")
        start = time.time()
        min_loss = 20
        for e in range(epochs):
            loss = 0.0
            pbar = tqdm(total=len(self.batcher), desc=("epochs %d" % (e)), leave=False)
            for batch in self.batcher:
                batch_loss = self.train_one_batch(batch)
                loss += batch_loss
                pbar.update()
            pbar.close()

            loss = loss / len(self.batcher)
            print("epochs %d, seconds for %d epoch: %.2f , loss: %f" % (e, 1, time.time() - start, loss))
            start = time.time()

            if min_loss >= loss:
                min_loss = loss
                self.save_model()


if __name__ == "__main__":
    train_processor = Train()
    train_processor.trainIters(config.max_epochs, config.model_file_path)
