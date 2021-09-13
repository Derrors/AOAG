#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 
@Author       : Qinghe Li
@Create time  : 2021-02-23 15:08:26
@Last update  : 2021-07-20 17:19:32
"""

import torch
import config
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EmbeddingLayer(nn.Module):
    """共享嵌入层"""

    def __init__(self, emb_matrix=None):
        super(EmbeddingLayer, self).__init__()

        if emb_matrix is not None:
            self.emb_mat = nn.Embedding.from_pretrained(torch.FloatTensor(emb_matrix), freeze=False, padding_idx=1)
        else:
            self.emb_mat = nn.Embedding(config.vocab_size, config.emb_dim)

    def forward(self, inputs):
        return self.emb_mat(inputs)


class Encoder(nn.Module):
    """编码器"""

    def __init__(self):
        super(Encoder, self).__init__()

        self.lstm = nn.LSTM(config.emb_dim * 2,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

    def forward(self, embedded_input, seq_lens):
        packed = pack_padded_sequence(embedded_input, seq_lens, batch_first=True, enforce_sorted=False)

        # 编码得到每个单词的隐层表示、最后一个step的h和c
        output, state = self.lstm(packed)           # hidden : ((2, b, h), (2, b, h))

        encoder_outputs, _ = pad_packed_sequence(output, batch_first=True)  # (b, l, 2h)
        encoder_outputs = encoder_outputs.contiguous()                      # (b, l, 2h)

        return encoder_outputs, state


class SelfAttention(nn.Module):
    def __init__(self, input_size):
        """自注意力层"""
        super(SelfAttention, self).__init__()

        self.input_size = input_size
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Tanh(),
            nn.Linear(input_size, 1, bias=False),
        )

    def forward(self, inputs, mask):
        """
        Args:
            inputs [tensor]: input tensor (b, l, h)
            mask [tensor]: mask matrix (b, l)
        Returns:
            outputs [tensor]: output tensor (b, h)
        """
        b, l, h = inputs.size()
        mask = mask.unsqueeze(1)                # (b, 1, l)
        alpha = F.softmax(self.attention(inputs), dim=2).reshape(b, 1, l) * mask            # (b, 1, l)
        outputs = alpha.bmm(inputs.view(b, l, h)).squeeze()            # (b, h)
        return outputs


class AOEncoder(nn.Module):
    def __init__(self, emb_matrix=None):
        super(AOEncoder, self).__init__()

        self.embedding_layer = EmbeddingLayer(emb_matrix)
        self.encoder_layer = Encoder()
        self.self_attention_layer = SelfAttention(input_size=config.hidden_dim * 2)

        self.revelance = nn.Bilinear(config.hidden_dim * 2 + config.emb_dim, config.hidden_dim * 2 + config.emb_dim, 1, bias=False)

    def forward(self,
                que_batch, que_lens, que_mask, que_asp, que_asp_mask,
                rev_batch, rev_lens, rev_mask, rev_asp, rev_opi, rev_asp_mask,
                ratings, rev_extend_vocab):

        # Step1: Embedding Layer
        q_a = self.embedding_layer(que_asp)                                 # (b, 3, d)
        r_a = self.embedding_layer(rev_asp)                                 # (b, k, 5, d)
        r_o = self.embedding_layer(rev_opi)                                 # (b, k, 5, d)

        l_r = rev_batch.size(2)
        e_q = self.embedding_layer(que_batch)                               # (b, l_q, d)
        e_r = self.embedding_layer(rev_batch.view(-1, l_r))                 # (b * k, l_r, d)

        b, l_q, d = e_q.size()
        n_q = q_a.size(1)
        n_r = r_a.size(2)
        b_r = e_r.size(0)
        k = rev_batch.size(1)

        # Step2: Aspect-aware Encoder Layer
        h_q, q_state = self.encoder_layer(
            torch.cat([e_q.unsqueeze(1).expand(b, n_q, l_q, d), q_a.unsqueeze(2).expand(b, n_q, l_q, d)], dim=3).reshape(b * n_q, l_q, d * 2),
            que_lens.unsqueeze(1).expand(b, n_q).reshape(b * n_q, ))                    # (b * n_q, l_q, 2h), ((2, b * n_q, h), (2, b * n_q, h))
        h = h_q.size(2)
        h_q = que_asp_mask.unsqueeze(2).expand(b, n_q, l_q).view(b, n_q, l_q, 1) * h_q.reshape(b, n_q, l_q, h)      # (b, n_q, l_q, 2h)

        h_r, _ = self.encoder_layer(
            torch.cat([e_r.unsqueeze(1).expand(b_r, n_r, l_r, d), r_a.view(b_r, n_r, d).unsqueeze(2).expand(b_r, n_r, l_r, d)], dim=3).reshape(b_r * n_r, l_r, d * 2),
            rev_lens.unsqueeze(2).expand(b, k, n_r).reshape(b_r * n_r, ))                 # (b_r * n_r, l_r, 2h)
        h_r = rev_asp_mask.unsqueeze(3).expand(b, k, n_r, l_r).view(b_r, n_r, l_r).unsqueeze(3) * h_r.reshape(b_r, n_r, l_r, h)

        s, c = q_state
        s = (que_asp_mask.unsqueeze(0).expand(2, b, n_q).view(2, b, n_q, 1) * s.reshape(2, b, n_q, -1)).sum(dim=2) / que_asp_mask.unsqueeze(0).expand(2, b, n_q).sum(dim=2, keepdim=True)
        c = (que_asp_mask.unsqueeze(0).expand(2, b, n_q).view(2, b, n_q, 1) * c.reshape(2, b, n_q, -1)).sum(dim=2) / que_asp_mask.unsqueeze(0).expand(2, b, n_q).sum(dim=2, keepdim=True)
        q_state = (s, c)                                # ((2, b, h), (2, b, h))

        # Step3: 评论筛选：使用 Self Attention 分别得到问题与评论表示并计算它们之间的相关性得分
        # s_q = self.self_attention_layer(h_q.sum(dim=1) / que_asp_mask.sum(dim=1, keepdim=True).expand(b, l_q).unsqueeze(2), que_mask)                             # (b, 2h)
        s_q = self.self_attention_layer(h_q.view(b * n_q, l_q, -1), que_mask.unsqueeze(1).expand(b, n_q, l_q).contiguous().view(b * n_q, l_q)).view(b, n_q, -1)
        _s_q = s_q.sum(dim=1) / que_asp_mask.sum(dim=1, keepdim=True)                                               # (b, 2h)
        _q_a = q_a.sum(dim=1) / que_asp_mask.sum(dim=1, keepdim=True)                                               # (b, d)

        # _h_r, _ = h_r.max(dim=1)                                                                                  # (b_r, l_r, 2h)
        # s_r = self.self_attention_layer(_h_r, rev_mask.view(b_r, l_r))                                            # (b_r, 2h)
        s_r = self.self_attention_layer(h_r.view(b_r * n_r, l_r, -1), rev_mask.unsqueeze(2).expand(b, k, n_r, l_r).contiguous().view(b_r * n_r, l_r)).view(b_r, n_r, -1)
        _s_r = s_r.sum(dim=1) / rev_asp_mask.view(b_r, n_r).sum(dim=1, keepdim=True)                                # (b_r, 2h)
        _r_a = r_a.view(b_r, n_r, d).sum(dim=1) / rev_asp_mask.view(b_r, n_r).sum(dim=1, keepdim=True)              # (b_r, d)

        a_s_q = torch.cat([_s_q, _q_a], dim=1)                                                                      # (b, 2h + d)
        a_s_r = torch.cat([_s_r, _r_a], dim=1).view(b, k, -1)                                                       # (b, k, 2h + d)

        # 计算问题与每条评论之间的相关性
        revelant_scores = torch.tanh(self.revelance(a_s_q.unsqueeze(1).expand_as(a_s_r).contiguous(), a_s_r).squeeze())          # (b, k)

        # 依据相关性得分来选择 k 条相关的评论
        topk_scores, idx = torch.topk(revelant_scores, k=config.topk, dim=1)

        topk_s_r = torch.zeros((b, config.topk, n_r, h), dtype=s_r.dtype, device=s_r.device)
        topk_h_r = torch.zeros((b, config.topk, n_r, l_r, h), dtype=h_r.dtype, device=h_r.device)
        topk_r_a = torch.zeros((b, config.topk, n_r, d), dtype=r_a.dtype, device=r_a.device)
        topk_r_o = torch.zeros((b, config.topk, n_r, d), dtype=r_o.dtype, device=r_o.device)
        topk_r_o_mask = torch.zeros((b, config.topk, n_r), dtype=rev_asp_mask.dtype, device=rev_asp_mask.device)
        topk_rev_batch = torch.zeros((b, config.topk, l_r), dtype=rev_batch.dtype, device=rev_batch.device)
        topk_rev_mask = torch.zeros((b, config.topk, l_r), dtype=rev_mask.dtype, device=rev_mask.device)
        topk_ratings = torch.zeros((b, config.topk), dtype=ratings.dtype, device=ratings.device)
        topk_rev_extend_vocab = torch.zeros((b, config.topk, rev_extend_vocab.size(2)), dtype=rev_extend_vocab.dtype, device=rev_extend_vocab.device)

        for i in range(b):
            topk_s_r[i] = torch.index_select(s_r.view(b, k, n_r, h)[i], dim=0, index=idx[i])
            topk_h_r[i] = torch.index_select(h_r.view(b, k, n_r, l_r, h)[i], dim=0, index=idx[i])                   # (b, topk, 5, l_r, 2h)
            topk_r_a[i] = torch.index_select(r_a[i], dim=0, index=idx[i])
            topk_r_o[i] = torch.index_select(r_o[i], dim=0, index=idx[i])
            topk_r_o_mask[i] = torch.index_select(rev_asp_mask[i], dim=0, index=idx[i])
            topk_rev_batch[i] = torch.index_select(rev_batch[i], dim=0, index=idx[i])
            topk_rev_mask[i] = torch.index_select(rev_mask[i], dim=0, index=idx[i])
            topk_ratings[i] = torch.index_select(ratings[i], dim=0, index=idx[i])
            topk_rev_extend_vocab[i] = torch.index_select(rev_extend_vocab[i], dim=0, index=idx[i])

        return h_q, q_state, q_a, topk_h_r, topk_s_r, topk_r_a, topk_r_o, topk_r_o_mask, topk_rev_batch, topk_rev_mask, topk_ratings, topk_rev_extend_vocab, idx, topk_scores


class OpinionClassifier(nn.Module):
    """评论观点预测"""

    def __init__(self):
        super(OpinionClassifier, self).__init__()

        self.W = nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.hidden_dim * 2 + config.emb_dim, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim * 2 + config.emb_dim, config.hidden_dim * 2 + config.emb_dim, bias=True),
            nn.Tanh(),
            nn.Linear(config.hidden_dim * 2 + config.emb_dim, 5, bias=False))

    def forward(self, s_r, r_o, r_o_mask, alpha):
        b, k, n_r, _ = s_r.size()

        o_s_r = torch.cat([s_r, r_o], dim=3).reshape(b * k, n_r, -1)                                # (b * k, n_r, h + d)
        weight = self.W(o_s_r).bmm(o_s_r.transpose(1, 2))                                           # (b * k, n_r, n_r)
        w, _ = torch.max(weight, dim=2, keepdim=True)                                               # (b * k, n_r, 1)

        o_r = F.softmax(r_o_mask.view(b * k, n_r, 1) * w, dim=1).transpose(1, 2).bmm(o_s_r)         # (b * k, 1, h + d)
        o_r = o_r.squeeze().reshape(b, k, -1)                                                       # (b, k, h + d)

        p_o = F.softmax(self.classifier(o_r), dim=2)                                                # (b, k, 5)
        o_m = F.softmax(alpha, dim=1).unsqueeze(1).bmm(o_r).squeeze()                               # (b, h + d)

        return p_o, o_m


class ReduceState(nn.Module):
    """将编码器最后一个step的隐层状态与问题属性词、观点表示进行拼接、降维以适应解码器"""

    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(config.hidden_dim * 4 + config.emb_dim, config.hidden_dim)
        self.reduce_c = nn.Linear(config.hidden_dim * 4 + config.emb_dim, config.hidden_dim)

    def forward(self, hidden, o_m):
        h, c = hidden                                                                       # ((2, b, h), (2, b, h))

        h_in = h.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)               # (b, 2h)
        c_in = c.transpose(0, 1).contiguous().view(-1, config.hidden_dim * 2)               # (b, 2h)

        hidden_reduced_h = F.relu(self.reduce_h(torch.cat([h_in, o_m], dim=-1)))            # (b, h)
        hidden_reduced_c = F.relu(self.reduce_c(torch.cat([c_in, o_m], dim=-1)))            # (b, h)

        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0))               # ((1, b, h), (1, b, h))


class DynamicSelect(nn.Module):
    def __init__(self, input_size):
        super(DynamicSelect, self).__init__()

        self.W = nn.Bilinear(input_size, config.emb_dim, 1, bias=False)

    def forward(self, s_t, h_s, s_a):
        d = s_t.size(1)
        b, n, l, h = h_s.size()

        w = torch.tanh(self.W(s_t.unsqueeze(1).expand(b, n, d).contiguous(), s_a)).transpose(1, 2)  # (b, 1, n_q)
        w = w / (w.sum(dim=2, keepdim=True) + config.eps)
        w_h_s = w.bmm(h_s.view(b, n, -1)).squeeze().reshape(b, -1, h)                               # (b, l_q, h)
        w_a = w.bmm(s_a).squeeze()                                                                  # (b, d)

        return w_h_s, w_a


class DecoderAttention(nn.Module):
    def __init__(self):
        super(DecoderAttention, self).__init__()

        self.U = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)

        if config.is_coverage:
            self.Wc = nn.Linear(1, config.hidden_dim * 2, bias=False)

        self.Ws = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        self.Wqr = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2)
        self.Vr = nn.Linear(config.hidden_dim * 2, 1, bias=False)

    def forward(self, h_q, q_mask, h_r, r_mask, s_t, qr_cov):
        b, l_q, h = h_q.size()                  # (b, l_q, h)
        _, k, l_r, _ = h_r.size()               # (b, k, l_r, h)

        # 计算问题与相关评论之间的 Cross-Attention
        _h_q = h_q.unsqueeze(1).expand(b, k, l_q, h).reshape(b * k, l_q, h)
        _h_r = h_r.reshape(b * k, l_r, h)
        _q_mask = q_mask.unsqueeze(1).expand(b, k, l_q).reshape(-1, l_q)
        _r_mask = r_mask.reshape(-1, l_r)

        U = torch.tanh(torch.bmm(self.U(_h_q), _h_r.transpose(1, 2).contiguous()))                      # (b * k, l_q, l_r)

        U_q, _ = torch.max(U, dim=2)                                                                    # (b * k, l_q)
        U_r, _ = torch.max(U, dim=1)                                                                    # (b * k, l_r)
        alpha_q = F.softmax(U_q, dim=-1)                                                                # (b * k, l_q)
        alpha_r = F.softmax(U_r, dim=-1)                                                                # (b * k, l_r)

        pai_q = ((alpha_q * _q_mask).unsqueeze(2) * _h_q).reshape(b, k, l_q, h).mean(dim=1)             # (b, l_q, h)
        pai_r = ((alpha_r * _r_mask).unsqueeze(2) * _h_r).reshape(b, k, l_r, h).reshape(b, k * l_r, h)  # (b, k * l_r, h)

        dec_fea = self.Ws(s_t.unsqueeze(1).expand(b, l_q + k * l_r, h)) + self.Wqr(torch.cat([pai_q, pai_r], dim=1))

        if config.is_coverage:
            cov_input = qr_cov.view(b, l_q + k * l_r, 1)
            cov_fea = self.Wc(cov_input)
            dec_fea = dec_fea + cov_fea

        alpha_qr = self.Vr(torch.tanh(dec_fea)).squeeze()                                               # (b, l_q + k * l_r)
        mask = torch.cat([q_mask, r_mask.view(b, k * l_r)], dim=1)

        alpha_qr_ = F.softmax(alpha_qr, dim=1) * mask                                                   # (b, l_q + k * l_r)
        normalization_factor = alpha_qr_.sum(dim=1, keepdim=True)
        alpha_qr = (alpha_qr_ / normalization_factor)                                                   # (b, l_q + k * l_r)

        c_t = torch.bmm(alpha_qr.unsqueeze(1), torch.cat([pai_q, pai_r], dim=1)).squeeze()              # (b, h)

        if config.is_coverage:
            qr_cov = qr_cov + alpha_qr

        return c_t, alpha_qr, qr_cov


class Decoder(nn.Module):
    """解码器"""

    def __init__(self):
        super(Decoder, self).__init__()

        self.x_context = nn.Linear(config.emb_dim + config.hidden_dim * 2, config.emb_dim)
        self.lstm = nn.LSTM(config.emb_dim,
                            config.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.ds_q = DynamicSelect(config.hidden_dim * 2)
        self.ds_a = DynamicSelect(config.hidden_dim * 2 + config.emb_dim)
        self.ds_o = DynamicSelect(config.hidden_dim * 4 + config.emb_dim)

        # 门控机制部分
        # self.W_s = nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2, bias=False)
        # self.W_f = nn.Linear(config.emb_dim * 2, config.hidden_dim * 2, bias=False)
        # self.W_g = nn.Linear(config.hidden_dim * 2, 1)

        self.decoder_attention = DecoderAttention()

        if config.pointer_gen:
            self.p_gen = nn.Linear(config.hidden_dim * 4, 1)

        self.pred_vocab = nn.Sequential(
            nn.Linear(config.hidden_dim * 4, config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.vocab_size))

    def forward(self, y_t, s_t_0, c_t_0, o_m, h_q, q_mask, q_a, h_r, r_mask, r_a, r_o, qr_cov, batch_extend_vocab, extra_zeros, step):

        b, k, n_r, l_r, h = h_r.size()
        _, n_q, d = q_a.size()

        # 解码器初始状态
        if not self.training and step == 0:
            h_decoder, c_decoder = s_t_0
            s_t_hat = torch.cat([h_decoder.view(b, -1), c_decoder.view(b, -1)], dim=1)              # (b, 2h)
            q_t, q_a_t = self.ds_q(s_t_hat, h_q, q_a)                                                       # (b ,l_q, h) (b, d)
            s_a_t = torch.cat([s_t_hat.unsqueeze(1).expand(b, k, h), q_a_t.unsqueeze(1).expand(b, k, d)], dim=2).reshape(b, k, -1).reshape(b * k, -1)
            h_a_t, r_a_t = self.ds_a(s_a_t, h_r.view(b * k, n_r, l_r, h), r_a.view(b * k, n_r, d))          # (b * k, l_r, h) (b * k, d)
            s_o_t = torch.cat([s_t_hat.unsqueeze(1).expand(b, k, h), o_m.unsqueeze(1).expand(b, k, h + d)], dim=2).reshape(b, k, -1).reshape(b * k, -1)
            h_o_t, r_o_t = self.ds_o(s_o_t, h_r.view(b * k, n_r, l_r, h), r_o.view(b * k, n_r, d))              # (b * k, l_r, h) (b * k, d)

            # beta = self.W_g(self.W_s(s_t_hat.unsqueeze(1).expand(b, k, h).reshape(b * k, h)) + self.W_f(torch.cat([r_a_t, r_o_t], dim=1)))         # (b * k, 1)
            # beta = beta.unsqueeze(1).expand(b * k, l_r, 1)
            # r_t = (beta * h_a_t + (1 - beta) * h_o_t).reshape(b, k, l_r, h)                                     # (b, k, l_r, h)
            r_t = (h_a_t + h_o_t).reshape(b, k, l_r, h)                                                          # (b, k, l_r, h)

            c_t_0, _, qr_cov_next = self.decoder_attention(q_t, q_mask, r_t, r_mask, s_t_hat, qr_cov)
            qr_cov = qr_cov_next

        x = self.x_context(torch.cat([y_t, c_t_0], dim=1))
        _, s_t = self.lstm(x.unsqueeze(1), s_t_0)                                                           # 解码器在当前 step 的解码输出

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat([h_decoder.view(b, -1), c_decoder.view(b, -1)], dim=1)                          # (b, h)

        q_t, q_a_t = self.ds_q(s_t_hat, h_q, q_a)                                                           # (b ,l_q, h) (b, d)

        s_a_t = torch.cat([s_t_hat.unsqueeze(1).expand(b, k, h), q_a_t.unsqueeze(1).expand(b, k, d)], dim=2).reshape(b, k, -1).reshape(b * k, -1)
        h_a_t, r_a_t = self.ds_a(s_a_t, h_r.view(b * k, n_r, l_r, h), r_a.view(b * k, n_r, d))              # (b * k, l_r, h) (b * k, d)
        s_o_t = torch.cat([s_t_hat.unsqueeze(1).expand(b, k, h), o_m.unsqueeze(1).expand(b, k, h + d)], dim=2).reshape(b, k, -1).reshape(b * k, -1)
        h_o_t, r_o_t = self.ds_o(s_o_t, h_r.view(b * k, n_r, l_r, h), r_o.view(b * k, n_r, d))              # (b * k, l_r, h) (b * k, d)

        # TODO：进行简化实验
        # beta = self.W_g(self.W_s(s_t_hat.unsqueeze(1).expand(b, k, h).reshape(b * k, h)) + self.W_f(torch.cat([r_a_t, r_o_t], dim=1)))         # (b * k, 1)
        # beta = beta.unsqueeze(1).expand(b * k, l_r, 1)
        # r_t = (beta * h_a_t + (1 - beta) * h_o_t).reshape(b, k, l_r, h)                                   # (b, k, l_r, h)
        r_t = (h_a_t + h_o_t).reshape(b, k, l_r, h)                                                          # (b, k, l_r, h)

        c_t, alpha_qr, qr_cov_next = self.decoder_attention(q_t, q_mask, r_t, r_mask, s_t_hat, qr_cov)

        if self.training or step > 0:
            qr_cov = qr_cov_next

        h_s_t = torch.cat([s_t_hat, c_t], dim=-1)                           # (b, 2h)

        p_gen = None
        if config.pointer_gen:
            p_gen = torch.sigmoid(self.p_gen(h_s_t))          # (b, 1)

        p_v = F.softmax(self.pred_vocab(h_s_t), dim=1)                      # (b, v)

        if config.pointer_gen:
            vocab_dist_ = p_gen * p_v
            alpha_qr_ = (1 - p_gen) * alpha_qr                              # (b, l_q + k * l_r)

            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, batch_extend_vocab, alpha_qr_)
        else:
            final_dist = p_v

        return final_dist, s_t, c_t, alpha_qr, qr_cov


class Model(nn.Module):
    def __init__(self, model_file_path=None, emb_matrix=None):
        super(Model, self).__init__()

        self.ao_encoder = AOEncoder(emb_matrix)
        self.opinion_classifier = OpinionClassifier()
        self.reduce_state = ReduceState()
        self.decoder = Decoder()

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.ao_encoder.load_state_dict(state["ao_encoder_state_dict"])
            self.opinion_classifier.load_state_dict(state["opinion_classifier_state_dict"])
            self.reduce_state.load_state_dict(state["reduce_state_dict"])
            self.decoder.load_state_dict(state["decoder_state_dict"], strict=False)
