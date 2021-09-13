#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : Decode 解码，使用模型进行测试
@Author       : Qinghe Li
@Create time  : 2021-02-23 16:37:33
@Last update  : 2021-09-13 09:46:44
"""

import os
import time
import torch

import config
import data
from data import Vocab, get_batch_data_list, get_input_from_batch, get_init_embeddings
from model import Model
from utils import write_for_eval, eval_decode_result, write_dec_ref


class Beam(object):
    def __init__(self, tokens, log_probs, state, qr_context, qr_cov, reviews_ids, pre_o, true_o):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.qr_context = qr_context
        self.qr_cov = qr_cov
        self.reviews_ids = reviews_ids
        self.pre_o = pre_o
        self.true_o = true_o

    def extend(self, token, log_prob, state, qr_context, qr_cov):
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    qr_context=qr_context,
                    qr_cov=qr_cov,
                    reviews_ids=self.reviews_ids,
                    pre_o=self.pre_o,
                    true_o=self.true_o)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        self._decode_dir = os.path.join(config.log_root, "decode_%s" % (config.save_stamp))
        self.decode_dir = os.path.join(self._decode_dir, "decode_result")
        self.ref_dir = os.path.join(self._decode_dir, "ref_dir")
        self.dec_dir = os.path.join(self._decode_dir, "dec_dir")

        # 创建3个目录
        for p in [self._decode_dir, self.decode_dir, self.ref_dir, self.dec_dir]:
            if not os.path.exists(p):
                os.makedirs(p)

        # 读取并分批测试数据
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = get_batch_data_list(config.decode_data_path, self.vocab,
                                           batch_size=config.beam_size, mode="decode")
        time.sleep(15)
        # 加载模型
        self.model = Model(model_file_path, get_init_embeddings(self.vocab._id_to_word))
        self.model.to(config.DEVICE)
        self.model.eval()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def beam_search(self, batch):
        # 每个batch中只有1个样例被复制beam_size次
        que_batch, que_mask, que_lens, que_extend_vocab, que_asp, que_asp_mask, rev_batch, rev_mask, rev_lens, rev_extend_vocab, rev_asp, rev_opi, rev_asp_mask, extra_zeros, rating_batch, c_t_0, qr_cov = \
            get_input_from_batch(batch)

        h_q, q_state, q_a, topk_h_r, topk_s_r, topk_r_a, topk_r_o, topk_r_o_mask, _, topk_rev_mask, topk_ratings, topk_rev_extend_vocab, idx, topk_scores = \
            self.model.ao_encoder(que_batch, que_lens, que_mask, que_asp, que_asp_mask,
                                  rev_batch, rev_lens, rev_mask, rev_asp, rev_opi, rev_asp_mask,
                                  rating_batch, rev_extend_vocab)

        p_o, o_m = self.model.opinion_classifier(topk_s_r, topk_r_o, topk_r_o_mask, topk_scores)
        s_t = self.model.reduce_state(q_state, o_m)

        dec_h, dec_c = s_t
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      qr_context=c_t_0[0],
                      qr_cov=(qr_cov[0] if config.is_coverage else None),
                      reviews_ids=idx[i],
                      pre_o=p_o[i],
                      true_o=topk_ratings[i])
                 for i in range(config.beam_size)]

        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN)
                             for t in latest_tokens]
            y_t = self.model.ao_encoder.embedding_layer(torch.tensor(latest_tokens, dtype=torch.long, device=config.DEVICE))
            all_state_h = []
            all_state_c = []
            all_qr_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_qr_context.append(h.qr_context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_qr_context, 0)

            qr_cov_t_1 = None
            if config.is_coverage:
                all_qr_cov = []
                for h in beams:
                    all_qr_cov.append(h.qr_cov)
                qr_cov_t_1 = torch.stack(all_qr_cov, 0)

            batch_extend_vocab = torch.cat([que_extend_vocab, topk_rev_extend_vocab.view(config.beam_size, -1)], dim=1)
            final_dist, s_t, c_t, _, qr_cov_t = \
                self.model.decoder(y_t, s_t_1, c_t_1, o_m, h_q, que_mask, q_a,
                                   topk_h_r, topk_rev_mask, topk_r_a, topk_r_o, qr_cov_t_1,
                                   batch_extend_vocab, extra_zeros, steps)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                qr_context_i = c_t[i]
                qr_cov_i = (qr_cov_t[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        qr_context=qr_context_i,
                                        qr_cov=qr_cov_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break
            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

    def decode(self):
        start = time.time()

        counter = 0
        questions = []
        answers = []
        decoded_results = []
        reviews = []
        selected_reviews = []
        pred_o = []
        true_o = []

        for batch in self.batcher:
            # 运行beam search得到解码结果
            best_result = self.beam_search(batch)

            idx = best_result.reviews_ids
            pred_o.append(best_result.pre_o.detach().cpu().tolist())
            true_o.append(best_result.true_o.detach().cpu().tolist())

            # 提取解码得到的单词ID，忽略解码的第1个[START]单词的ID，然后将单词ID转换为对应的单词
            output_ids = [int(t) for t in best_result.tokens[1:]]
            decoded_words = data.outputids2words(
                output_ids, self.vocab, (batch.oovs[0] if config.pointer_gen else None))

            # 如果解码结果中有[STOP]单词，那么去除它
            try:
                fst_stop_idx = decoded_words.index(data.STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            # 将解码结果以及参考答案处理并写入文件，以便后续计算ROUGE分数
            write_dec_ref(batch.original_answers[0],
                          decoded_words,
                          counter,
                          self.ref_dir,
                          self.dec_dir)

            questions.append(batch.original_questions[0])
            answers.append(batch.original_answers[0])
            decoded_results.append(decoded_words)
            reviews.append(batch.original_reviews[0])
            selected_reviews.append([batch.original_reviews[0][i] for i in idx])

            counter += 1
            if counter % 1000 == 0:
                print("%d example in %d sec" % (counter, time.time() - start))
                start = time.time()

        # 将解码结果以及参考答案处理并写入文件，以便后续计算ROUGE分数
        write_for_eval(questions,
                       answers,
                       decoded_results,
                       reviews,
                       selected_reviews,
                       pred_o,
                       true_o,
                       self.decode_dir)

        print("Decoder has finished reading dataset.")
        print("Now starting eval...")
        eval_decode_result(self.ref_dir, self.dec_dir, self.decode_dir)
