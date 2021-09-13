#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Description  : 构建模型的训练、测试数据
@Author       : Qinghe Li
@Create time  : 2021-02-19 10:28:48
@Last update  : 2021-06-08 08:52:51
"""
import collections
import json
import os
import random
import struct
import sys

from tensorflow.core.example import example_pb2

# 待处理的数据文件
CATEGORY = sys.argv[1]
FILE_NAME = sys.argv[2]
DATA_DIR = os.path.join("../data2", FILE_NAME)
READ_PATH = "../../DATA_PREPARE/data/processed_data/BM25_top10_DP_" + CATEGORY + ".jsonl"
SAVE_TRAIN_PATH = os.path.join(DATA_DIR, str(FILE_NAME + "_train.jsonl"))
SAVE_TEST_PATH = os.path.join(DATA_DIR, str(FILE_NAME + "_test.jsonl"))
FINISHED_DIR = os.path.join(DATA_DIR)

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

VOCAB_SIZE = 50000     # 词汇表大小
CHUNK_SIZE = 3000      # 每个分块example的数量，用于分块的数据


def split_dataset():
    """划分训练和验证数据"""
    items_list = []
    with open(READ_PATH, "r") as fs:
        for line in fs.readlines():
            item = json.loads(line)
            items_list.append(item)
    random.shuffle(items_list)
    n = int(len(items_list) * 0.8)

    train_list = items_list[:n]
    test_list = items_list[n:]

    with open(SAVE_TRAIN_PATH, "w") as ftr:
        for item in train_list:
            ftr.write(json.dumps(item) + "\n")

    with open(SAVE_TEST_PATH, "w") as fte:
        for item in test_list:
            fte.write(json.dumps(item) + "\n")


def chunk_file(in_file, chunks_dir, set_name):
    """构建二进制文件"""
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, "%s_%03d.bin" % (set_name, chunk))
        with open(chunk_fname, "wb") as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack("q", len_bytes)[0]
                example_str = struct.unpack("%ds" % str_len, reader.read(str_len))[0]
                writer.write(struct.pack("q", str_len))
                writer.write(struct.pack("%ds" % str_len, example_str))
            chunk += 1


def chunk_all():
    chunks_dir = os.path.join(FINISHED_DIR, "chunked")
    # 创建一个文件夹来保存分块
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # 将数据分块
    for set_name in ["train", "test"]:
        print("Splitting %s data into chunks..." % set_name)
        in_file = os.path.join(FINISHED_DIR, str(set_name + ".bin"))
        chunk_file(in_file, chunks_dir, set_name)
    print("Saved chunked data in %s" % chunks_dir)


def write_to_bin(in_file, out_file, makevocab=False):
    """生成模型需要的数据文件"""
    if makevocab:
        vocab_counter = collections.Counter()

    count = 0
    with open(out_file, "wb") as writer:
        with open(in_file, "r") as infile:
            for line in infile.readlines():
                item = json.loads(line)
                question_text = item["questions"]["questionText"].strip().lower()
                question_aspect = " ".join(item["questions"]["key"])
                answer_text = item["answers"][0]["answerText"].strip().lower()
                a_aspect = []
                for ao in item["answers"][0]["dp"]:
                    a_aspect.append(ao[1])
                answer_aspect = " ".join(a_aspect)

                reviews = item["reviews"]
                review_texts = []
                review_aspects = []
                review_opinions = []
                review_ratings = []
                for r in reviews:
                    review_texts.append(r["reviewText"].lower())
                    review_ratings.append(int(r["rate"]))
                    aspects = []
                    opinions = []
                    for dp in r["dp"]:
                        aspects.append(dp[1])
                        opinions.append(dp[0])
                    review_aspects.append(" ".join(aspects))
                    review_opinions.append(" ".join(opinions))

                # 写入tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature["question"].bytes_list.value.extend([question_text.encode()])
                tf_example.features.feature["question_aspect"].bytes_list.value.extend([question_aspect.encode()])
                tf_example.features.feature["answer"].bytes_list.value.extend([answer_text.encode()])
                tf_example.features.feature["answer_aspect"].bytes_list.value.extend([answer_aspect.encode()])
                tf_example.features.feature["reviews"].bytes_list.value.extend([str(review_texts).encode()])
                tf_example.features.feature["review_aspects"].bytes_list.value.extend([str(review_aspects).encode()])
                tf_example.features.feature["review_opinions"].bytes_list.value.extend([str(review_opinions).encode()])
                tf_example.features.feature["ratings"].bytes_list.value.extend([str(review_ratings).encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack("q", str_len))
                writer.write(struct.pack("%ds" % str_len, tf_example_str))

                # 可选，将词典写入文件
                if makevocab:
                    q_tokens = question_text.split(" ")
                    a_tokens = answer_text.split(" ")
                    r_tokens = " ".join(review_texts).split(" ")

                    tokens = q_tokens + a_tokens + r_tokens
                    tokens = [t.strip() for t in tokens]        # 去掉句子开头结尾的空字符
                    tokens = [t for t in tokens if t != ""]     # 删除空行
                    vocab_counter.update(tokens)

    print("Finished writing file %s" % out_file)

    # 将词典写入文件
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(FINISHED_DIR, "vocab"), "w") as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + " " + str(count) + "\n")
        print("Finished writing vocab file")


if __name__ == "__main__":
    # 划分数据集为训练集和测试集
    split_dataset()

    # 数据处理，构建为模型输入类型
    write_to_bin(SAVE_TRAIN_PATH, os.path.join(FINISHED_DIR, "train.bin"), makevocab=True)
    write_to_bin(SAVE_TEST_PATH, os.path.join(FINISHED_DIR, "test.bin"))

    # 模型数据分块
    chunk_all()
