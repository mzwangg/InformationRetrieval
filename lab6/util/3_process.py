import os
from pyhanlp import *
import pickle

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
dir_path = os.path.dirname(os.path.abspath(__file__))
raw_data_path = os.path.join(dir_path, "Data" + os.sep + "classification_data.txt")
stopwords_path = os.path.join(dir_path, "Data" + os.sep + "stopwords.txt")
words_path = os.path.join(dir_path, "Data" + os.sep + "words.pkl")
train_data_path = os.path.join(dir_path, "Data" + os.sep + "train_data.pkl")

def save(data, path):
    with open(path, 'wb') as f_write:
        pickle.dump(data, f_write)

# 创建分词器
segment = HanLP.newSegment().enableNameRecognize(True).enableOrganizationRecognize(True).enablePlaceRecognize(True).enableCustomDictionaryForcing(True)

# 读取并分词
segments_list = []
with open(raw_data_path, 'r', encoding='utf-8') as f_read:
    for line in f_read:
        line = line.strip()
        tokens = line.split(',')

        # 对文本进行分词
        word_nature = segment.seg(tokens[1])
        segment_list = [term.word for term in word_nature]

        # 将分词结果拼接为字符串
        segments_list.append((segment_list, int(tokens[0])))

# 构建词列表
words = []
for segments in segments_list:
    words.extend(segments[0])
# 去重并排序单词列表
words = sorted(list(set(words)))

# 去掉停用词
HanLP.Config.ShowTermNature = False
with open(stopwords_path, 'r', encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])
words = [term for term in words if term not in stopwords]

#保存词列表
save(words, words_path)

data = []
for segment_list in segments_list:
    bag = [0] * len(words)
    for s in segment_list[0]:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # 词在词典中
    data.append([bag, segment_list[1]])
save(data, train_data_path)