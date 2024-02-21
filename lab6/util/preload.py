from pyhanlp import HanLP
from util.neo4j_models import Neo4j_Handle
from util.feedbackward_netwark import feedbackward_netwark
import torch
import pickle
import os

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
dir_path = os.path.dirname(os.path.abspath(__file__))
words_path = os.path.join(dir_path, "Data" + os.sep + "words.pkl")
config_path = os.path.join(dir_path, "config.ini")
model_path = os.path.join(dir_path, "model.pth")

# 初始化模型
def init_model():
    with open(words_path, 'rb') as f_words:
        words = pickle.load(f_words)
    
    # 加载训练好的模型
    model = feedbackward_netwark(len(words), 15)
    model.load_state_dict(torch.load(model_path))
    return model, words

# 初始化pyhanlp
def init_hanlp():
    segment = HanLP.newSegment().enableNameRecognize(True).enableOrganizationRecognize(True).enablePlaceRecognize(True).enableCustomDictionaryForcing(True)
    return segment

# 初始化neo4j
def init_neo4j():
    neo4jconn = Neo4j_Handle()
    neo4jconn.connectNeo4j()
    return neo4jconn

# 初始化
segment = init_hanlp()

# 初始化
neo4jconn = init_neo4j()

# 初始化分类模型，词典等
model_dict = init_model()
