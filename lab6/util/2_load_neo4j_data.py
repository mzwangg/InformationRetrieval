import re
import pandas as pd
from py2neo import Graph
from py2neo import Node
from py2neo import Relationship
from py2neo import NodeMatcher
import warnings
import os
import configparser
import json
from tqdm import tqdm

# 忽略warning
warnings.filterwarnings("ignore")

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, "Data" + os.sep + "spider_data.json")
config_path = os.path.join(dir_path, "config.ini")

config = configparser.ConfigParser()
config.read(config_path)
MIN_SLEEP_TIME = float(config['DEFAULT']['MIN_SLEEP_TIME'])
MAX_SLEEP_TIME = float(config['DEFAULT']['MAX_SLEEP_TIME'])
CATALOG_PAGE_NUM = int(config['DEFAULT']['CATALOG_PAGE_NUM'])
MAX_FILM_NUM = int(config['DEFAULT']['MAX_FILM_NUM'])
MAX_ACTOR_NUM = int(config['DEFAULT']['MAX_ACTOR_NUM'])

# 查询节点
def match_node(graph, label, attrs):
    n = "_.name=\"" + str(attrs["name"]) + "\""
    matcher = NodeMatcher(graph)
    return matcher.match(label).where(n).first()


# 建立一个节点
def create_node(graph, label, attrs):
    # 查询是否已经存在，若存在则返回节点，否则返回None
    value = match_node(graph, label, attrs)
    # 如果要创建的节点不存在则创建
    if value is None:
        node = Node(label, **attrs)
        n = graph.create(node)
        return n
    return False


def load_neo4j_data(graph, data):
    # 对数据进行预处理
    data = data.applymap(lambda x: x.strip("/") if isinstance(x, str) else x)
    data['title'] = data['title'].apply(lambda x: re.compile(r"[\u4e00-\u9fff]+").findall(x)[0])
    data['date'] = data['date'].apply(lambda x: re.findall(r'\d+', x)[0])
    data['mark'] = data['mark'].apply(lambda x: str(x))

    for i in tqdm(range(MAX_FILM_NUM), total=MAX_FILM_NUM, desc="Processing data"):
        # 电影名称
        film_title = data.loc[i, 'title']

        # 电影节点
        row_dict = data.iloc[i].drop('title').to_dict()
        row_dict['name'] = film_title
        n = create_node(graph, "电影", row_dict)
        if n == False:
            continue

        # 导演节点
        director_list = data.loc[i, 'director'].split("/")
        for director in director_list:
            if director == '':
                continue
            create_node(graph, "导演", {"name": director})
            director_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                        "导演",
                                        graph.nodes.match("导演", name=director).first())
            graph.create(director_rel)

        # 编剧节点
        creator_list = data.loc[i, 'creator'].split("/")
        for creator in creator_list:
            if creator == '':
                continue
            create_node(graph, "编剧", {"name": creator})
            creator_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                       "编剧",
                                       graph.nodes.match("编剧", name=creator).first())
            graph.create(creator_rel)

        # 主演节点
        actor_list = data.loc[i, 'actor'].split("/")
        for j in range(min(MAX_ACTOR_NUM, len(actor_list))):
            actor = actor_list[j]
            if actor == '':
                continue
            create_node(graph, "主演", {"name": actor})
            actor_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                     "主演",
                                     graph.nodes.match("主演", name=actor).first())
            graph.create(actor_rel)

        # 类型节点
        type_list = data.loc[i, 'type'].split("/")
        for type in type_list:
            if type == '':
                continue
            create_node(graph, "类型", {"name": type})
            type_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                    "类型",
                                    graph.nodes.match("类型", name=type).first())
            graph.create(type_rel)

        # 制片国家/地区节点
        location_list = data.loc[i, 'country'].split("/")
        for location in location_list:
            if location == '':
                continue
            create_node(graph, "制片国家或地区", {"name": location})
            location_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                        "制片国家或地区",
                                        graph.nodes.match("制片国家或地区", name=location).first())
            graph.create(location_rel)

        # 语言节点
        language_list = data.loc[i, 'language'].split("/")
        for language in language_list:
            if language == '':
                continue
            create_node(graph, "语言", {"name": language})
            language_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                        "语言",
                                        graph.nodes.match("语言", name=language).first())
            graph.create(language_rel)

        # 制片年份节点
        year = data.loc[i, 'date']
        create_node(graph, "制片年份", {"name": year})
        year_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                "制片年份",
                                graph.nodes.match("制片年份", name=year).first())
        graph.create(year_rel)

        # 评分节点
        mark = data.loc[i, 'mark']
        create_node(graph, "评分", {"name": mark})
        mark_rel = Relationship(graph.nodes.match(name=film_title).first(),
                                "评分",
                                graph.nodes.match("评分", name=mark).first())
        graph.create(mark_rel)


if __name__ == '__main__':
    # 读取数据
    datas = []
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            datas.append(json.loads(line))
    datas = pd.DataFrame(datas)

    # 连接本地的 Neo4j 数据库，并清空
    graph = Graph("bolt: // localhost:7687", auth=("neo4j", "12345678"))
    graph.delete_all()

    # 加载neo4j数据
    load_neo4j_data(graph, datas)