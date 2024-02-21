import json
import os
import configparser
import warnings
import json
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from datetime import datetime
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
import time

warnings.filterwarnings("ignore")

es = Elasticsearch('http://localhost:9200')

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "static" + os.sep + "data.json")
config_path = os.path.join(project_path, "config.ini")
snapshot_path = os.path.join(project_path, "static" + os.sep + "snapshot")

# 读取配置文件
config = configparser.ConfigParser()
config.read(config_path)
BULK_NUM = int(config['DEFAULT']['BULK_NUM'])
SNAPSHOT_MAX_WORKERS = int(config['DEFAULT']['SNAPSHOT_MAX_WORKERS'])
MAX_REPEAT_NUM = int(config['DEFAULT']['MAX_REPEAT_NUM'])

# 创建索引
def create_index(index_name):
    settings = {
            "index": {"number_of_replicas": 2},
            "analysis": {
                "filter": {
                    "ik_stopword": {
                        "type": "stop",
                        "stopwords": "_chinese_"  
                    },
                    "ngram_filter": {
                        "type": "edge_ngram",
                        "min_gram": 2,
                        "max_gram": 15,
                    }
                },
                "analyzer": {
                    "ngram_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_smart",
                        "filter": ["lowercase", "ngram_filter"],
                    }
                }
            }
        }
    mappings = {
        "properties": {
            "url": {
                'type': 'text'
            },
            "title": {
                'type': 'text'
            },
            "author": {
                'type': 'keyword'
            },
            "date":{
                'type': 'date'
            },
            "grade": {
                'type': 'float'
            },
            "people": {
                'type': 'integer'
            },
            "intro": {
                'type': 'text'
            },
            "tags": {
                'type': 'keyword'
            },
            "pagerank": {
                'type': 'float'
            }
        }
    }
    if es.indices.exists(index=index_name):
        es.options(ignore_status=404).indices.delete(index=index_name)
    es.indices.create(index=index_name, settings=settings, mappings=mappings)

def insert_data(data_list, index_name):
    # 定义 Elasticsearch 批量插入的动作列表
    ACTIONS = []

    # 遍历数据列表，按批次进行插入
    for begin in range(0, len(data_list), BULK_NUM):
        for i in range(begin, min(begin + BULK_NUM, len(data_list))):
            data = data_list[i]

            # 构造单条数据的插入动作
            action = {
                "_index": index_name,
                "_source": {
                    "url": data["url"],
                    "title": data["title"],
                    "author": data["author"],
                    "date": datetime.strptime(data["date"], '%Y-%m-%d').date(),
                    "grade": data["grade"],
                    "people": data["people"],
                    "intro": data["intro"],
                    "tags": data["tags"],
                    "pagerank": data["pagerank"]
                }
            }
            # 将单条动作添加到批量插入列表
            ACTIONS.append(action)
        
        # 使用 Elasticsearch 批量插入 API 插入数据，并清空动作列表
        bulk(es, ACTIONS, index=index_name, raise_on_error=True)
        ACTIONS.clear()

    # 刷新索引，确保数据立即可用
    es.indices.refresh(index=index_name)

def get_save_snapshot(spider_data_list):
    # 定义在线程中执行的截图函数
    def get_save_snapshot_thread(data):
        try:
            # 重试最大次数
            for i in range(1, MAX_REPEAT_NUM + 1):
                # 随机休眠0.5-1秒
                time.sleep(random.uniform(0.5, 1))
                
                # 获取URL和保存图片名
                url = data['url']
                saveImgName = url.split('/')[-2]

                # 配置无界面浏览器驱动
                options = webdriver.EdgeOptions()
                options.add_argument('--headless')
                driver = webdriver.Edge(options=options)
                driver.maximize_window()

                # 返回网页的高度的JS代码
                driver.get(url)

                # 若不是最后一次重试并且不是正确的页面，则继续下一次重试
                if(i != MAX_REPEAT_NUM and driver.current_url != url):
                    continue

                # 获取网页高度和宽度，设置浏览器窗口大小
                scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
                scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
                driver.set_window_size(scroll_width, scroll_height)

                # 保存截图
                driver.save_screenshot(snapshot_path + os.sep + saveImgName + ".png")
                break
        except:
            return
        
    if not os.path.exists(snapshot_path):
        os.mkdir(snapshot_path)
    
    # 使用线程池执行截图任务
    with ThreadPoolExecutor(max_workers=SNAPSHOT_MAX_WORKERS) as executor:
        list(tqdm(executor.map(get_save_snapshot_thread, spider_data_list), total=len(spider_data_list), desc="Get Save Snapshot"))


if __name__ == '__main__':
    # 读取数据
    data_list = []
    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            data_list.append(json.loads(line))

    # 创建索引并插入数据
    create_index('douban_index')
    insert_data(data_list, 'douban_index')

    # 获取网页快照
    get_save_snapshot(data_list)