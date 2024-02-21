import requests
import json
import random
from tqdm import tqdm
from bs4 import BeautifulSoup
import os
import configparser
import warnings
import networkx as nx
import re
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import jieba

# 忽略warning
warnings.filterwarnings("ignore")

# 要爬取的url
book_tag_url = 'https://book.douban.com/tag/'
music_tag_url = 'https://music.douban.com/tag/'

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "static" + os.sep + "data.json")
config_path = os.path.join(project_path, "config.ini")
snapshot_path = os.path.join(project_path, "static" + os.sep + "snapshot")
stopwords_psth = os.path.join(project_path, "static" + os.sep + "baidu_stopwords.txt")

# 读取配置文件
config = configparser.ConfigParser()
config.read(config_path)
TAG_NUM = int(config['DEFAULT']['TAG_NUM'])
LIST_PAGE_NUM = int(config['DEFAULT']['LIST_PAGE_NUM'])
SPIDER_MAX_WORKERS = int(config['DEFAULT']['SPIDER_MAX_WORKERS'])
SNAPSHOT_MAX_WORKERS = int(config['DEFAULT']['SNAPSHOT_MAX_WORKERS'])
MAX_REPEAT_NUM = int(config['DEFAULT']['MAX_REPEAT_NUM'])
MIN_SLEEP_TIME = float(config['DEFAULT']['MIN_SLEEP_TIME'])
MAX_SLEEP_TIME = float(config['DEFAULT']['MAX_SLEEP_TIME'])
CLUSER_MIN_NUM = int(config['DEFAULT']['CLUSER_MIN_NUM'])
CLUSER_MAX_NUM = int(config['DEFAULT']['CLUSER_MAX_NUM'])
CLUSER_STRIDE = int(config['DEFAULT']['CLUSER_STRIDE'])

# 随机的user-agent信息
ua = UserAgent()
headers = { 'User-Agent':ua.random}

# 爬取标签页，得到目录页的url列表
def get_list_url(tag_url):
    list_url_list = []
    for _ in range(MAX_REPEAT_NUM):
        time.sleep(random.uniform(MIN_SLEEP_TIME, MAX_SLEEP_TIME)) # 随机休眠
        r = requests.get(tag_url, headers=headers)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')

            # 对于豆瓣音乐，只爬取风格类的url_list
            if tag_url == music_tag_url:
                soup = soup.find('div', id='风格', class_='mod')

            # 查找所有符合条件的<td>元素
            td_elements = soup.find_all('td')

            # 提取URL并保存到列表中
            origin_list_url_list = []
            for td_element in td_elements:
                a_element = td_element.find('a')
                if a_element and 'href' in a_element.attrs:
                    url = tag_url[:-5] + a_element['href']
                    origin_list_url_list.append(url)
            
            # 控制爬取标签页的数量， 使得music_dict和book_dict大小一致
            if len(origin_list_url_list) > TAG_NUM:
                origin_list_url_list = random.sample(origin_list_url_list, TAG_NUM)
            
            # 添加不同的页数
            for origin_url in origin_list_url_list:
                temp_url = origin_url + '?start={}&type=T'
                for i in range(LIST_PAGE_NUM):
                    url = temp_url.format(i * 20)
                    list_url_list.append(url)
            break
    return list_url_list

# 爬取目录页，得到详情页的url列表
def get_detail_url(list_url):
    def process_list_url(list_url):
        for _ in range(MAX_REPEAT_NUM):
            time.sleep(random.uniform(MIN_SLEEP_TIME, MAX_SLEEP_TIME)) # 随机休眠
            r = requests.get(list_url, headers=headers)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')

                # 对于豆瓣读书与豆瓣音乐，使用不同的选择器
                if list_url.split('/')[2] == "book.douban.com":
                    link_elements = soup.select('ul.subject-list li.subject-item div.pic a.nbg')
                else:
                    link_elements = soup.find_all('a', class_='nbg')

                # 构建(tag,url)元组的列表
                tags = list_url.split('?')[0].split('/')[-1]
                links = [(tags, link['href']) for link in link_elements]

                return links
            return []

    # 并行爬取
    with ThreadPoolExecutor(max_workers=SPIDER_MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_list_url, list_url), total=len(list_url), desc="Processing URLs"))

    # 合并结果
    detail_url_list = []
    for result in results:
        detail_url_list.extend(result)
    return detail_url_list

def get_data_list(detail_url_list, max_workers=5):
    def process_detail_url(tag_url_tuple):
        tags, url = tag_url_tuple
        for _ in range(MAX_REPEAT_NUM):
            time.sleep(random.uniform(MIN_SLEEP_TIME, MAX_SLEEP_TIME)) # 随机休眠
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                try:
                    soup = BeautifulSoup(r.text, 'html.parser')

                    # 获取数据
                    data = dict()
                    type = 'book' if url.split('/')[2] == 'book.douban.com' else 'music'

                    # 网址
                    data['url'] = url

                    # 标签
                    data['tags'] = [tags]

                    # 标题
                    title_element = soup.find('h1').find('span')
                    if title_element:
                        data['title'] = title_element.get_text().strip()
                    else:
                        return None

                    # 作者
                    text = '作者' if type == 'book' else '表演者'
                    author_span = soup.select_one(f'div#info span.pl:contains({text})')
                    if author_span:
                        data['author'] = author_span.find_next('a').text.strip().replace('\n', '').split('/')
                        data['tags'] += data['author']
                    else:
                        return None

                    # 评分
                    grade_element = soup.select_one('.ll.rating_num')
                    if grade_element:
                        try:
                            data['grade'] = float(grade_element.text.strip())
                        except ValueError:
                            return None
                    else:
                        return None

                    # 评分人数
                    people_element = soup.select_one('div.rating_sum a.rating_people span[property="v:votes"]')
                    if people_element:
                        try:
                            data['people'] = int(people_element.text.strip())
                        except ValueError:
                            return None
                    else:
                        return None

                    # 简介
                    intro_element = soup.find('div', class_='intro') if type == 'book' else soup.find('span', property='v:summary')
                    if intro_element:
                        data['intro'] = intro_element.text.strip().replace('\u3000', '')
                    else:
                        return None

                    # 日期
                    date_match = re.search(r'\d{4}-\d{1,2}-\d{1,2}', r.text)
                    if date_match:
                        data['date'] = date_match.group()
                    else:
                        date_match = re.search(r'\d{4}-\d{1,2}', r.text)
                        if data_path:
                            data['date'] = date_match.group() + "-01"
                        else:
                            return None

                    # 关联衔接
                    url_elements = soup.select('div.knnlike div dl dt a') if type == 'book' else soup.select(
                        'dl.subject-rec-list dt a')
                    if url_elements:
                        data['link'] = [url['href'] for url in url_elements]
                    else:
                        return None

                    return data
                except:
                    return None
        
    # 并行爬取
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_detail_url, detail_url_list), total=len(detail_url_list), desc="Processing details"))

    # 合并结果
    merged_results = {}
    for result in results:
        if result is None:
            continue

        # 可能在不同标签页指向的目录页中，有重复的详情页，所以进行去重
        # 只保留一个，并将url合并
        url = result['url']
        tags = result['tags']
        if url not in merged_results:
            merged_results[url] = result
        else:
            merged_results[url]['tags'] = list(set(merged_results[url]['tags'] + tags))

    return list(merged_results.values())

def calculate_pagerank(json_list):
    # 创建有向图
    G = nx.DiGraph()

    # 添加节点和边，不会添加重复的节点和边
    for data in json_list:
        url = data['url']
        G.add_node(url)
        for related_url in data.get('link', []):
            G.add_edge(url, related_url)
        data.pop('link')

    # 计算 pagerank
    pageranks = nx.pagerank(G)

    # 将 pagerank 添加到数据中
    for data in json_list:
        url = data['url']
        data['pagerank'] = pageranks.get(url, 0.0)
    
    return json_list

# 定义一个函数用于加载中文停用词，输入参数是停用词文件的路径
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取文件中的停用词，并去除每行两端的空格和换行符
        stopwords = [line.strip() for line in file]
    return stopwords

# 定义一个函数用于对文本进行预处理，包括中文分词
def preprocess_text(text):
    # 使用结巴分词对文本进行中文分词
    words = jieba.cut(text)
    # 将分词结果拼接成一个字符串并返回
    return ' '.join(words)

# 定义一个函数用于对文本进行聚类和标签添加
def cluster_and_label(json_list, stopwords_path=stopwords_psth):
    # 加载中文停用词
    stopwords = load_stopwords(stopwords_path)

    # 提取每个json数据的标题和简介，进行中文分词
    texts = [preprocess_text(item.get('title', '') + "\n" + item.get('intro', '')) for item in json_list]

    # 使用TF-IDF向量化文本
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform(texts)

    # 自动选择聚类数目
    best_score = -1
    best_k = 0

    # 在指定范围内以步长为CLUSER_STRIDE进行聚类数目的搜索
    for k in range(CLUSER_MIN_NUM, CLUSER_MAX_NUM, CLUSER_STRIDE):  
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)

        # 更新最优聚类数和对应的Silhouette分数
        if score > best_score:
            best_score = score
            best_k = k

    # 在最优聚类数的左右CLUSER_STRIDE个数再进行搜索
    for k in range(best_k - CLUSER_STRIDE + 1, best_k + CLUSER_STRIDE):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        labels = kmeans.labels_
        score = silhouette_score(X, labels)

        # 更新最优聚类数和对应的Silhouette分数
        if score > best_score:
            best_score = score
            best_k = k
    
    # 打印最优聚类数
    print("best_k:", best_k)

    # 使用最佳聚类数目进行聚类
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_

    # 为每个聚类添加标签
    for item, label in zip(json_list, labels):
        item["tags"].append(f'type{label+1}')

    # 返回带有聚类标签的json数据列表
    return json_list

if __name__ == '__main__':
    if os.path.exists(data_path):
        os.remove(data_path)
        
    # 标签页爬取
    book_list_url_list = get_list_url(book_tag_url)
    music_list_url_list = get_list_url(music_tag_url)
    list_url_list = book_list_url_list + music_list_url_list
    print("get_list_url end!")
    
    # 目录页爬取
    detail_url_list = get_detail_url(list_url_list)
    random.shuffle(detail_url_list)
    print("len:", len(detail_url_list))
    print("get_detail_url end!")

    # 详情页爬取
    json_list = get_data_list(detail_url_list)
    print("get_data end!")

    # 计算 pagerank
    json_list = calculate_pagerank(json_list)
    print("calculate_pagerank end!")

    # 聚类，计算tags
    # json_list = []
    # with open(data_path, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()
    #     for line in lines:
    #         json_list.append(json.loads(line))
    json_list = cluster_and_label(json_list)
    print("cluster_and_label end!")
    
    # 将每个 JSON 对象写入一行
    with open(data_path, 'w', encoding='utf-8') as file:
        for json_obj in json_list:
            json.dump(json_obj, file, ensure_ascii=False)
            file.write('\n')  # 换行符用于分隔每个 JSON 对象
    print("save_data end!")