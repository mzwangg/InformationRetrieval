# 导入所需模块
import pandas as pd
import requests
from lxml import etree
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import time
import re
import warnings
import random
import os
import configparser
import json
from tqdm import tqdm

# 忽略warning
warnings.filterwarnings("ignore")

# 随机的user-agent信息
ua = UserAgent()
request_headers = {
    'user-agent':ua.random,
    'Referer': 'https://www.douban.com'
}

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, "spider_data.json")
config_path = os.path.join(dir_path, "config.ini")

config = configparser.ConfigParser()
config.read(config_path)
MIN_SLEEP_TIME = float(config['DEFAULT']['MIN_SLEEP_TIME'])
MAX_SLEEP_TIME = float(config['DEFAULT']['MAX_SLEEP_TIME'])
CATALOG_PAGE_NUM = int(config['DEFAULT']['CATALOG_PAGE_NUM'])

def get_per_page_newslinks(catalog_urls):
    detail_urls = []
    for url in tqdm(catalog_urls, total = len(catalog_urls), desc="Processing catalog_urls"):
        time.sleep(random.uniform(MIN_SLEEP_TIME, MAX_SLEEP_TIME)) # 随机休眠
        web_data = requests.get(url=url, headers=request_headers)
        status_code = web_data.status_code
        if status_code == 200:
            html_text = web_data.text
            # 解析为 soup 文档
            soup_document = BeautifulSoup(html_text, 'html.parser')
            # 定位到所有超链的 a 标签
            homepage_a_elements = soup_document.select('#content > div > div.article > ol > li > div > div.info > div.hd > a')
            # 遍历每一条电影
            for news_homepage_a_element in homepage_a_elements:
                # 提取href属性值
                url = news_homepage_a_element.get('href').strip()
                detail_urls.append(url)       
    return detail_urls

def get_infos(detail_urls):
    datas = []
    for url in tqdm(detail_urls, total = len(detail_urls), desc="Processing detail_urls"):
        web_data = requests.get(url=url, headers=request_headers)
        # 需要说明编码，否则会出乱码
        web_data.encoding = 'utf-8'
        status_code = web_data.status_code
        if status_code == 200:
            html_text = web_data.text
            # 使用 lxml 解析为 XPath 可定位的树状文档
            selector = etree.HTML(html_text)
            soup_document = BeautifulSoup(html_text, 'html.parser')

            data = {}
            data['url'] = url

            # 获取名字
            if len(selector.xpath('//*[@id="content"]/h1/span[1]/text()')) != 0:
                title = selector.xpath('///*[@id="content"]/h1/span[1]/text()')[0].strip()
                data['title']=title
            else:
                return

            # 导演
            author_name_element = selector.xpath('//*[@id="info"]/span[1]/span[2]/a/text()')
            director = ''
            for e in author_name_element:
                director = director + e + '/'
            else:
                director = director[:-1] + ''
            # 使用正则表达式，去除多余的空格
            # 1. 指定匹配规则：匹配所有的空白符
            p = re.compile('\s+')
            # 2. 删除==替换为 ''
            director_name = re.sub(p, '', director)
            data['director'] = director_name

            # 编剧
            cre_name_element = selector.xpath('//*[@id="info"]/span[2]/span[2]/a/text()')
            creator = ''
            for e in cre_name_element:
                creator = creator + e + '/'
            else:
                creator = creator[:-1] + ''
            # author_name = author_name_element[0].strip() if author_name_element != 0 else "来源不明，或许是非常规网站"
            # 使用正则表达式，去除多余的空格
            # 1. 指定匹配规则：匹配所有的空白符
            p = re.compile('\s+')
            # 2. 删除==替换为 ''
            cre_name = re.sub(p, '', creator)
            data['creator'] = cre_name

            # 主演
            act_name_element = soup_document.find_all(name='a',rel='v:starring')
            actor = ''
            for e in act_name_element:
                e = e.get_text().strip()
                actor = actor + e + '/'
            # 使用正则表达式，去除多余的空格
            # 1. 指定匹配规则：匹配所有的空白符
            p = re.compile('\s+')
            # 2. 删除==替换为 ''
            act_name = re.sub(p, '', actor)
            data['actor'] = act_name

            # 获取类型
            typ_element = soup_document.find_all(name='span',property='v:genre')
            tyss = ''
            for e in typ_element:
                et = e.get_text().strip()
                tyss = tyss + et + '/'
            else:
                tyss = tyss[:-1] + ''
            p = re.compile('\s+')
            typ_name = re.sub(p, '', tyss)
            data['type'] = typ_name

            # 获取国家和语言
            country = selector.xpath('//span[contains(text(), "制片国家或地区:")]')[0].tail.strip()
            lan = selector.xpath('//span[contains(text(), "语言:")]')[0].tail.strip()
            p = re.compile('\s+')
            cou_name = re.sub(p, '', country)
            lan_name = re.sub(p, '', lan)
            data['country'] = cou_name
            data['language'] = lan_name

            # 获取上映日期
            date_element = soup_document.find_all(name='span', property='v:initialReleaseDate')
            date = ''
            for e in date_element:
                et = e.get_text().strip().replace(' ','')
                date = date + et + '/'
            else:
                date = date[:-1] + ''
            p = re.compile('\s+')
            date = re.sub(p, '', date)
            data['date'] = date

            # 获取评分
            mark = soup_document.select('#interest_sectl > div.rating_wrap.clearbox > div.rating_self.clearfix > strong')[0].get_text().strip().replace(' ','')
            data['mark'] = float(mark)

            # 获取完整简介
            op = soup_document.find(name='a',class_='j a_show_full')
            if(op):
                intro = soup_document.find(name='span',class_='all hidden').get_text().strip().replace(' ','')
            else:
                intro = soup_document.find(name='span',property="v:summary").get_text().strip().replace(' ','')
            data['intro'] = intro

            datas.append(data)
    return datas

# 声明主程序入口
if __name__ == '__main__':
    if os.path.exists(data_path):
        os.remove(data_path)

    # 声明所有的列表页的 urls list
    catalog_urls = ['https://movie.douban.com/top250?start={}'.format(str(i * 25)) for i in range(CATALOG_PAGE_NUM)]

    # 遍历每一个列表页，调用方法，获取列表页中的信息
    detail_urls = get_per_page_newslinks(catalog_urls)

    # 遍历每一个详情页，提取信息
    datas = get_infos(detail_urls)
    
    # 将每个 JSON 对象写入一行
    with open(data_path, 'w', encoding='utf-8') as file:
        for data in datas:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')  # 换行符用于分隔每个 JSON 对象
    print("save_data end!")