from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q, SF
import os
import configparser
import warnings

warnings.filterwarnings("ignore")

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
project_path = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(project_path, "config.ini")

# 读取配置文件
config = configparser.ConfigParser()
config.read(config_path)
MAX_RESULT_NUM = int(config['DEFAULT']['MAX_RESULT_NUM'])
RECOMMEND_BASE_NUM = int(config['DEFAULT']['RECOMMEND_BASE_NUM'])
RECOMMEND_NUM = int(config['DEFAULT']['RECOMMEND_NUM'])

def my_search(time_begin=None, time_end=None, web='', domain='all', pattern='match', user_interests ={}, querys=[]):
    # 初始化 Elasticsearch 客户端
    es = Elasticsearch("http://localhost:9200")

    # 构建 Elasticsearch 的查询语句
    s = Search(using=es, index='douban_index')
    s = s.extra(size=100)

    # 构建时间范围查询，实现指定时间范围的搜索
    if time_begin is not None:
        s = s.filter('range', date={'gte': time_begin})
    if time_end is not None:
        s = s.filter('range', date={'lte': time_end})

    # 构建包含域名的查询，实现站内查询
    if web:
        s = s.filter('regexp', url=f'.*{web}.*')

    # 根据pattern选择搜索的模式
    search_type = 'match'
    if pattern == "regexp":
        search_type = 'regexp'
    elif pattern == "wildcard":
        search_type = 'wildcard'
    elif pattern == "phrase":
        search_type = 'match_phrase'

    # 构建domain，在对应字段范围进行查询
    if domain == 'title':
        must_queries = [Q(search_type, title=query) for query in querys]
    elif domain == 'intro':
        must_queries = [Q(search_type, intro=query) for query in querys]
    elif domain == 'author':
        must_queries = [Q(search_type, author=query) for query in querys]
    else:
        must_queries = [Q(search_type, title=query) | 
                        Q(search_type, intro=query) |
                        Q(search_type, author=query) for query in querys]
    bool_query = Q('bool', must=must_queries)

    # 构建脚本评分，优化查询结果，进行个性化查询
    script_source = """
        double interestsScore = 1.0;
        for (tag in doc['tags']) {
            // 如果用户兴趣中包含当前标签，增加兴趣得分
            if (params.userInterests.containsKey(tag)) {
                interestsScore += params.userInterests[tag] * 0.1;
            }
        }
        
        // 防御性检查，避免零或负数
        double grade = Math.max(doc['grade'].value, 1) * 0.1;  // 避免零
        double peopleLog = Math.log(doc['people'].value + 1); // 避免零
        double pagerank = Math.max(doc['pagerank'].value, 1);  // 避免零

        // 返回最终评分，考虑兴趣、评分、人数、PageRank等因素
        return _score * interestsScore * grade * peopleLog * pagerank;
    """
    
    # 使用elasticsearch_dsl构建查询
    script_score = SF('script_score', script={"source": script_source, "params": {"userInterests": user_interests}})
    s = s.query(Q('function_score', query=bool_query, functions=[script_score]))

    # 执行搜索
    response = s.execute()

    # 处理搜索结果
    search_results = []
    for hit in response:
        search_results.append(hit.to_dict())

    return search_results

def my_recommend(search_results, user_interests={}):
    # 统计搜索结果中，前RECOMMEND_BASE_NUM歌结果标签的出现次数
    tags_dict = {}
    for search_result in search_results[:RECOMMEND_BASE_NUM]:
        for tag in search_result['tags']:
            if tag in tags_dict:
                tags_dict[tag] += 1
            else:
                tags_dict[tag] = 1

    # 初始化 Elasticsearch 客户端
    es = Elasticsearch("http://localhost:9200")  # 根据实际情况设置 Elasticsearch 地址和端口

    # 构建 Elasticsearch 的查询语句
    s = Search(using=es, index='douban_index')  # 将 'your_index_name' 替换为你的索引名称
    s = s.extra(size=100)

    # 构建 Elasticsearch 的脚本评分源码
    script_source = """
        double tagsScore = 0.001;
        for (tag in doc['tags']) {
            if (params.tags_dict.containsKey(tag)) {
                tagsScore += params.tags_dict[tag];
            }
        }

        double interestsScore = 1.0;
        for (tag in doc['tags']) {
            if (params.userInterests.containsKey(tag)) {
                interestsScore += params.userInterests[tag] * 0.1;
            }
        }

        // 防御性检查，避免零或负数
        double grade = Math.max(doc['grade'].value, 1) * 0.1;  // 避免零
        double peopleLog = Math.log(doc['people'].value + 1); // 避免零
        double pagerank = Math.max(doc['pagerank'].value, 1);  // 避免零

        return _score * interestsScore * grade * peopleLog * pagerank * tagsScore;
    """

    # 构建 Elasticsearch 的脚本评分
    script_score = SF('script_score', script={"source": script_source, "params": {"userInterests": user_interests, "tags_dict": tags_dict}})
    q = Q("wildcard", title="*")
    s = s.query(Q('function_score', query=q, functions=[script_score]))

    # 执行搜索
    response = s.execute()

    # 处理搜索结果，如果没在搜索结果中，则放在推荐结果中
    recommend_results = []
    for hit in response:
        hit_dict = hit.to_dict()
        if hit_dict not in search_results:
            recommend_results.append(hit_dict)

    # 返回前 RECOMMEND_NUM 个推荐结果
    return recommend_results[:RECOMMEND_NUM]


if __name__ == '__main__':
    search_results = my_search(domain='all', pattern='wildcard', querys=['*'])
    recommend_results = my_recommend(search_results)
    print(len(search_results))
    print(len(recommend_results))
    for result in recommend_results:
        print(result["tags"])