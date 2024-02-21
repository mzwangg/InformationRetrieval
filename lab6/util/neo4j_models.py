# -*- coding: utf-8 -*-
from py2neo import Graph, NodeMatcher


class Neo4j_Handle():
    graph = None
    matcher = None
    base_movie = {}

    def __init__(self):
        print("Neo4j Init ...")

    def connectNeo4j(self):
        self.graph = Graph("bolt: // localhost:7687", auth=("neo4j", "12345678"))
        self.matcher = NodeMatcher(self.graph)

    def process_ans(self, answer, url_valie = True):
        json_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            result["source"] = rel.start_node['name']
            result['rel_type'] = list(rel.types())[0]
            result['target'] = rel.end_node['name']
            if url_valie:
                result['url'] = rel.start_node['url']
            json_list.append(result)

        return json_list

    # 一.实体查询
    def get_entity_info(self, name) -> list:
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where source.name = $name " +
            "return rel ", name=name).data()
        return self.process_ans(answer)

    # 二.关系查询
    # 1.关系查询:实体1(与实体1有直接关系的实体与关系)
    def find_relation_by_entity1(self, entity1):
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where source.name = $name " +
            "return rel ", name=entity1).data()
        return self.process_ans(answer)

    # 2.关系查询：实体2
    def find_relation_by_entity2(self, entity1):
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where target.name = $name " +
            "return rel ", name=entity1).data()
        return self.process_ans(answer)

    # 3.关系查询：实体1+关系
    def find_other_entities1(self, entity1, relation):
        answer = self.graph.run(
            "match (source)-[rel:" + relation + "]->(target)  where source.name = $name " +
            "return rel ", name=entity1).data()
        return self.process_ans(answer)

    # 4.关系查询：关系+实体2
    def find_other_entities2(self, entity2, relation):

        answer = self.graph.run(
            "match (source)-[rel:" + relation + "]->(target)  where target.name = $name " +
            "return rel ", name=entity2).data()
        return self.process_ans(answer)

    # 5.关系查询：实体1+实体2
    def find_relation_by_entities(self, entity1, entity2):
        answer = self.graph.run(
            "match (source)-[rel]-(target)  where source.name= $name1 and target.name = $name2 " +
            "return rel ", name1=entity1, name2=entity2).data()
        return self.process_ans(answer)

    # 6.关系查询：实体1+关系+实体2(实体-关系->实体)
    def find_entity_relation(self, entity1, relation, entity2):
        answer = self.graph.run(
            "match (source)-[rel:" + relation + "]->(target)  where source.name= $name1 and target.name = $name2 " +
            "return rel ", name1=entity1, name2=entity2).data()
        return self.process_ans(answer)

    # 三.电影推荐
    #评分函数
    def recommend_sort_key(self, movie):
        # 得到节点对应的字典
        base_movie_dict = self.base_movie['n']
        movie_dict = movie['n']
        # 初始化得分
        score = 0

        # 根据属性的相似程度计算得分
        # 属性：主演、导演、编剧、类型、制片国家/地区、语言、上映年份
        attr_list1 = ['actor', 'director', 'creator', 'type', 'country', 'language', 'date']
        for attr in attr_list1:
            set1 = set(base_movie_dict[attr].split('/'))
            set2 = set(movie_dict[attr].split('/'))
            local_score = len(set1.intersection(set2))
            score += local_score

        return score * float(movie_dict['mark'])

    # 电影推荐
    def get_movie_recommend(self, name):
        # 进行查询
        movies = self.graph.run("match (n:电影) return n", name=name).data()
        self.base_movie = self.graph.run("match (n:电影{name:$name}) return n", name=name).data()[0]

        #根据评分得到推荐的5个电影节点(之所以取前6个是因为还有电影本身)
        sorted_movies = sorted(movies, key=self.recommend_sort_key, reverse=True)[:6]

        # 存储答案
        answer_dict = {}
        answer_name = []
        answer_list = []

        # 遍历推荐电影，得到关系
        for movie in sorted_movies:
            answer = self.graph.run(
                "match (source)-[rel]-(target)  where source.name = $name " +
                "return rel ", name=movie['n']['name']).data()
            answer_list.extend(self.process_ans(answer, False))
            answer_name.append({'name':movie['n']['name'], 
                                       'url':answer[0]['rel'].start_node['url']})

        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []
        return answer_dict

    # 四.问答
    '''
	0:nm 评分
	1:nm 上映时间
	2:nm 类型
	3:nm 简介
	4:nm 演员列表
	5:nnt ng电影作品
	6:nnt 电影作品
	7:nnt 参演评分大于 x
	8:nnt 参演评分小于 x
	9:nnt 电影类型
	10:nnt nnr合作电影列表
	11:nnt 电影数量
	12:评分大于x电影
	13:评分大于x的ng类型电影
	14:与nm导演相同的其他电影
	
	（nm:电影名 nnt:演员名 ng:电影类型）  
	'''

    # 0:nm 评分
    def movie_mark(self, name):
        answer = self.graph.run(
            "match (m:电影) where m.name = $name return m.mark as mark", name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            relation_type = '评分'
            start_name = name
            end_name = an['mark']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append(str(end_name) + ' 分')
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 1:nm 上映时间
    def movie_showtime(self, name):
        answer = self.graph.run(
            "match (m:电影) where m.name = $name return m.date as showtime ", name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            relation_type = '上映时间'
            start_name = name
            end_name = an['showtime']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name', end_name})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 2: nm 类型
    def movie_category(self, name):
        answer = self.graph.run(
            "match (m:电影) where m.name = $name return m.type as category", name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        answer = answer[0]['category'].split('/')
        for an in answer:
            result = {}
            relation_type = '类型'
            start_name = name
            end_name = an
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name', end_name})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 3:nm 简介
    def movie_info(self, name):
        answer = self.graph.run(
            "match (m:电影) where m.name = $name return m.introduction as info", name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            relation_type = '简介'
            start_name = name
            end_name = an['info']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name', end_name})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 4:nm 演员列表
    def movie_actors(self, name):
        answer = self.graph.run(
            "match (m:电影) where m.name = $name return m.actor as actors", name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        answer = answer[0]['actors'].split('/')
        for an in answer:
            result = {}
            relation_type = '主演'
            start_name = name
            end_name = an
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name', end_name})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 5:nnt ng电影作品
    def actor_category_movie(self, name, category):
        answer = self.graph.run(
            "MATCH (m:电影)-[rel:主演]->(p:主演) where p.name='" + name + "' and m.type=~'.*" + category + ".*' return rel").data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            result['url'] = rel.start_node['url']
            answer_list.append(result)
            answer_name.append({'name':start_name, 
                                'url':rel.start_node['url']})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 6:nnt 电影作品
    def actor_movie(self, name):
        answer = self.graph.run(
            "MATCH (m:电影)-[rel:主演]->(p:主演) where p.name = $name return rel", name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            result['url'] = rel.start_node['url']
            answer_list.append(result)
            answer_name.append({'name':start_name, 
                    'url':rel.start_node['url']})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 7:nnt 参演评分大于 x
    def actor_gt_mark_movie(self, name, mark):
        answer = self.graph.run(
            "MATCH (m:电影)-[rel:主演]->(p:主演) where toFloat(m.mark)>$mark and p.name=$name return rel", mark=myToFloat(mark),
            name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            result['url'] = rel.start_node['url']
            answer_list.append(result)
            answer_name.append({'name':start_name, 
                                'url':rel.start_node['url']})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 8:nnt 参演评分小于 x
    def actor_lt_mark_movie(self, name, mark):
        answer = self.graph.run(
            "MATCH (m:电影)-[rel:主演]->(p:主演) where toFloat(m.mark)<$mark and p.name=$name return rel", mark=myToFloat(mark),
            name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            rel = an['rel']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            result['url'] = rel.start_node['url']
            answer_list.append(result)
            answer_name.append({'name':start_name, 
                                'url':rel.start_node['url']})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 9:nnt 电影类型
    def actor_movie_category(self, name):
        answer = self.graph.run(
            "match (m:电影)-[:主演]->(p:主演{name:$name}) return p.name as name, m.type as category",
            name=name).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        movie_cat_set = set()
        name = ''
        for an in answer:
            name = an['name']
            category = an['category']
            category = category.split('/')
            for cat in category:
                movie_cat_set.add(cat)

        for cat in movie_cat_set:
            result = {}
            relation_type = '出演的电影风格'
            start_name = name
            end_name = cat

            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name', end_name})

        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 10:nnt nnr合作电影列表
    def actor_actor_movie(self, name1, name2):
        answer = self.graph.run(
            "match (p:主演{name:$pname})<-[rel1:主演]-(m:电影)-[rel2:主演]->(other:主演{name:$oname}) return rel1, rel2",
            pname=name1,
            oname=name2).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            rel = an['rel1']
            relation_type = list(rel.types())[0]
            start_name = rel.start_node['name']
            end_name = rel.end_node['name']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name':start_name,
                                'url':rel.start_node['url']})

            rel2 = an['rel2']
            relation_type2 = list(rel2.types())[0]
            start_name2 = rel2.start_node['name']
            end_name2 = rel2.end_node['name']
            result2 = {}
            result2["source"] = {'name': start_name2}
            result2['type'] = relation_type2
            result2['target'] = {'name': end_name2}
            answer_list.append(result2)

        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 11:nnt 电影数量
    def actor_movie_count(self, name):
        answer = self.graph.run(
            "match (m:电影)-[:主演]->(p:主演) where p.name = $name return p.name as name, count(m) as count",
            name=name).data()
        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            relation_type = '电影数量'
            start_name = an['name']
            end_name = an['count']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': str(end_name)}
            answer_list.append(result)
            answer_name.append({'name', str(end_name)})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 12:评分大于x电影
    def gt_mark_movie(self, mark):
        answer = self.graph.run(
            "MATCH (m:电影) where toFloat(m.mark)>$mark return m", mark=myToFloat(mark)).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            relation_type = '评分'
            start_name = an['m']['name']
            end_name = an['m']['mark']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name':start_name,
                                'url':an['m']['url']})
        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []

        return answer_dict

    # 13:评分大于x的ng类型电影
    def gt_mark_category_movie(self, mark, category):
        answer = self.graph.run(
            "MATCH (m:电影) where toFloat(m.mark)>$mark and m.type =~'.*" + category + ".*' return m",
            mark=myToFloat(mark)).data()

        answer_dict = {}
        answer_name = []
        answer_list = []
        for an in answer:
            result = {}
            relation_type = '评分'
            start_name = an['m']['name']
            end_name = an['m']['mark']
            result["source"] = {'name': start_name}
            result['type'] = relation_type
            result['target'] = {'name': end_name}
            answer_list.append(result)
            answer_name.append({'name':start_name,
                                'url':an['m']['url']})

            result = {}
            relation_type_2 = '类型'
            end_name_2 = an['m']['type']
            result["source"] = {'name': start_name}
            result['type'] = relation_type_2
            result['target'] = {'name': end_name_2}
            answer_list.append(result)

        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []
        return answer_dict

    # 14:与nm导演相同的其他电影
    def same_director_movie(self, name):
        directors = self.graph.run(
            "match (m:电影) where m.name = $name return m.director as director",name=name).data()
        director_list=directors[0]['director'].split('/')

        answer_dict = {}
        answer_name = []
        answer_list = []

        for director in director_list:
            answer = self.graph.run(
                "match (m:电影)-[导演]->(n:导演) where n.name = $name return m", name = director).data()

            for an in answer:
                result = {}
                relation_type = '导演'
                start_name = an['m']['name']
                end_name = director
                result["source"] = {'name': start_name}
                result['type'] = relation_type
                result['target'] = {'name': end_name}
                answer_list.append(result)
                answer_name.append({'name':start_name,
                                    'url':an['m']['url']})

        answer_dict['answer'] = answer_name
        answer_dict['list'] = answer_list

        if len(answer_name) == 0:
            return []
        return answer_dict

#将字符串转为float
def myToFloat(input):
    try:
        return float(input)
    except ValueError:
        return input