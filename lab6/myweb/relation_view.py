# -*- coding: utf-8 -*-
from django.shortcuts import render
from util.preload import neo4jconn

import json

#电影实体查询
def search_entity(request):
    ctx = {}
    #根据传入的实体名称搜索出关系
    if(request.GET):
        entity = request.GET['user_text'].strip().lower()
        entity = ''.join(entity.split())

        entityRelation = neo4jconn.get_entity_info(entity)
        if len(entityRelation) == 0:
            #若数据库中无法找到该实体，则返回数据库中无该实体
            ctx= {'title' : '<h2>知识库中暂未添加该实体</h1>'}
            return render(request,'entity.html',{'ctx':json.dumps(ctx,ensure_ascii=False)})
        else:
            return render(request,'entity.html',{'entityRelation':json.dumps(entityRelation,ensure_ascii=False)})
    #需要进行类型转换
    return render(request, 'entity.html', {'ctx':ctx})

#关系查询
def search_relation(request):
    ctx = {}
    if(request.GET):
        # 实体1
        entity1 = request.GET['entity1_text'].strip().lower()
        entity1 = ''.join(entity1.split())

        # 关系
        relation = request.GET['relation_name_text']

        # 实体2
        entity2 = request.GET['entity2_text'].strip().lower()
        entity2 = ''.join(entity2.split())

        # 1.若只输入entity1,则输出与entity1有直接关系的实体和关系
        if(len(entity1) != 0 and len(relation) == 0 and len(entity2) == 0):
            searchResult = neo4jconn.find_relation_by_entity1(entity1)
            if(len(searchResult)>0):
                return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

        # 2.若只输入entity2则,则输出与entity2有直接关系的实体和关系
        if(len(entity2) != 0 and len(relation) == 0 and len(entity1) == 0):
            searchResult = neo4jconn.find_relation_by_entity2(entity2)
            if(len(searchResult)>0):
                return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

        # 3.若输入entity1和relation，则输出与entity1具有relation关系的其他实体
        if(len(entity1)!=0 and len(relation)!=0 and len(entity2) == 0):
            searchResult = neo4jconn.find_other_entities1(entity1,relation)
            if(len(searchResult)>0):
                return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

        # 4.若输入entity2和relation，则输出与entity2具有relation关系的其他实体
        if(len(entity2)!=0 and len(relation)!=0 and len(entity1) == 0):
            searchResult = neo4jconn.find_other_entities2(entity2,relation)
            if(len(searchResult)>0):
                return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

        # 5.若输入entity1和entity2,则输出entity1和entity2之间的关系
        if(len(entity1) !=0 and len(relation) == 0 and len(entity2)!=0):
            searchResult = neo4jconn.find_relation_by_entities(entity1,entity2)
            if(len(searchResult)>0):
                return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

        # 6.若输入entity1,entity2和relation,则输出entity1、entity2是否具有相应的关系
        if(len(entity1)!=0 and len(entity2)!=0 and len(relation)!=0):
            searchResult = neo4jconn.find_entity_relation(entity1,relation,entity2)
            if(len(searchResult)>0):
                return render(request,'relation.html',{'searchResult':json.dumps(searchResult,ensure_ascii=False)})

        # 7.若全为空
        if(len(entity1)!=0 and len(relation)!=0 and len(entity2)!=0 ):
            pass

        ctx= {'title' : '<h1>暂未找到相应的匹配</h1>'}
        return render(request,'relation.html',{'ctx':ctx})

    return render(request,'relation.html',{'ctx':ctx})
