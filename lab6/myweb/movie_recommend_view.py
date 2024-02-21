# -*- coding: utf-8 -*-
from django.shortcuts import render
from util.preload import neo4jconn

#电影实体查询
def get_recommend(request):
    context = {'ctx': ''}
    if (request.GET):
        base_movie_name = request.GET['base_movie_name'].strip().lower()
        ret_dict = neo4jconn.get_movie_recommend(base_movie_name)

        if (len(ret_dict) != 0):
            return render(request, 'movie_recommend.html', {'ret': ret_dict})

        return render(request, 'movie_recommend.html', {'ctx': '暂未找到答案'})

    return render(request, 'movie_recommend.html', context)