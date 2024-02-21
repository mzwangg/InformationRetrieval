# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from django.shortcuts import render
from util.preload import segment, neo4jconn, model_dict

model, words = model_dict
model.eval()

def question_answering(request):
    context = {'ctx': ''}
    if (request.GET):
        question = request.GET['question'].strip().lower()
        word_nature = segment.seg(question)
        print('word_nature:{}'.format(word_nature))
        classfication_num = chatbot_response(word_nature)
        print('类别：{}'.format(classfication_num))
        # 实体识别

        # 返回格式：答案和关系图{‘answer':[], 'result':[]}
        ret_dict = []
        if classfication_num == 0:
            for term in word_nature:
                if str(term.nature) == 'nm':
                    ret_dict = neo4jconn.movie_mark(term.word)
                    break

        elif classfication_num == 1:
            for term in word_nature:
                if str(term.nature) == 'nm':
                    ret_dict = neo4jconn.movie_showtime(term.word)
                    break

        elif classfication_num == 2:
            for term in word_nature:
                if str(term.nature) == 'nm':
                    ret_dict = neo4jconn.movie_category(term.word)
                    break

        elif classfication_num == 3:
            for term in word_nature:
                if str(term.nature) == 'nm':
                    ret_dict = neo4jconn.movie_info(term.word)
                    break

        elif classfication_num == 4:
            for term in word_nature:
                if str(term.nature) == 'nm':
                    ret_dict = neo4jconn.movie_actors(term.word)
                    break

        elif classfication_num == 5:
            name = ''
            category = ''
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    name = term.word
                elif str(term.nature) == 'ng':
                    category = term.word
                if (name != '') and (category != ''):
                    ret_dict = neo4jconn.actor_category_movie(name, category)
                    break

        elif classfication_num == 6:
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    ret_dict = neo4jconn.actor_movie(term.word)
                    break

        elif classfication_num == 7:
            name = ''
            mark = ''
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    name = term.word
                elif str(term.nature) == 'm':
                    mark = term.word
                if (name != '') and (mark != ''):
                    ret_dict = neo4jconn.actor_gt_mark_movie(name, mark)
                    break

        elif classfication_num == 8:
            name = ''
            mark = ''
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    name = term.word
                elif str(term.nature) == 'm':
                    mark = term.word
                if (name != '') and (mark != ''):
                    ret_dict = neo4jconn.actor_lt_mark_movie(name, mark)
                    break

        elif classfication_num == 9:
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    ret_dict = neo4jconn.actor_movie_category(term.word)
                    break

        elif classfication_num == 10:
            name1 = ''
            name2 = ''
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    if name1 == '':
                        name1 = term.word
                    else:
                        name2 = term.word
                if (name1 != '') and (name2 != ''):
                    ret_dict = neo4jconn.actor_actor_movie(name1, name2)
                    break

        elif classfication_num == 11:
            for term in word_nature:
                if str(term.nature) == 'nnt':
                    ret_dict = neo4jconn.actor_movie_count(term.word)
                    break

        elif classfication_num == 12:
            for term in word_nature:
                if str(term.nature) == 'm':
                    ret_dict = neo4jconn.gt_mark_movie(term.word)
                    break

        elif classfication_num == 13:
            mark = ''
            category = ''
            for term in word_nature:
                if str(term.nature) == 'm':
                    mark = term.word
                elif str(term.nature) == 'ng':
                    category = term.word

                if (mark != '') and (category != ''):
                    ret_dict = neo4jconn.gt_mark_category_movie(mark, category)
                    break

        elif classfication_num == 14:
            for term in word_nature:
                if str(term.nature) == 'nm':
                    ret_dict = neo4jconn.same_director_movie(term.word)
                    break

        if (len(ret_dict) != 0):
            return render(request, 'question_answering.html', {'ret': ret_dict})

        return render(request, 'question_answering.html', {'ctx': '暂未找到答案'})

    return render(request, 'question_answering.html', context)


# 分词，需要将电影名，演员名和评分数字转为nm，nnt，ng
def _sentence_segment(word_nature):
    sentence_words = []
    for term in word_nature:
        if str(term.nature) == 'nnt':
            sentence_words.append('nnt')
        elif str(term.nature) == 'nm':
            sentence_words.append('nm')
        elif str(term.nature) == 'ng':
            sentence_words.append('ng')
        elif str(term.nature) == 'm':
            sentence_words.append('x')
        else:
            sentence_words.append(term.word)
    return sentence_words


def _bow(word_nature):
    sentence_words = _sentence_segment(word_nature)
    # 词袋
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1  # 词在词典中
    return [bag]


def _predict_class(word_nature):
    sentence_bag = _bow(word_nature)
    with torch.no_grad():
        outputs = model(torch.FloatTensor(sentence_bag))
    predicted_prob, predicted_index = torch.max(F.softmax(outputs, 1), 1)  # 预测最大类别的概率与索引
    results = []
    results.append({'intent': predicted_index.detach().numpy()[0], 'prob': predicted_prob.detach().numpy()[0]})
    return results


def _get_response(predict_result):
    tag = predict_result[0]['intent']
    return tag


def chatbot_response(word_nature):
    predict_result = _predict_class(word_nature)
    res = _get_response(predict_result)
    return res
