import json
import random
import os
import warnings
import json
import os
from flask import Flask, render_template, redirect, url_for, request, session, jsonify
from search import my_search, my_recommend
from datetime import datetime
from log import write_log

index_tag_dict = {}
index_title_dict = {}

warnings.filterwarnings("ignore")

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
project_path = os.path.dirname(os.path.abspath(__file__))
user_path = os.path.join(project_path, "static" + os.sep + "user.json")
data_path = os.path.join(project_path, "static" + os.sep + "data1.json")

app = Flask(__name__)
app.secret_key = '123456'

# 读取用户信息
def read_users():
    try:
        with open(user_path, 'r', encoding='utf-8') as file:
            users = json.load(file)
    except FileNotFoundError:
        users = {}
    return users

# 写入用户信息
def write_users(users):
    with open(user_path, 'w', encoding='utf-8') as file:
        json.dump(users, file, ensure_ascii=False, indent=2)

# 登录页面路由
@app.route('/', methods=['GET', 'POST'])
def login():
    # 处理POST请求
    if request.method == 'POST':
        # 获取表单提交的用户名和密码
        username = request.form['username']
        password = request.form['password']
        
        # 读取用户信息（这里假设read_users()是一个函数用于读取用户数据）
        users = read_users()
        
        # 检查用户名和密码是否匹配
        # 如果匹配，将用户名存入会话(session)中， 然后重定向到搜索页面
        if username in users and users[username]['password'] == password:
            session['username'] = username
            return redirect(url_for('search'))

    # 渲染登录页面模板（login.html）
    return render_template('login.html')


# 注册页面路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    # 处理POST请求
    if request.method == 'POST':
        # 获取表单提交的用户名和密码
        username = request.form['username']
        password = request.form['password']
        
        # 读取已注册用户数据，并将用户名、密码存入用户字典，然后保存
        users = read_users()
        users[username] = {'password': password, 'tags': {}}
        write_users(users)
        
        # 注册成功后重定向到登录页面
        return redirect(url_for('login'))

    # 渲染注册页面模板（register.html）
    return render_template('register.html')


# 搜索页面，需要登录才能访问
@app.route('/search', methods=['GET', 'POST'])
def search():
    if 'username' in session:
        if request.method == 'POST':
            # 获取表单提交的数据
            time_begin = request.form.get('time_begin', None)
            time_begin = datetime.strptime(time_begin, '%Y-%m-%d').date() if time_begin else None

            time_end = request.form.get('time_end', None)
            time_end = datetime.strptime(time_end, '%Y-%m-%d').date() if time_end else None

            web = request.form.get('web', '')

            domain = request.form.get('domain', 'all')

            pattern = request.form.get('pattern', 'match')
            
            querys = request.form.get('querys', '').split(' ')
            
            # 调用 my_search 函数进行搜索
            users = read_users()
            search_results = my_search(time_begin, time_end, web, domain, pattern, users[session['username']]['tags'], querys)
            recommend_results = my_recommend(search_results, users[session['username']]['tags'])
            write_log(session['username'], 'search ' + request.form.get('querys', ''))
            
            # 返回搜索结果
            return render_template('search.html', username=session['username'], search_results=search_results, recommend_results = recommend_results)

        return render_template('search.html', username=session['username'])
    else:
        return redirect(url_for('login'))
    
@app.route('/process-url', methods=['POST'])
def process_url():
    try:
        # 从请求中获取JSON数据
        data = request.get_json()
        url = data.get('url')

        # 读取用户数据
        users = read_users()
        tags = users[session['username']]['tags']

        # 从URL中提取索引号
        index = url.split('/')[-2]

        # 更新用户标签信息
        for item in index_tag_dict[index]:
            if item in tags:
                tags[item] += 2
            else:
                tags[item] = 2
        write_users(users)

        # 记录用户行为到日志
        write_log(session['username'], 'click ' + index_title_dict[index] + ' ' + index)

        # 返回一个渲染模板的响应，展示网页内容
        return render_template('show_webpage.html', url=url)
    except Exception as e:
        # 处理异常情况，返回JSON格式的错误信息
        return jsonify({'error': str(e)})


@app.route('/show_snapshot/<snapshotIndex>')
def display_snapshot(snapshotIndex):
    users = read_users()
    tags = users[session['username']]['tags']
    for item in index_tag_dict[snapshotIndex]:
        if item in tags:
            tags[item] += 1
        else:
            tags[item] = 1
    write_users(users)

    write_log(session['username'], 'sanpshot ' + index_title_dict[snapshotIndex] + ' ' + snapshotIndex)

    image_path = url_for('static', filename=f'snapshot/{snapshotIndex}.png')
    return render_template('show_snapshot.html', image_path=image_path)

# 退出登录
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    if not os.path.exists(user_path):
        with open(user_path, 'w', encoding='utf-8') as file:
            json.dump({'user1': {'password': '111111', 'tags':{}}, 
                       'user2': {'password': '222222', 'tags':{}}}, 
                       file, ensure_ascii=False, indent=2)

    with open(data_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            json_obj = json.loads(line)
            index_tag_dict[json_obj['url'].split('/')[-2]] = json_obj['tags']
            index_title_dict[json_obj['url'].split('/')[-2]] = json_obj['title']

    app.run(debug=True)