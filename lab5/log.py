import os
import warnings

warnings.filterwarnings("ignore")

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
project_path = os.path.dirname(os.path.abspath(__file__))
log_dir_path = os.path.join(project_path, "static" + os.sep + "log")

def write_log(username, log):
    # 检查 log_dir_path 是否存在，不存在则创建
    if not os.path.exists(log_dir_path):
        os.mkdir(log_dir_path)

    # 检查 log_path 是否存在，不存在则创建
    log_path = log_dir_path + os.sep + f"{username}.txt"
    if not os.path.exists(log_path):
        with open(log_path, 'w', encoding="utf8") as file:
            file.write('')

    # 将 log 写入到文件中
    with open(log_path, 'a', encoding="utf8") as file:
        file.write(log + '\n')