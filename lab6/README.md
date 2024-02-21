# 启动步骤
1、打开neo4jl服务

2、在data文件夹运行load_neo4j_data.py以导入neo4j数据

3、将data\自定义词典文件夹下的txt文件导入pyhanlp的词典中，并更改属性CustomDictionaryPath如下：

```
	# CustomDictionaryPath=data/dictionary/custom/CustomDictionary.txt; 现代汉语补充词库.txt; 全国地名大全.txt ns; 人名词典.txt; 机构名词典.txt; 上海地名.txt ns;data/dictionary/person/nrf.txt nrf;
	CustomDictionaryPath=data/dictionary/custom/CustomDictionary.txt; 现代汉语补充词库.txt; 全国地名大全.txt ns; 人名词典.txt; 机构名词典.txt; 上海地名.txt ns; 电影类型.txt ng; 电影名.txt nm; 演员名.txt nnt; other.txt;data/dictionary/person/nrf.txt nrf;
```

4、运行intent_classification文件夹下的train.py，并将生成的pkl和.h5文件复制到myweb/util里

5、在myweb文件夹中运行run.bat，启用django提供的web服务器

# require:

1.neo4j-community-5.9.0

2.pyhanlp 0.1.84

3.django 2.0.3

4.py2neo  2021.2.3

5.pytorch 1.8.0

6.torchtext 0.6.0