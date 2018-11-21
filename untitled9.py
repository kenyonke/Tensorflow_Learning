# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 17:18:23 2018

@author: kenyon
"""

import jieba

test = list(jieba.cut("我来到北京清华大学", cut_all=True))
print(test)

test = list(jieba.cut("我来到北京清华大学", cut_all=False))
print(test)

test = list(jieba.cut("他来到了网易杭研大厦"))  # 默认是精确模式
print(test)

test = list(jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造"))# 搜索引擎模式
print(test)
print('')


#================================================================================================
#增加，删除 词
jieba.add_word('石墨烯')
jieba.add_word('凱特琳')
jieba.del_word('自定义词')

test_sent = (
"李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿\n"
"例如我输入一个带“韩玉赏鉴”的标题，在自定义词库中也增加了此词为N类\n"
"「台中」正確應該不會被切開。mac上可分出「石墨烯」；此時又可以分出來凱特琳了。"
)
words = jieba.cut(test_sent)
print('/'.join(words))

terms = jieba.cut('easy_install is great')
print('/'.join(terms))
terms = jieba.cut('python 的正则表达式是好用的')
print('/'.join(terms))
print('')
#添加自定义词典
#jieba.load_userdict(file_name)


#================================================================================================
#返回词语在原文的起止位置 tokenzize
#default model
result = jieba.tokenize(u'永和服装饰品有限公司')
for tk in result:
    print((tk[0],tk[1],tk[2]))
print('----')
#search model
result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
for tk in result:
    print((tk[0],tk[1],tk[2]))
print('')


#================================================================================================    
#词性标注
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
   print('%s %s' % (word, flag))


#================================================================================================    
#并行分词
#原理：将目标文本按行分隔后，把各行文本分配到多个 Python 进程并行分词，然后归并结果，从而获得分词速度的可观提升
#！！！不支持windows
#jieba.enable_parallel(4) # 开启并行分词模式，参数为并行进程数
#jieba.disable_parallel() # 关闭并行分词模式
