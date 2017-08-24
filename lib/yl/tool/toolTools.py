# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import re

def filterList(strs, key):
    '''
    对一个str列表 找出其中存在字符串 key的所有元素
    '''
    return list(filter((lambda strr: key in strr),strs))

def findint(strr):
    '''
    返回字符串或字符串列表中的所有整数
    '''
    if isinstance(strr,(list,tuple)):
        return list(map(findint, strr))
    return list(map(int,re.findall(r"\d+\d*",strr)))
         
if __name__ == "__main__":
     
    string=["A001.45，b5，6.45，8.82",'sd4 dfg77']
    print findint(string)
    pass