# -*- coding: utf-8 -*-
import os


from yllibInterface import dicto, glob, imread, imsave
from yllibInterface import show, loga, logl

def getNamesAndFormat(globpath):
    paths = glob(globpath)
    rind = 1-len(globpath)+(globpath.rindex('*'))
    imgPathFormat = globpath[:rind-1]+'%s'+globpath[rind:]
    names = sorted([os.path.basename(p)[:rind] for p in paths])
    return names,imgPathFormat


 
def setMod(mod='train'):
    c.mod = mod
    if mod == 'train':
        setTrain()
    if mod == 'test':
        setTest()
    if mod == 'predict':
        setPredict()

#    global name, img
#    name = c.names[0]
#    img = readimg(name)
    



cf = dicto()

c = dicto()
#from lib import cf,c
#
#cf.trainGlob =  u'G:\\experiment\\Data\\HKU-IS\\Imgs\\*.jpg'
#cf.toGtPath = lambda path:path.replace('.jpg','.png')
#cf.args = {}
#
#cf.test = 0.1
#cf.toTestGtPath = None
#
#cf.predictGlob = u'G:\\experiment\\Data\\HKU-IS\\Imgs\\*.jpg'
#
#cf.toTestGtPath = cf.toTestGtPath or cf.toGtPath



def setTest():
    if c.test and isinstance(c.test,(float,int)):
        c.allNames,c.imgPathFormat = getNamesAndFormat(c.trainGlob)
        n = len(c.allNames)
        splitAt = n - (int(n*c.test) if c.test < 1 else c.test)
        c.trainNames, c.testNames = c.allNames[:splitAt],c.allNames[splitAt:]
    elif isinstance(c.test,(str,unicode)):
        c.testNames,c.imgPathFormat = getNamesAndFormat(c.test)
    else:
        raise Exception,"cf.test is not define couldn't test!" 
    c.names = c.testNames

def setTrain():
    c.trainNames,c.imgPathFormat = getNamesAndFormat(c.trainGlob)
    if c.test and isinstance(c.test,(float,int)):
        n = len(c.trainNames)
        splitAt = n - (int(n*c.test) if c.test < 1 else c.test)
        c.trainNames, c.testNames = c.trainNames[:splitAt],c.trainNames[splitAt:]
    c.names = c.trainNames
def setPredict():
    c.names,c.imgPathFormat = getNamesAndFormat(c.trainGlob)
    
#c.update(cf)
#
#setMod('train')
#setMod('test')

indexOf = c.indexOf = lambda name:c.names.index(name)
# toimg, togt 将 name 转换为对应path
toimg = c.toimg = lambda name: c.imgPathFormat%(name)
togt = c.togt = lambda name:c.toGtPath(c.toimg(name))

readimg = c.readimg = lambda name:imread(c.toimg(name))
readgt = c.readgt = lambda name:imread(c.togt(name))

getImgGtNames = c.getImgGtNames = lambda names:[(c.toimg(n),c.togt(n)) for n in names]


if __name__ == '__main__':

    pass
