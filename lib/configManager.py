# -*- coding: utf-8 -*-
import os,sys

import yllibInterface as yl
from yllibInterface import dicto, glob, fileJoinPath,pathjoin
from yllibInterface import show, loga, logl, imread, imsave

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

    c.netpath = fileJoinPath(__file__,'../nets/'+c.netdir)
    if c.netpath not in sys.path:
        sys.path = [c.netpath]+sys.path
    
#    global name, img
#    name = c.names[0]
#    img = readimg(name)
    



cf = dicto()
c = dicto()
args = dicto()
c.args = args
c.tmpdir = 'TMP'
c.tmpdir = ''

def setTrain():
    c.trainNames,c.imgPathFormat = getNamesAndFormat(c.trainGlob)
    if c.test and isinstance(c.test,(float,int)):
        n = len(c.trainNames)
        splitAt = n - (int(n*c.test) if c.test < 1 else c.test)
        c.trainNames, c.testNames = c.trainNames[:splitAt],c.trainNames[splitAt:]
    c.names = c.trainNames

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

def setPredict():
    c.names,c.imgPathFormat = getNamesAndFormat(c.predictGlob)
    
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

def doc(mod):
    if '__doc__' in dir(mod):
        print mod.__doc__
    else:
        help(mod)

def makdirs(dirr):
    if not os.path.isdir(dirr):
        os.makedirs(dirr)
    
def makeTrainEnv(args):
    makdirs(os.path.dirname(args.prefix))
        
def makePredictEnv(args):
    makdirs(args.out)
makeTestEnv = makePredictEnv

if __name__ == '__main__':

    pass
