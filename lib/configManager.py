# -*- coding: utf-8 -*-
import os,sys

import yllibInterface as yl
from yllibInterface import dicto, glob, fileJoinPath,pathjoin,addPathToSys
from yllibInterface import show, loga, tree, imread, imsave

def getNamesAndFormat(globpath):
    paths = glob(globpath)
    assert len(paths),'glob.golb("%s") is Empty!'%globpath
    rind = 1-len(globpath)+(globpath.rindex('*'))
    imgPathFormat = globpath[:rind-1]+'%s'+globpath[rind:]
    names = sorted([os.path.basename(p)[:rind] for p in paths])
    return names,imgPathFormat


 
def setMod(mod='train'):
    c.mod = mod
    if mod == 'train':
        setTrain()
    if mod == 'val':
        setVal()
    if mod == 'test':
        setTest()

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
    if c.val and isinstance(c.val,(float,int)):
        n = len(c.trainNames)
        splitAt = n - (int(n*c.val) if c.val < 1 else c.val)
        c.trainNames, c.valNames = c.trainNames[:splitAt],c.trainNames[splitAt:]
    c.names = c.trainNames

def setVal():
    if c.val and isinstance(c.val,(float,int)):
        c.allNames,c.imgPathFormat = getNamesAndFormat(c.trainGlob)
        n = len(c.allNames)
        splitAt = n - (int(n*c.val) if c.val < 1 else c.val)
        c.trainNames, c.valNames = c.allNames[:splitAt],c.allNames[splitAt:]
    elif isinstance(c.val,(str,unicode)):
        c.valNames,c.imgPathFormat = getNamesAndFormat(c.val)
    else:
        raise Exception,"cf.val is not define couldn't val!" 
    c.names = c.valNames
    c.toGtPath = c.toValGtPath

def setTest():
    c.names,c.imgPathFormat = getNamesAndFormat(c.testGlob)
    
#c.update(cf)
#
#setMod('train')
#setMod('val')

indexOf = c.indexOf = lambda name:c.names.index(name)
# toimg, togt 将 name 转换为对应path
toimg = c.toimg = lambda name: c.imgPathFormat%(name)
togt = c.togt = lambda name:c.toGtPath(c.toimg(name))

readimg = c.readimg = lambda name:imread(c.toimg(name))
def readgt(name):
    gt = imread(c.togt(name))
    if 'args' in c and 'classn' in c.args and c.args.classn == 2:
        return gt > .5
    return gt
c.readgt = readgt
getImgGtNames = c.getImgGtNames = lambda names:[(c.toimg(n),c.togt(n)) for n in names]

def doc(mod):
    if '__doc__' in dir(mod):
        print mod.__doc__
    else:
        help(mod)

def makdirs(dirr):
    if not os.path.isdir(dirr):
        os.makedirs(dirr)
def addUpDirectoryToSys(_file_):
    '''
    add up directory to your sys.path 
    '''
    return addPathToSys(_file_, '..')

def makeTrainEnv(args):
    makdirs(os.path.dirname(args.prefix))
        
def makeTestEnv(args):
    makdirs(args.out)
makeValEnv = makeTestEnv

if __name__ == '__main__':
    pass
