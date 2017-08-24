# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from tool.toolDataStructureAndObject import FunAddMagicMethod
from tool.toolFuncation import dynamicWraps
from ylnp import isNumpyType

import os
from ylimgTool import cv2, sk, sio, np, plt, da
from ylimgTool import (show, loga, mapp, normalizing, imsave, imread, 
                       r ,labelToColor)

def binaryDiff(re,gt,size=0.5,lines=50,bound=False):
    '''
    对二分类问题的结果上色
    False Positive: red mgrid
    False Negative: blue mgrid
    lines: 网格线条的间距
    size:线条多粗 即线条是间距的多少倍
    bound:是否画出边缘
    '''
    re, gt = re > 0.5, gt > 0.5
    rem = np.zeros(gt.shape,int)
    tp = (re) * (gt)
#    tn = (~re) * (~gt)
    fp = re*(~gt)
    fn = (~re)*gt
#    show(tp,tn,fp,fn)
    rem[tp] = 1
    rem[fp] = 2
    rem[fn] = 3
    c=[[0]*3,[1]*3,[1,0,0],[.1,.1,.6]]
    diff = classDiff(rem,re,c,size=size,lines=lines,bound=bound)
    return diff

def drawBoundAndBackground(img,mask,bg=None,replace=False,lines=50,
                           size=0.2,bound=True,boundmode='thick'):
    '''
    给出mask 将给mask区域填充背景色bg的线条 并加上黑白边框
    mask: 所要标注区域
    bg : 背景填充 可以为颜色|图片 默认为红色
    replace : 是否在原图上操作
    lines: 线条的间距
    size:线条多粗 即线条是间距的多少倍
    bound:是否画出边缘
    boundmode: thick 粗正好在边界 'inner'只在前景里面 但是较细
    '''
    assert mask.ndim ==2, u'mask 必须为布尔值'
    if not mask.any():
        return img
    isint = isNumpyType(img,int)
    if not replace:
        img = img.copy()
    if bg is None:
        bg =  [max(255,img.max()),128,0]if isint else [1.,.5,0]
    white = max(255,img.max()) if isint else 1.
    m,n=img.shape[:2]
    i,j = np.mgrid[:m,:n]
    
    step = (m+n)//2//lines
    a = int(step*(1-size))
    drawInd = ~np.where(((i%step<a)& (j%step<a)),True,False)
#    from tool import g
#    g.x = mask,drawInd, bg,img
    if isinstance(bg,np.ndarray) and bg.ndim >=2:
        img[mask*drawInd] = bg[mask*drawInd]
    else:
        img[mask*drawInd] = bg
    if bound:
        from skimage.segmentation import find_boundaries
        boundind = find_boundaries(mask, mode=boundmode,background=True)
        boundBg = np.where((i+j)%10<5,white,0)
        img[boundind] = boundBg[boundind][...,None]
    return (img)
   
def classDiff(rem,gtm,colors=None,size=.15,reMod=False,lines=50,bound=True):
    '''
    对多分类问题的gt进行上色
    对于错误分类 加网格(网格颜色是resoult的颜色) 加有边框
    rem :多分类结果 二维矩阵
    gtm :GroundTruth
    colors:标签对应的颜色
    size:网格所占用的比重
    reMod:对resoult上色, 对于错误分类 网格颜色是 GroundTruth的颜色
    lines:网格线的数量
    bound:是否画出边缘
    '''
    assert rem.ndim==2 and gtm.ndim==2,'rem,gtm 必须为2维多标签矩阵'
    rgb = labelToColor(rem if reMod else gtm, colors) 
    clas = range(len(colors))
    for c,color in enumerate((colors)): #c mean iter every Class
        for oc in clas: # oc means OtherClass
            if oc==c:
                continue
            mask = (rem==c)*(gtm==oc)
            if mask.any():
                bg = colors[oc if reMod else c]
#                print bg,c,oc
                drawBoundAndBackground(rgb,mask,bg,bound=bound,
                                       size=size,replace=True,lines=lines)
    return rgb



@dynamicWraps
def getWeightCore(hh,ww=None,mappFun=None,seta=0.5):
    '''
    返回一个权重二维矩阵 ，默认是seta=0.5的二维高斯分布
    mappFun: 对矩阵每个点执行mappFun(i,j) 返回的结果构成二维矩阵
    '''
    if ww is None:
        ww = hh
    if mappFun is None:
#        ijToCenter = lambda x,i,j:(((i/float(hh)-1/2.)**2+(j/float(ww)-1/2.)**2))
#        wc = weightCore = mapp(ijToCenter,weightCore,need_i_j=True)
        i,j = np.mgrid[:hh,:ww]
        wc = (((i/float(hh)-1/2.)**2+(j/float(ww)-1/2.)**2))
        wc = 1./(2*np.pi*seta**2)*np.e**(-wc/(2*seta**2))
        wc = wc/wc.max()
        #show(normalizing(img[:hh,:ww]*wc[...,None]),img[:hh,:ww])
#        polt3dSurface(wc)
        return wc
    weightCore = np.zeros((hh,ww))
    return mapp(lambda x,i,j:mappFun(i,j),weightCore,need_i_j=True)

def smallImg(img,hh=360,ww=480, step=None,f=None):
    '''
    将大图切割成固定大小的小图
        step:切割的步长, 默认为(hh,ww) 可以为int|tuple(steph,stepw)|float
        f: 若有f 则执行f(simg,i,j),其中：
            simg:被切割的小图片
            i: simg所在img的row
            j: simg所在img的col
    '''
    if step is None:
        steph,stepw = hh,ww
    if isinstance(step,int):
        steph,stepw = step,step
    if isinstance(step,float):
        steph,stepw = int(hh*step),int(ww*step)
    if isinstance(step,(tuple,list)):
        steph,stepw = step
    h,w = img.shape[:2]
    simgs = []
    for i in range(0,h-hh,steph)[:]+[h-hh]:
        for j in range(0,w-ww,stepw)[:]+[w-ww]:
            simg = img[i:i+hh,j:j+ww]
            simgs.append(simg)
            if f:
                f(simg,i,j)
    return simgs


def autoSegmentWholeImg(img,simgShape,handleSimg,step=None,weightCore=None):
    '''
    将img分割到 simgShape 的小图，执行handleSimg(simg),将结果拼接成回img形状的矩                                                                                                                                                     阵
    img:被执行的图片
    simgShape: 小图片的shape
    handleSimg: 用于处理小图片的函数 handleSimg(simg)，比如 net.pridict(simg)
    step: 切割的步长, 默认为simgShape 可以为int|tuple(steph,stepw)|float
    weightCore: 'avg'取平均,'gauss'结果的权重 在重叠部分可以用到 使之越靠经中心                                                                                                                                                     的权重越高 默认为直接覆盖
    '''
    if isinstance(simgShape,int):
        hh,ww = simgShape,simgShape
    hh,ww = simgShape
    h,w = img.shape[:2]
    if weightCore in ['avg']:
        weightCore = np.ones((hh,ww))
    elif isinstance(weightCore,np.ndarray):
        pass
    elif weightCore in ['guss','gauss']:
        weightCore = getWeightCore(hh,ww)
    else:
        raise Exception,'Illegal argus `weightCore` in `autoSegmentWholeImg`!'
    weight = np.zeros((h,w))
    class c:
        re=None
    def f(simg,i,j):
        sre = handleSimg(simg)
        if c.re is None:
            c.re = np.zeros((h,w)+sre.shape[2:],sre.dtype)
        if weightCore is None:
            c.re[i:i+hh,j:j+ww]= sre
            return
        resoult = c.re
        oldw = weight[i:i+hh,j:j+ww]
        ws = weightCore
        if sre.ndim!=2:
            ws = ws[...,None]
            oldw = oldw[...,None]
    #    map(loga,[ws,sre,resoult,oldw,resoult[i:i+hh,j:j+ww]*oldw])
        resoult[i:i+hh,j:j+ww] = (ws*sre + resoult[i:i+hh,j:j+ww]*oldw)/(ws+oldw                                                                                                                                                     )
        weight[i:i+hh,j:j+ww] += weightCore
    #    show(resoult,weight)
    simgs = smallImg(img,hh,ww,step=step,f=f)
    (simgs)
    resoult = c.re
#    show(weight,resoult)
    return resoult



from collections import Iterator 
class GenSimg(Iterator):
    '''
    随机生成小图片simg及gt 的迭代器，默认使用1Gb内存作为图片缓存
    默认生成simg总面积≈所有图像总面积时 即结束
    '''
    def __init__(self, imggts, simgShape, handleImgGt=None,
                 batch=1, cache=None,iters=None,
                 timesPerRead=1,infinity=False):
        '''
        imggts: zip(jpgs,pngs)
        simgShape: simg的shape
        handleImgGt: 对输出结果运行handleImgGt(img,gt)处理后再返回
        batch: 每次返回的batch个数
        cache: 缓存图片数目, 默认缓存1Gb的数目
        timesPerRead: 平均每次读的图片使用多少次(不会影响总迭代次数),默认1次
        iters: 固定输出小图片的总数目，与batch无关
        infinity: 无限迭代
        '''
        if isinstance(simgShape,int):
            simgShape = (simgShape,simgShape)
        self.handleImgGt = handleImgGt
        self.imggts = imggts
        self.simgShape = simgShape
        self.batch = batch
        self._iters = iters
        self.iters = self._iters
        self.infinity = infinity
        
        hh,ww = simgShape
        jpg,png = imggts[0]
        img = imread(jpg)
        h,w = img.shape[:2]
        if cache is None:
            cache = int(1e9/img.nbytes)
        cache = min(cache,len(imggts))
        self.maxPerCache = int(cache*(h*w)*1./(hh*ww))* timesPerRead/batch
        self.cache = cache
        self.n = len(imggts)
        self._times = max(1,int(round(self.n*1./cache/timesPerRead)))
        self.times = self._times
        self.totaln = self.sn = iters or int((h*w)*self.n*1./(hh*ww))
        self.willn = iters or self.maxPerCache*self.times*batch
        self.count = 0
        self.reset()
        
        self.bytes = img.nbytes
        argsStr = '''imggts=%s pics in dir: %s, 
        simgShape=%s, 
        handleImgGt=%s,
        batch=%s, cache=%s,iters=%s,
        timesPerRead=%s, infinity=%s'''%(self.n , os.path.dirname(jpg) or './', simgShape, handleImgGt,
                                 batch, cache,iters,
                                 timesPerRead,infinity)
        generatorStr = '''maxPerCache=%s, readTimes=%s
        Will generator maxPerCache*readTimes*batch=%s'''%(self.maxPerCache, self.times,
                                                          self.willn)
        if iters:
            generatorStr = 'Will generator iters=%s'%iters
        self.__describe = '''GenSimg(%s)
        
        Total imgs Could generator %s simgs,
        %s simgs.
        '''%(argsStr,self.totaln,
             generatorStr,)
    def reset(self):
        if (self.times<=0 and self.iters is None) and not self.infinity:
            self.times = self._times
            raise StopIteration
        self.now = self.maxPerCache
        inds = np.random.choice(range(len(self.imggts)),self.cache,replace=False)
        datas = {}
        for ind in inds:
            jpg,png = self.imggts[ind]
            img,gt = imread(jpg),imread(png)
            datas[jpg] = img,gt
        self.data = self.datas = datas
        self.times -= 1
    def next(self):
        self.count += 1
        if (self.iters is not None) and not self.infinity:
            if self.iters <= 0:
                self.iters = self._iters
                raise StopIteration
            self.iters -= self.batch
        if self.now <= 0:
            self.reset()
        self.now -= 1
        hh,ww = self.simgShape
        datas = self.datas
        imgs, gts = [], []
        for t in range(self.batch):
            img,gt = datas[np.random.choice(datas.keys(),1,replace=False)[0]]
            h,w = img.shape[:2]
            i= np.random.randint(h-hh+1)
            j= np.random.randint(w-ww+1)
            (img,gt) =  img[i:i+hh,j:j+ww],gt[i:i+hh,j:j+ww]
            imgs.append(img), gts.append(gt)
        (imgs,gts) = map(np.array,(imgs,gts))
        if self.handleImgGt:
            return self.handleImgGt(imgs,gts)
        return (imgs,gts)
    @property
    def imgs(self):
        return [img for img,gt in self.datas.values()]
    @property
    def gts(self):
        return [gt for img,gt in self.datas.values()]
    def __str__(self):
        batch = self.batch
        n = len(self.datas)
        return self.__describe + \
        '''
    status:
        iter  in %s/%s(%.2f)
        batch in %s/%s(%.2f)
        cache imgs: %s
        cache size: %.2f MB
        '''%(self.count*batch,self.willn,self.count*1.*batch/self.willn,
            self.count,self._times*self.maxPerCache,
            self.count*1./(self._times*self.maxPerCache),
            n, (n*self.bytes/2**20))
        
    __repr__ = __str__

class Evalu(dict):
    '''
    '''
    __loged__ = False
    evalus = {}
    def __init__(self,name,evaluFun,logFormat=None):
        '''
        evaluFun: evaluFun(re,gt) should return a dict or tuple (ps:re mean resoult )
        '''
        if not Evalu.__loged__:
            from tool import log
            log()
        self.name = name
        self.f = evaluFun
        self.logFormat = logFormat
#        isTuple = '%s' in logFormat or '%f' in logFormat or '%d' in logFormat 
#        isDict = '{' in logFormat and '}' in logFormat 
#        if (isTuple and isDict) or not(isTuple or isDict):
#            raise Exception,'logFormat must be %s or {format}'
#        self.__isTuple = isTuple
        self.__isTuple = None
        self.keylist = []
        self.df = None
        
        self.nameFormath = './tmp/EvaluSave/Evalu_%s'
#        Evalu.evalus[id(self)] = self
    def evalu(self, re, gt, key=None):
        
        return 
    def __str__(self,):
        
        return self.name
        pass
    def summary(self,):
        self.__str__()
        pass
    def save(self, saveGts=True):
        pass
    def __getitem__(self,key):
        if key in self:
            return dict.__getitem__(self,key)
        if isinstance(key,int):
            key = self.keylist[key]
        return  dict.__getitem__(self,key)
    def __logstr(self,data):
        tmp = self.logFormat
        self.__isTuple = isinstance(data,tuple)
        if self.__isTuple:
            return tmp%tuple(data)
        else:
            return tmp.format(**data)
        pass
    def __call__(self, reOrKey, gt=None, key=None):
        if reOrKey in self:
            return (self.df[reOrKey])
        pass
    def __del__(self,):
        '''
        失效了 可能会导致内存泄漏
        '''
        if Evalu and id(self) in Evalu.evalus:
            del Evalu.evalus[id(self)]
        del self
    dell = __del__
    def f(self,):
        pass
    __repr__ = __str__
if __name__ == '__main__':

    import pandas as pd
    df = pd.DataFrame({
                       0:range(5),
                       1:range(10,15),
                       'a':list("abcde"),
                       })
    df.set_index(0,inplace=True)
#    e = Evalu('sd','','%s')
#    e = Evalu('zd','','%s')
#    ed = Evalu.evalus

    pass
