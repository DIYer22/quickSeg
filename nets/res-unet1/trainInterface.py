# coding: utf-8
'''
res-unet
全图训练 自动填充黑边 以适应上下采样

Parameters
----------
step : int
    填充黑边 将图片shape 调整为step的整数倍
'''
from lib import *

import logging
logging.basicConfig(level=logging.INFO)
npm = lambda m:m.asnumpy()
npm = FunAddMagicMethod(npm)

import mxnet as mx

from netdef import getNet

class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
from collections import Iterator 
class genImg(Iterator):
    def __init__(self,names,batch=1,
                 handleImgGt=None,
                 timesPerRead=1,
                 ):
        self.names = names
        self.batch = batch
        self.tpr = timesPerRead
        self.handleImgGt = handleImgGt
        self.genNameBatchs()
    def genNameBatchs(self):
        import random
        self.now = 0
        random.shuffle(self.names)
        batch = self.batch
        nameBatchs = listToBatch(self.names,batch)
        more = (batch - len(nameBatchs[-1]))
        nameBatchs[-1] += tuple(random.sample(self.names,more))
        self.nameBatchs = nameBatchs
        self.lenn = len(nameBatchs)
    reset = genNameBatchs
    def next(self):
        now,lenn,names = self.now,self.lenn,self.nameBatchs
        if lenn == now:
            self.genNameBatchs()
            raise StopIteration
        self.now += 1
        imgs = [];gts = []
        for img,gt in names[now]:
            imgs.append(imread(img))
            gts.append(imread(gt))
        if self.handleImgGt:
            return self.handleImgGt(imgs,gts)
        return (imgs,gts)
    
labrgb = lambda lab:cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)
randint = lambda x:np.random.randint(-x,x)
def imgToLab(img,gt):
    labr=cv2.cvtColor(img,cv2.COLOR_RGB2LAB)#/np.float32(255)
    return labr
def imgAug(img,gt,prob=.5):
    lab = img
    if np.random.random()<prob:
        lab = imgToLab(img,gt)
    if np.random.random()<prob:
        lab=np.fliplr(lab)
        gt=np.fliplr(gt)
#    show(labrgb(lab),img)
    return lab,gt

def imgGtAdd0Fill(step=1):
    def innerf(imgs,gts):
        img = imgs[0][::c.resize,::c.resize]
        h,w = img.shape[:2]
        hh = ((h-1)//step+1)*step
        ww = ((w-1)//step+1)*step
        nimgs,ngts=[],[]
        for img,gt in zip(imgs,gts):
            gt=gt>.5
            img,gt = img[::c.resize,::c.resize],gt[::c.resize,::c.resize]
            img,gt = imgAug(img,gt)
            img = img/255.
            nimg = np.zeros((hh,ww,3))
            ngt = np.zeros((hh,ww),np.bool)
            h,w = img.shape[:2]
            nimg[:h,:w] = img
            ngt[:h,:w]=gt
            nimgs.append(nimg)
            ngts.append(ngt)
        imgs,gts=np.array(nimgs),np.array(ngts)
#        return imgs,gts
        imgs = imgs.transpose(0,3,1,2)
        mximgs = map(mx.nd.array,[imgs])
        mxgtss = map(mx.nd.array,[gts])
        mxdata = SimpleBatch(mximgs,mxgtss)
        return mxdata
    return innerf



class GenSimgInMxnet(genImg):
    @property
    def provide_data(self):
        return [('data', (c.batch, 3, c.simgShape[0], c.simgShape[1]))]
    @property
    def provide_label(self):
        return  [('softmax1_label', (c.batch, c.simgShape[0], c.simgShape[1])),]


def saveNow(name = None):
    f=mx.callback.do_checkpoint(name or args.prefix)
    f(-1,mod.symbol,*mod.get_params())
    
c = dicto(
 gpu = 1,
 lr = 0.01,
 epochSize = 10000,
 step=64
 )
c.resize = 1

if __name__ == '__main__':
    from train import args
    
else:
    from configManager import args
c.update(args)
args = c

img = imread(c.names[0][0])
img = img[::c.resize,::c.resize]
h,w = img.shape[:2]
hh = ((h-1)//c.step+1)*c.step
ww = ((w-1)//c.step+1)*c.step
args.simgShape = (hh,ww)

net = getNet(args.classn)

if args.resume:
    print('resume training from epoch {}'.format(args.resume))
    _, arg_params, aux_params = mx.model.load_checkpoint(
        args.prefix, args.resume)
else:
    arg_params = None
    aux_params = None

if 'plot' in args:
    mx.viz.plot_network(net, save_format='pdf', shape={
        'data': (1, 3, 640, 640),
        'softmax1_label': (1, 640, 640), }).render(args.prefix)
    exit(0)
mod = mx.mod.Module(
    symbol=net,
    context=[mx.gpu(k) for k in range(args.gpu)],
    data_names=('data',),
    label_names=('softmax1_label',)
)
c.mod = mod

#if 0:
args.names = args.names[:]
#    data = GenSimgInMxnet(args.names, args.simgShape, 
#                          handleImgGt=handleImgGt,
#                          batch=args.batch,
#                          cache=None,
#                          iters=args.epochSize
#                          )
gen = GenSimgInMxnet(args.names,c.batch,handleImgGt=imgGtAdd0Fill(c.step))
g.gen = gen
total_steps = len(c.names) * args.epoch
lr_sch = mx.lr_scheduler.MultiFactorScheduler(
    step=[total_steps // 2, total_steps // 4 * 3], factor=0.1)
def train():
    mod.fit(
        gen,
        begin_epoch=args.resume,
        arg_params=arg_params,
        aux_params=aux_params,
        batch_end_callback=mx.callback.Speedometer(args.batch),
        epoch_end_callback=mx.callback.do_checkpoint(args.prefix),
        optimizer='sgd',
        optimizer_params=(('learning_rate', args.lr), ('momentum', 0.9),
                          ('lr_scheduler', lr_sch), ('wd', 0.0005)),
        num_epoch=args.epoch)
if __name__ == '__main__':
    pass


if 0:
    #%%
    ne = g.gen.next()
#for ne in dd:
    ds,las = ne.data, ne.label
    d,la = npm-ds[0],npm-las[0]
    im = d.transpose(0,2,3,1)
    show(labrgb(uint8(im[0])));show(la)
