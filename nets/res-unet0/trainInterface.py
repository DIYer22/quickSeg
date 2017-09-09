# coding: utf-8

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
        
class GenSimgInMxnet(GenSimg):
    @property
    def provide_data(self):
        return [('data', (c.batch, 3, c.simgShape[0], c.simgShape[1]))]
    @property
    def provide_label(self):
        return  [('softmax1_label', (c.batch, c.simgShape[0], c.simgShape[1])),
                 ('softmax2_label', (c.batch, c.simgShape[0]//2, c.simgShape[1]//2)),
                 ('softmax3_label', (c.batch, c.simgShape[0]//4, c.simgShape[1]//4)),
                 ('softmax4_label', (c.batch, c.simgShape[0]//8, c.simgShape[1]//8))]
def handleImgGt(imgs, gts,):
    gts = gts>0.5
    for i in range(len(imgs)):
        if np.random.randint(2):
            imgs[i] = np.fliplr(imgs[i])
            gts[i] = np.fliplr(gts[i])
        if np.random.randint(2):
            imgs[i] = np.flipud(imgs[i])
            gts[i] = np.flipud(gts[i])
    imgs = imgs.transpose(0,3,1,2)
    gtss = [gts[:,::sc,::sc] for sc in [1,2,4,8]]
    mximgs = map(mx.nd.array,[imgs])
    mxgtss = map(mx.nd.array,gtss)
    mxdata = SimpleBatch(mximgs,mxgtss)
    return mxdata
def saveNow(name = None):
    f=mx.callback.do_checkpoint(name or args.prefix)
    f(-1,mod.symbol,*mod.get_params())
    
c = dicto(
 gpu = 1,
 lr = 0.01,
 epochSize = 10000,
 )
def train(args):
    c.update(args)
    args = c
    args.simgShape = args.window
    if not isinstance(args.window,(tuple,list,np.ndarray)):
        args.simgShape = (args.window,args.window)
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
            'softmax1_label': (1, 640, 640),
            'softmax2_label': (1, 320, 320),
            'softmax3_label': (1, 160, 160),
            'softmax4_label': (1, 80, 80), }).render(args.prefix)
        exit(0)
    mod = mx.mod.Module(
        symbol=net,
        context=[mx.gpu(k) for k in range(args.gpu)],
        data_names=('data',),
        label_names=('softmax1_label', 'softmax2_label',
                     'softmax3_label', 'softmax4_label',)
    )
    c.mod = mod

#if 0:
    args.names = args.names[:]
    data = GenSimgInMxnet(args.names, args.simgShape, 
                          handleImgGt=handleImgGt,
                          batch=args.batch,
                          cache=None,
                          iters=args.epochSize
                          )
    
    total_steps = args.epochSize * args.epoch
    lr_sch = mx.lr_scheduler.MultiFactorScheduler(
        step=[total_steps // 2, total_steps // 4 * 3], factor=0.1)

    mod.fit(
        data,
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
    ne = data.next()
#for ne in dd:
    ds,las = ne.data, ne.label
    d,la = npm-ds[0],npm-las[0]
    im = d.transpose(0,2,3,1)
    show(im);show(la)
