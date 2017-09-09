# encoding: utf-8
'''
res-unet
全图预测 自动填充黑边 以适应上下采样

Parameters
----------
'''
from lib import *
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx

from collections import namedtuple
from configManager import args

if __name__ == '__main__':
    import test
    
sym, arg_params, aux_params = mx.model.load_checkpoint(
    args.prefix, args.restore)
# print(sym.list_outputs())
Batch = namedtuple('Batch', ['data'])
mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.gpu())
    
def predict(name):
    img = imread(name)/255.
#    img = img[::3,::3]
    step = args.step
    h,w = img.shape[:2]
    hh = ((h-1)//step+1)*step
    ww = ((w-1)//step+1)*step
    
    nimg = np.zeros((hh,ww,3))
    nimg[:h,:w] = img
    if not mod.binded:
        mod.bind(for_training=False, data_shapes=[
                 ('data', (1, 3, hh, ww))], label_shapes=None and [
                         ('softmax%d_label'%(i+1),(1,hh//2**i,ww//2**i)) for i in (0,)])
        mod.set_params(arg_params, aux_params, allow_missing=True)
    simg = nimg.transpose(2,0,1)
    mod.forward(Batch(data=[mx.nd.array(np.expand_dims(
            simg, 0))]), is_train=False)
    prob = mod.get_outputs()[0].asnumpy()[0]
    
    re= prob.transpose(1,2,0)
    re = re[:h,:w] 
    return re       
    
if __name__ == '__main__':
    pass
    from test import *