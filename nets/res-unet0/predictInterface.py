from lib import *
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import mxnet as mx

from collections import namedtuple


def getPredict(args):
    args.simgShape = args.window
    if not isinstance(args.window,(tuple,list,np.ndarray)):
        args.simgShape = (args.window,args.window)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        args.prefix, args.restore)
    # print(sym.list_outputs())
    mod = mx.mod.Module(symbol=sym, label_names=None, context=mx.gpu())
    mod.bind(for_training=False, data_shapes=[
             ('data', (1, 3, args.simgShape[0], args.simgShape[1]))], label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    Batch = namedtuple('Batch', ['data'])
    
    dic = {}
    
    def handleSimg(simg):
        simg = simg.transpose(2,0,1)
        mod.forward(Batch(data=[mx.nd.array(np.expand_dims(
                simg, 0))]), is_train=False)
        prob = mod.get_outputs()[0].asnumpy()[0]
        return prob.transpose(1,2,0)
    
    def predict(name,step='null',weightCore='gauss'):
        img = imread(name)
        step=args.step if step == 'null' else args.step
        re = autoSegmentWholeImg(img, args.simgShape, handleSimg,step=step,weightCore=weightCore)
        return re
    return predict
    
        
    
if __name__ == '__main__':
    pass
