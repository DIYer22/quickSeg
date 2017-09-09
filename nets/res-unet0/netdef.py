# coding: utf-8
import mxnet as mx
def conv(data, kernel=(3, 3), stride=(1, 1), pad=(0, 0), num_filter=None, name=None):
    return mx.sym.Convolution(data=data, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter, name='conv_{}'.format(name))


def bn_relu(data, name):
    return mx.sym.Activation(data=mx.sym.BatchNorm(data=data, momentum=0.99, name='bn_{}'.format(name)), act_type='relu', name='relu_{}'.format(name))


def conv_bn_relu(data, kernel=(3, 3), stride=(1, 1), pad=(0, 0), num_filter=None, name=None):
    return bn_relu(conv(data, kernel, stride, pad, num_filter, 'conv_{}'.format(name)), 'relu_{}'.format(name))


def down_block(data, f, name):
    x = mx.sym.Pooling(data=data, kernel=(2,2), stride=(2,2), pool_type='max')
    # temp = conv_bn_relu(data, (3, 3), (2, 2), (1, 1),
    #                     f, 'layer1_{}'.format(name))
    temp = conv_bn_relu(x, (3, 3), (1, 1), (1, 1),
                        2*f, 'layer2_{}'.format(name))
    bn = mx.sym.BatchNorm(data=conv(temp, (3, 3), (1, 1), (1, 1), f, 'layer3_{}'.format(
        name)), momentum=0.99, name='layer3_bn_{}'.format(name))
    bn += x
    act = mx.sym.Activation(data=bn, act_type='relu',
                            name='layer3_relu_{}'.format(name))
    return bn, act


def up_block(act, bn, f, name):
    x = mx.sym.UpSampling(
        act, scale=2, sample_type='nearest', name='upsample_{}'.format(name))
    # temp = mx.sym.Deconvolution(data=act, kernel=(3, 3), stride=(2, 2), pad=(
    #    1, 1), adj=(1, 1), num_filter=32, name='layer1_dconv_{}'.format(name))
    temp = mx.sym.concat(bn, x, dim=1)
    temp = conv_bn_relu(temp, (3, 3), (1, 1), (1, 1),
                        2*f, 'layer2_{}'.format(name))
    bn = mx.sym.BatchNorm(data=conv(temp, (3, 3), (1, 1), (1, 1), f, 'layer3_{}'.format(
        name)), momentum=0.99, name='layer3_bn_{}'.format(name))
    bn += x
    act = mx.sym.Activation(data=bn, act_type='relu',
                            name='layer3_relu_{}'.format(name))
    return act


def getNet(classn=2):
    data = mx.sym.Variable('data')
    x = conv_bn_relu(data, (3, 3), (1, 1), (1, 1), 32, 'conv0_1')
    net = conv_bn_relu(x, (3, 3), (1, 1), (1, 1), 64, 'conv0_2')
    bn1 = mx.sym.BatchNorm(data=conv(
        net, (3, 3), (1, 1), (1, 1), 32, 'conv0_3'), momentum=0.99, name='conv0_3_bn')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name='conv0_3_relu')

    bn2, act2 = down_block(act1, 32, 'down1')
    bn3, act3 = down_block(act2, 32, 'down2')
    bn4, act4 = down_block(act3, 32, 'down3')
    bn5, act5 = down_block(act4, 32, 'down4')
    bn6, act6 = down_block(act5, 32, 'down5')
    bn7, act7 = down_block(act6, 32, 'down6')

    temp = up_block(act7, bn6, 32, 'up6')
    temp = up_block(temp, bn5, 32, 'up5')
    temp = up_block(temp, bn4, 32, 'up4')
    score4 = conv(temp, (1, 1), (1, 1), (0, 0), classn, 'score4')
    net4 = mx.sym.SoftmaxOutput(score4, multi_output=True, name='softmax4')

    temp = up_block(temp, bn3, 32, 'up3')
    score3 = conv(temp, (1, 1), (1, 1), (0, 0), classn, 'score3')
    net3 = mx.sym.SoftmaxOutput(score3, multi_output=True, name='softmax3')

    temp = up_block(temp, bn2, 32, 'up2')
    score2 = conv(temp, (1, 1), (1, 1), (0, 0), classn, 'score2')
    net2 = mx.sym.SoftmaxOutput(score2, multi_output=True, name='softmax2')

    temp = up_block(temp, bn1, 32, 'up1')
    score1 = conv(temp, (1, 1), (1, 1), (0, 0), classn, 'score1')
    net1 = mx.sym.SoftmaxOutput(score1, multi_output=True, name='softmax1')

    return mx.sym.Group([net1, net2, net3, net4])



if __name__ == '__main__':
    
    pass






