import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def transformFeature(input_flow):
    b,inChannels,h,w = input_flow.size()
    kernelSize=3
    
    transformModule = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=inChannels,out_channels=2,kernel_size=kernelSize,stride=1,padding=(kernelSize - 1) // 2,bias=False),
        torch.nn.BatchNorm2d(2)
        )
            
    transformLayer = transformModule.cuda()
    
    for name,m in transformLayer.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias)
            torch.nn.init.xavier_uniform_(m.weight)
            
    output_flow = transformLayer(input_flow)
    softMax = torch.nn.LogSoftmax(dim=1)
    out_logSoftMax = softMax(output_flow)
    return out_logSoftMax
    
def checkRandomWeightIsLossOk(input_flow):
    b,inChannels,h,w = input_flow.size()
    correctIndex=1
    out_logSoftMax = input_flow.clone()
#    out_logSoftMax = transformFeature(out_logSoftMax)
#    pre = torch.max(out_logSoftMax,dim=1,keepdim=True)[1]
#    correctPointsNum = torch.sum(pre==correctIndex)
#    errPointsNum = torch.sum(pre==0)
#    pro = correctPointsNum.double()/(b*h*w)
#    con = errPointsNum.double()/(b*h*w)
#    torch.mean(out_logSoftMax[0])/torch.mean(out_logSoftMax[1])
    return torch.mean(out_logSoftMax[0])/torch.mean(out_logSoftMax[1])

def PR(output, target, ci):

    outputdata = output.data
    targetdata = target
    pre = torch.max(outputdata,dim=1,keepdim=True)[1]

    divionp = torch.sum(pre==ci)
    diviont = torch.sum(target==ci)

    if divionp<10:
        divionp = torch.tensor(1e10,dtype=torch.float64).cuda()

    if diviont<10:
        diviont = torch.tensor(1e10,dtype=torch.float64).cuda()

    P = torch.sum(targetdata[pre==ci]==ci)/divionp.double()
    R = torch.sum(targetdata[pre==ci]==ci)/diviont.double()

    return P, R


def getMultiscale_weights(model, fine_tune, epoch):

    if model == 'KittiSeg7':
        MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
    elif model == 'KittiSeg5':
        MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
    else:
        if fine_tune:# 250 epoch
            if epoch < 100:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
            elif epoch >= 100 and epoch < 150:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
            elif epoch >= 150 and epoch < 200:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
            else:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
        else:#250 epoch
            if epoch < 100:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
            elif epoch >= 100 and epoch < 150:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
            elif epoch >=150 and epoch < 200:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]
            else:
                MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0]

    return MY_MULTISCALE_WEIGHTS

def myloss(output, target):
    o = output.cpu().data.numpy()
    t = target.numpy()

    pre = np.zeros(target.shape)

    pre[np.argmax(o, axis=2) == 1] = 1
    pre[np.argmax(o, axis=2) == 0] = 0

    err = t-pre


    return


class weightcrossentropylossFun(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    def __init__(self, kernelSize=15, gpu=1):
        super(weightcrossentropylossFun, self).__init__()
        self.kernel_size = kernelSize
        self.gpu = gpu


    def forward(self, input, target):
        kerS= self.kernel_size # odd number
        b,c,h,w=input.size()


#        filters = Variable(torch.ones(c,1, kerS, kerS).float()).cuda(self.gpu)
        filters = Variable(torch.ones(c,1, kerS, kerS).float()).cuda()
        mask = Variable(target ==0).float()
        for i in range(c):
            if i!=0:
                slice = Variable(target == i).float()
                mask = torch.cat((mask,slice),dim=1)# one-hot
        weight = nn.functional.conv2d(mask, filters, padding=kerS//2, groups=c)
        weight = torch.mul(weight, mask)
        weight = (kerS*kerS)/torch.sum(weight, dim=1, keepdim=True)

        # output = Variable(input[mask.data==1].clone())
        output = torch.mul(Variable(input), mask)
        output = torch.sum(output,dim=1, keepdim=True)
        output = torch.mul(output, -weight)
        output = torch.mean(output)

        self.save_for_backward(input, target)

        return output.data

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        input, target = self.saved_tensors
        kerS=self.kernel_size # odd number
        b,c,h,w=input.size()


        filters = Variable(torch.ones(c,1, kerS, kerS).float()).cuda(self.gpu)
        mask = Variable(target ==0).float()
        for i in range(c):
            if i!=0:
                slice = Variable(target == i).float()
                mask = torch.cat((mask,slice),dim=1)# one-hot
        weight = nn.functional.conv2d(mask, filters, padding=kerS//2, groups=c)
        weight = torch.mul(weight, mask)
        weight = (kerS*kerS)/torch.sum(weight, dim=1, keepdim=True)

        grad_input = torch.mul((-weight).data, grad_output)
        grad_input = grad_input.expand(b,c,h,w)
        grad_input = torch.mul(grad_input, mask.data)

        return grad_input,None




class Myweightcrossentropyloss(torch.nn.Module):
    def __init__(self,kernelSize=15, gpu=1):
        super(Myweightcrossentropyloss, self).__init__()
        self.kernel_size = kernelSize
        self.gpu = gpu
    def forward(self, input, target):
        result = weightcrossentropylossFun(self.kernel_size, self.gpu)(input, target)
        return result


def weightcrossentropyloss(weight=None, size_average=True, ignore_index=-100, reduce=True):
    return torch.nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce)

def weightmultiscaleCrossentropyloss(mulweights=None, weight=None, sparse=False):
    return crossentropyloss(weight=weight, size_average=True, ignore_index=-100, reduce=True)


def crossentropyloss(weight=None, size_average=True, ignore_index=-100, reduce=True):
    return torch.nn.CrossEntropyLoss(weight=weight, size_average=size_average, ignore_index=ignore_index, reduce=reduce)


def multiscaleCrossentropyloss(mulweights=None, weight=None, sparse=False):
    return crossentropyloss(weight=weight, size_average=True, ignore_index=-100, reduce=True)


def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    # print target_flow.shape
    # print target_flow.shape[0]*target_flow.shape[1]*target_flow.shape[2]
    # print input_flow.shape
    # print input_flow.shape[0]*input_flow.shape[1]*input_flow.shape[2]
    # print EPE_map.shape
    # print EPE_map.shape[0]*EPE_map.shape[1]*EPE_map.shape[2]

    if sparse:
        EPE_map = EPE_map[target_flow != 0]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()


def multiscaleEPE(network_output, target_flow, weights=None, sparse=False):
    def one_scale(output, target, sparse):

        b, _, h, w = output.size()

        if sparse:
            target_scaled = nn.functional.adaptive_max_pool2d(target, (h, w))
        else:
            target_scaled = nn.functional.adaptive_avg_pool2d(target, (h, w))
        return EPE(output, target_scaled, sparse, mean=False)

    if type(network_output) not in [tuple, list]:
        network_output = [network_output]
    if weights is None:
        weights = [0.005,0.01,0.02,0.08,0.32]  # as in original article
    assert(len(weights) == len(network_output))

    loss = 0
    for output, weight in zip(network_output, weights):
        loss += weight * one_scale(output, target_flow, sparse)
    return loss


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = nn.functional.upsample(output, size=(h,w), mode='bilinear')
    return EPE(upsampled_output, target, sparse, mean=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)
