import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as Fun
from torchvision import transforms
from PIL import Image
import glob


def downsampled_flow(output, target, sparse=False):
    b, _, h, w = output.size()

    if sparse:
        target_scaled = Fun.adaptive_max_pool2d(target, (h, w))
    else:
        target_scaled = Fun.adaptive_avg_pool2d(target, (h, w))
    return target_scaled


def myflip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i)-1, -1, -1).long()
                   for i in range(x.dim()))]

def myRotateTensor(x,kersize):
    xexpand = Fun.pad(x, (1, 1, 1, 1), "constant", 0)
    xnew = torch.FloatTensor(xexpand.shape[0],xexpand.shape[1],xexpand.shape[2],xexpand.shape[3])
    for j in range(0, xexpand.shape[2], kersize):
        xnew[:,:,j:j+kersize,:]=myflip(xexpand[:,:,j:j+kersize,:])

    for i in range(0, xexpand.shape[3], kersize):
        xnew[:, :, :, i:i+kersize] = myflip(xnew[:, :, :, i:i+kersize])

    return xnew




class CorrelationFunction(torch.autograd.Function):

    def __init__(self, kernelSize=3, stride=1, padding=1, maxDisp=192/8, corrMultiply=1, gpu=0):
        super(CorrelationFunction, self).__init__()
        self.kernel_size = kernelSize
        self.stride = stride
        self.padding = padding
        self.max_disp = maxDisp
        self.corr_multiply = corrMultiply
        self.gpu = gpu
        # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    def forward(self, inputs):

        self.save_for_backward(inputs)
        featureNum = inputs.shape[1]
        input1 = inputs[:,0:featureNum//2]
        input2 = inputs[:,featureNum//2:featureNum]

        h,c,h,w = input1.shape # b x c x h x w

        first = True
        input2_dis = Fun.pad(input2, (self.max_disp, 0, 0, 0), "constant", 0).data  # b x c x h x (d+w)
        filters = Variable(torch.FloatTensor(self.kernel_size, self.kernel_size).fill_(1).view(1, 1, self.kernel_size,
                                                                                               self.kernel_size)).cuda(self.gpu)  # double -> float out x in x h x w

        for d in reversed(range(self.max_disp+1)):

            input2_d = input2_dis[:,:,:,d:d+w] # b x c x h x w
            outputSlice = torch.mul(input1,input2_d) # element mul b x c x h x w

            # sum slice and convolution
            outputSlice = Variable(torch.sum(outputSlice,dim=1,keepdim=True))# b x 1 x h x w
            outputSlice = Fun.conv2d(outputSlice, filters, padding=self.padding)

            if not first:
                output = torch.cat((output, outputSlice), 1)# b x c(d) x h x w
            else:
                output = outputSlice.clone()
                first = False

        return output.data


    def backward(self, grad_output):

        inputs, = self.saved_tensors

        featureNum = inputs.shape[1]
        input1 = inputs[:,0:featureNum//2]# b x c x h x w
        input2 = inputs[:,featureNum//2:featureNum]

        b,c,h,w = input2.shape  # b x c x h x w
        #filters1 = myRotateTensor(input1, self.kernel_size) # b x c x h x w
        filters1 = input1 # b x c x h x w
        input2_dis = Fun.pad(input2, (self.max_disp, 0, 0, 0), "constant", 0).data  # b x c x h x (max_disp/8+w)

        filters = Variable(torch.FloatTensor(self.kernel_size, self.kernel_size).fill_(1).view(1, 1, self.kernel_size,
                                                                                                self.kernel_size)).cuda(self.gpu)  # double -> float out x in x h x w

        first = True
        for d in reversed(range(self.max_disp + 1)):
            grad_slice = grad_output[:,self.max_disp-d,:,:]# b x h x w
            grad_slice = torch.unsqueeze(grad_slice, dim=1)# b x 1 x h x w

            #filters2 = myRotateTensor(input2_dis[:,:,:,d:d+w],self.kernel_size) # b x c x h x w            print filters2.shape
            filters2 = input2_dis[:,:,:,d:d+w]

            # sum slice and convolution
            grad_slice = Fun.conv2d(Variable(grad_slice), filters, padding=self.padding).data
            grad_slice_expand = grad_slice.expand(b, c, h, w)


            output1 = torch.mul(grad_slice_expand, filters2) # element mul b x c x h x w
            output2 = torch.mul(grad_slice_expand, filters1)  # element mul b x c x h x w

            output2 = Fun.pad(output2, (0, self.max_disp-d, 0, 0), "constant", 0).data  # b x c x h x (w+self.max_disp/8-d)
            output2 = output2[:,:,:,self.max_disp-d:self.max_disp-d+w]


            if not first:
                grad_input1 = torch.add(grad_input1, output1)
                grad_input2 = torch.add(grad_input2, output2)
            else:
                grad_input1 = output1
                grad_input2 = output2
                first = False


        grad_inputs = torch.cat((grad_input1, grad_input2), dim=1)

        return grad_inputs



class Correlation(torch.nn.Module):
    def __init__(self, kernelSize=3, stride=1, padding=1, maxDisp=192/8, corrMultiply=1, gpu=0):
        super(Correlation, self).__init__()
        self.kernel_size = kernelSize
        self.stride = stride
        self.padding = padding
        self.max_disp = maxDisp
        self.corr_multiply = corrMultiply
        self.gpu = gpu

    def forward(self, inputs):

        result = CorrelationFunction(self.kernel_size, self.stride, self.padding, self.max_disp, self.corr_multiply, self.gpu)(inputs)

        return result




def corr(kernelSize=3, stride=1, padding=1, maxDisp=192/8, corrMultiply=1, gpu=0):
    return torch.nn.Sequential(
        Correlation(
            kernelSize = kernelSize,
            stride = stride,
            padding = padding,
            maxDisp = maxDisp,
            corrMultiply = corrMultiply,
            gpu = gpu
        ),
        torch.nn.LeakyReLU(0.1, inplace=True)

    )

def conv(inChannels, outChannels, kernelSize=3, stride=1, isBatchNorm=True):
    if isBatchNorm:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=kernelSize,
                stride=stride,
                padding=(kernelSize-1)//2,
                bias=False
            ),
            torch.nn.BatchNorm2d(outChannels),
            torch.nn.ReLU(inplace=True)

        )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=kernelSize,
                stride=stride,
                padding=(kernelSize - 1)//2,
                bias=True
            ),
            torch.nn.ReLU(inplace=True)

        )

def shutcut(inChannels, outChannels, kernelSize=1, stride=1, isBias = True, isBatchNorm=False):
    if isBatchNorm:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=kernelSize,
                stride=stride,
                padding=(kernelSize-1)//2,
                bias=isBias
            ),
            torch.nn.BatchNorm2d(outChannels),

        )
    else:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=kernelSize,
                stride=stride,
                padding=(kernelSize - 1)//2,
                bias=isBias
            )

        )


def maxPooling(kernelSize=2, stride=2):
    return torch.nn.MaxPool2d(
        kernel_size=kernelSize,
        stride=stride
    )

def fc(inChannels, outChannels, kernelSize=7, stride=1, isBias=True, isDropOut=True, isReLU=True, isBatchNorm=False):
    if isBatchNorm:
        if isDropOut:
            if isReLU:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.BatchNorm2d(outChannels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout2d(p=0.5)
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.BatchNorm2d(outChannels),
                    torch.nn.Dropout2d(p=0.5)
                )

        else:
            if isReLU:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.BatchNorm2d(outChannels),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.BatchNorm2d(outChannels)
                )
    else:
        if isDropOut:
            if isReLU:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout2d(p=0.5)
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.Dropout2d(p=0.5)
                )

        else:
            if isReLU:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    ),
                    torch.nn.ReLU(inplace=True)
                )
            else:
                return torch.nn.Sequential(
                    torch.nn.Conv2d(
                        in_channels=inChannels,
                        out_channels=outChannels,
                        kernel_size=kernelSize,
                        stride=stride,
                        padding=(kernelSize - 1) // 2,
                        bias=isBias
                    )
                )





def deconv(inChannels, outChannels, kernelSize=4, stride=2,bias=False, output_padding=0, isBatchNorm=False):
    if isBatchNorm:
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=kernelSize,
                stride=stride,
                padding=(kernelSize - 1) // 2,
                bias=bias,
                output_padding=output_padding
            ),
            torch.nn.BatchNorm2d(outChannels),
            torch.nn.LeakyReLU(0.1,inplace=True)
        )

    else:
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels=inChannels,
                out_channels=outChannels,
                kernel_size=kernelSize,
                stride=stride,
                padding=(kernelSize - 1) // 2,
                bias=bias,
                output_padding = output_padding
            ),
            #torch.nn.LeakyReLU(0.1,inplace=True)
        )


def logSoftMax(dim=1):
    return torch.nn.LogSoftmax(dim=dim)

def predict_seg(inChannels, outChannels=2, kernelSize=1, stride=1, bias=False):
    return torch.nn.Sequential(
        torch.nn.Conv2d(
        in_channels=inChannels,
        out_channels=outChannels,
        kernel_size=kernelSize,
        stride=stride,
        padding=(kernelSize - 1) // 2,
        bias=bias
        ),
        # torch.nn.LeakyReLU(0,inplace=True)
        #torch.nn.ReLU(inplace=True)
    )

def upsampled_seg(inChannels=2,outChannels=2,kernelSize=4,stride=2,bias=False):
    return torch.nn.ConvTranspose2d(
        in_channels=inChannels,
        out_channels=outChannels,
        kernel_size=kernelSize,
        stride=stride,
        padding=(kernelSize - 1) // 2,
        bias=bias
    )



class KittiSeg7(torch.nn.Module):
    def __init__(self, isBatchNorm=True, gpu=0):
        super(KittiSeg7, self).__init__()

        self.isBatchNorm = isBatchNorm
        self.conv1_1 = conv(inChannels=3, outChannels=64, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv1_2 = conv(inChannels=64, outChannels=64, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool1_2 = maxPooling(kernelSize=2, stride=2)


        self.conv2_1 = conv(inChannels=64, outChannels=128, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv2_2 = conv(inChannels=128, outChannels=128, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool2_2 = maxPooling(kernelSize=2, stride=2)


        self.conv3_1 = conv(inChannels=128, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_2 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_3 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool3_3 = maxPooling(kernelSize=2, stride=2)

        self.conv4_1 = conv(inChannels=256, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool4_3 = maxPooling(kernelSize=2, stride=2)


        self.conv5_1 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool5_3 = maxPooling(kernelSize=2, stride=2)

        self.fc6 = fc(inChannels=512, outChannels=4096, kernelSize=7, stride=1, isBias=True, isDropOut=True, isReLU=True)
        self.fc7 = fc(inChannels=4096, outChannels=4096, kernelSize=1, stride=1, isBias=True, isDropOut=True, isReLU=True)
        self.score_fr7 = fc(inChannels=4096, outChannels=2, kernelSize=1, stride=1, isBias=True, isDropOut=False, isReLU=False)
        #self.argmax =





        self.deconv5 = deconv(inChannels=2, outChannels=2)
        self.deconv4 = deconv(inChannels=2, outChannels=2)
        self.deconv3 = deconv(inChannels=2, outChannels=2, kernelSize=16, stride=8, output_padding=6)

        self.shutcut4 = shutcut(inChannels=512, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=True)
        self.shutcut3 = shutcut(inChannels=256, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=True)

        self.predict5 = predict_seg(inChannels=2, outChannels=2, kernelSize=3, stride=1, bias=False)
        self.predict4 = predict_seg(inChannels=2, outChannels=2, kernelSize=3, stride=1, bias=False)
        self.predict3 = predict_seg(inChannels=2, outChannels=2, kernelSize=3, stride=1, bias=False)

        #has a problem
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

            if isinstance(m, torch.nn.ConvTranspose2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

        #self.upsample1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, Img):

        #left part
        out_conv1_1 = self.conv1_1(Img)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        maxpool1_2 = self.maxpool1_2(out_conv1_2)

        out_conv2_1 = self.conv2_1(maxpool1_2)
        out_conv2_2 = self.conv2_2(out_conv2_1)
        out_maxpool2_2 = self.maxpool2_2(out_conv2_2)

        out_conv3_1 = self.conv3_1(out_maxpool2_2)
        out_conv3_2 = self.conv3_2(out_conv3_1)
        out_conv3_3 = self.conv3_3(out_conv3_2)
        out_maxpool3_3 = self.maxpool3_3(out_conv3_3)

        out_conv4_1 = self.conv4_1(out_maxpool3_3)
        out_conv4_2 = self.conv4_2(out_conv4_1)
        out_conv4_3 = self.conv4_3(out_conv4_2)
        out_maxpool4_3 = self.maxpool4_3(out_conv4_3)

        out_conv5_1 = self.conv5_1(out_maxpool4_3)
        out_conv5_2 = self.conv5_2(out_conv5_1)
        out_conv5_3 = self.conv5_3(out_conv5_2)
        out_maxpool5_3 = self.maxpool5_3(out_conv5_3)

        out_fc6 = self.fc6(out_maxpool5_3)
        out_fc7 = self.fc7(out_fc6)
        out_scorefr = self.score_fr7(out_fc7)



        out_deconv5 = self.deconv5(out_scorefr)
        out_shutcut4 = self.shutcut4(out_maxpool4_3)
        out_add4 = torch.add(out_deconv5, out_shutcut4)

        pre4 = self.predict4(out_add4)


        out_deconv4 = self.deconv4(out_add4)
        out_shutcut3 = self.shutcut3(out_maxpool3_3)
        out_add3 = torch.add(out_deconv4, out_shutcut3)

        pre3 = self.predict3(out_add3)

        out_deconv3 = self.deconv3(out_add3)#

        #pre= self.argmax(out_deconv3)


        return out_deconv3


class KittiSeg5(torch.nn.Module):
    def __init__(self, isBatchNorm=True, gpu=0):
        super(KittiSeg5, self).__init__()

        self.isBatchNorm = isBatchNorm
        self.conv1_1 = conv(inChannels=3, outChannels=64, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv1_2 = conv(inChannels=64, outChannels=64, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool1_2 = maxPooling(kernelSize=2, stride=2)


        self.conv2_1 = conv(inChannels=64, outChannels=128, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv2_2 = conv(inChannels=128, outChannels=128, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool2_2 = maxPooling(kernelSize=2, stride=2)


        self.conv3_1 = conv(inChannels=128, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_2 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_3 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool3_3 = maxPooling(kernelSize=2, stride=2)

        self.conv4_1 = conv(inChannels=256, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool4_3 = maxPooling(kernelSize=2, stride=2)


        self.conv5_1 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.maxpool5_3 = maxPooling(kernelSize=2, stride=2)


        self.score_fr5 = fc(inChannels=512, outChannels=2, kernelSize=1, stride=1, isDropOut=False, isReLU=False, isBatchNorm=True)
        #self.argmax =


        self.deconv5 = deconv(inChannels=2, outChannels=2)
        self.deconv4 = deconv(inChannels=2, outChannels=2)
        self.deconv3 = deconv(inChannels=2, outChannels=2, kernelSize = 16, stride = 8, output_padding=6)

        self.shutcut4 = shutcut(inChannels=512, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=True)
        self.shutcut3 = shutcut(inChannels=256, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=True)

        self.predict5 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)
        self.predict4 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)
        self.predict3 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)

        #has a problem
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

            if isinstance(m, torch.nn.ConvTranspose2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

        #self.upsample1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, Img):

        #left part
        out_conv1_1 = self.conv1_1(Img)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        maxpool1_2 = self.maxpool1_2(out_conv1_2)

        out_conv2_1 = self.conv2_1(maxpool1_2)
        out_conv2_2 = self.conv2_2(out_conv2_1)
        out_maxpool2_2 = self.maxpool2_2(out_conv2_2)

        out_conv3_1 = self.conv3_1(out_maxpool2_2)
        out_conv3_2 = self.conv3_2(out_conv3_1)
        out_conv3_3 = self.conv3_3(out_conv3_2)
        out_maxpool3_3 = self.maxpool3_3(out_conv3_3)

        out_conv4_1 = self.conv4_1(out_maxpool3_3)
        out_conv4_2 = self.conv4_2(out_conv4_1)
        out_conv4_3 = self.conv4_3(out_conv4_2)
        out_maxpool4_3 = self.maxpool4_3(out_conv4_3)

        out_conv5_1 = self.conv5_1(out_maxpool4_3)
        out_conv5_2 = self.conv5_2(out_conv5_1)
        out_conv5_3 = self.conv5_3(out_conv5_2)
        out_maxpool5_3 = self.maxpool5_3(out_conv5_3)

        out_scorefr = self.score_fr5(out_maxpool5_3)

        out_deconv5 = self.deconv5(out_scorefr)
        out_shutcut4 = self.shutcut4(out_maxpool4_3)
        out_add4 = torch.add(out_deconv5, out_shutcut4)
        pre4 = self.predict4(out_add4)


        out_deconv4 = self.deconv4(out_add4)
        out_shutcut3 = self.shutcut3(out_maxpool3_3)
        out_add3 = torch.add(out_deconv4, out_shutcut3)
        pre3 = self.predict3(out_add3)

        out_deconv3 = self.deconv3(out_add3)#

        # pre= self.argmax(out_deconv3)


        return out_deconv3


class KittiSegCNN(torch.nn.Module):
    def __init__(self, isBatchNorm=True, gpu=0):
        super(KittiSegCNN, self).__init__()

        self.isBatchNorm = isBatchNorm
        self.conv1_1 = conv(inChannels=3, outChannels=64, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv1_2 = conv(inChannels=64, outChannels=64, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.conv2_1 = conv(inChannels=64, outChannels=128, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv2_2 = conv(inChannels=128, outChannels=128, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.conv3_1 = conv(inChannels=128, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_2 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_3 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)

        self.conv4_1 = conv(inChannels=256, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.conv5_1 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.score_fr5 = fc(inChannels=512, outChannels=2, kernelSize=1, stride=1, isDropOut=False, isReLU=False, isBatchNorm=True)
        #self.argmax =


        self.deconv5 = deconv(inChannels=2, outChannels=2, isBatchNorm=self.isBatchNorm)
        self.deconv4 = deconv(inChannels=2, outChannels=2, kernelSize = 8, stride = 4, output_padding=2, isBatchNorm=self.isBatchNorm)
        self.deconv3 = deconv(inChannels=2, outChannels=2, kernelSize = 8, stride = 4, output_padding=2, isBatchNorm=self.isBatchNorm)

        self.shutcut4 = shutcut(inChannels=512, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=self.isBatchNorm)
        self.shutcut3 = shutcut(inChannels=256, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=self.isBatchNorm)
        self.shutcut2 = shutcut(inChannels=128, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=self.isBatchNorm)

        self.predict5 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)
        self.predict4 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)
        self.predict3 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)



        #has a problem
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

            if isinstance(m, torch.nn.ConvTranspose2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

        #self.upsample1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, Img):

        #left part
        out_conv1_1 = self.conv1_1(Img)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        #maxpool1_2 = self.maxpool1_2(out_conv1_2)

        out_conv2_1 = self.conv2_1(out_conv1_2)
        out_conv2_2 = self.conv2_2(out_conv2_1)
        #out_maxpool2_2 = self.maxpool2_2(out_conv2_2)

        out_conv3_1 = self.conv3_1(out_conv2_2)
        out_conv3_2 = self.conv3_2(out_conv3_1)
        out_conv3_3 = self.conv3_3(out_conv3_2)
        #out_maxpool3_3 = self.maxpool3_3(out_conv3_3)

        out_conv4_1 = self.conv4_1(out_conv3_3)
        out_conv4_2 = self.conv4_2(out_conv4_1)
        out_conv4_3 = self.conv4_3(out_conv4_2)
        #out_maxpool4_3 = self.maxpool4_3(out_conv4_3)

        out_conv5_1 = self.conv5_1(out_conv4_3)
        out_conv5_2 = self.conv5_2(out_conv5_1)
        out_conv5_3 = self.conv5_3(out_conv5_2)
        #out_maxpool5_3 = self.maxpool5_3(out_conv5_3)

        out_scorefr = self.score_fr5(out_conv5_3)


        out_deconv5 = self.deconv5(out_scorefr)
        out_shutcut4 = self.shutcut4(out_conv4_3)
        out_add4 = torch.add(out_deconv5, out_shutcut4)
        pre4 = self.predict4(out_add4)

        out_deconv4 = self.deconv4(out_add4)
        out_shutcut2 = self.shutcut2(out_conv2_2)
        out_add3 = torch.add(out_deconv4, out_shutcut2)
        pre3 = self.predict3(out_add3)

        out_deconv3 = self.deconv3(out_add3)#

        # pre= self.argmax(out_deconv3)


        return out_deconv3



class KittiSegWCNN(torch.nn.Module):
    def __init__(self, isBatchNorm=True, gpu=0):
        super(KittiSegWCNN, self).__init__()

        self.isBatchNorm = isBatchNorm
        self.conv1_1 = conv(inChannels=3, outChannels=64, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv1_2 = conv(inChannels=64, outChannels=64, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.conv2_1 = conv(inChannels=64, outChannels=128, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv2_2 = conv(inChannels=128, outChannels=128, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.conv3_1 = conv(inChannels=128, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_2 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv3_3 = conv(inChannels=256, outChannels=256, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)

        self.conv4_1 = conv(inChannels=256, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv4_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.conv5_1 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_2 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=1, isBatchNorm=self.isBatchNorm)
        self.conv5_3 = conv(inChannels=512, outChannels=512, kernelSize=3, stride=2, isBatchNorm=self.isBatchNorm)


        self.score_fr5 = fc(inChannels=512, outChannels=2, kernelSize=1, stride=1, isDropOut=False, isReLU=False, isBatchNorm=True)
        #self.argmax =


        self.deconv5 = deconv(inChannels=2, outChannels=2, isBatchNorm=self.isBatchNorm)
        self.deconv4 = deconv(inChannels=2, outChannels=2, kernelSize = 8, stride = 4, output_padding=2, isBatchNorm=self.isBatchNorm)
        self.deconv3 = deconv(inChannels=2, outChannels=2, kernelSize = 8, stride = 4, output_padding=2, isBatchNorm=self.isBatchNorm)

        self.shutcut4 = shutcut(inChannels=512, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=self.isBatchNorm)
        self.shutcut3 = shutcut(inChannels=256, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=self.isBatchNorm)
        self.shutcut2 = shutcut(inChannels=128, outChannels=2, kernelSize=1, stride=1, isBias=True, isBatchNorm=self.isBatchNorm)

        self.predict5 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)
        self.predict4 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)
        self.predict3 = predict_seg(inChannels=2, outChannels=2, kernelSize=1, stride=1, bias=False)

        self.logSoftMax = logSoftMax()



        #has a problem
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

            if isinstance(m, torch.nn.ConvTranspose2d):
                if m.bias is not None:
                    torch.nn.init.uniform(m.bias)
                torch.nn.init.xavier_uniform(m.weight)

        #self.upsample1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')


    def forward(self, Img):

        #left part
        out_conv1_1 = self.conv1_1(Img)
        out_conv1_2 = self.conv1_2(out_conv1_1)
        #maxpool1_2 = self.maxpool1_2(out_conv1_2)

        out_conv2_1 = self.conv2_1(out_conv1_2)
        out_conv2_2 = self.conv2_2(out_conv2_1)
        #out_maxpool2_2 = self.maxpool2_2(out_conv2_2)

        out_conv3_1 = self.conv3_1(out_conv2_2)
        out_conv3_2 = self.conv3_2(out_conv3_1)
        out_conv3_3 = self.conv3_3(out_conv3_2)
        #out_maxpool3_3 = self.maxpool3_3(out_conv3_3)

        out_conv4_1 = self.conv4_1(out_conv3_3)
        out_conv4_2 = self.conv4_2(out_conv4_1)
        out_conv4_3 = self.conv4_3(out_conv4_2)
        #out_maxpool4_3 = self.maxpool4_3(out_conv4_3)

        out_conv5_1 = self.conv5_1(out_conv4_3)
        out_conv5_2 = self.conv5_2(out_conv5_1)
        out_conv5_3 = self.conv5_3(out_conv5_2)
        #out_maxpool5_3 = self.maxpool5_3(out_conv5_3)

        out_scorefr = self.score_fr5(out_conv5_3)


        out_deconv5 = self.deconv5(out_scorefr)
        out_shutcut4 = self.shutcut4(out_conv4_3)
        out_add4 = torch.add(out_deconv5, out_shutcut4)
        pre4 = self.predict4(out_add4)

        out_deconv4 = self.deconv4(out_add4)
        out_shutcut2 = self.shutcut2(out_conv2_2)
        out_add3 = torch.add(out_deconv4, out_shutcut2)
        pre3 = self.predict3(out_add3)

        out_deconv3 = self.deconv3(out_add3)#

        # pre= self.argmax(out_deconv3)

        out_logSoftMax = self.logSoftMax(out_deconv3)


        return out_logSoftMax
