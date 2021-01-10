import os
import torch
from testKiSegDataGenerator import dataGenerator,getMergeFreeSpace
from kiSegModule import KittiSeg7, KittiSeg5
from kiSegLoss import multiscaleCrossentropyloss, crossentropyloss, AverageMeter, getMultiscale_weights, PR
import time
import torch.optim
# from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
####################################################################################
# Some parameters can change
LR = 0.0001
EPOCH = 250
GPU = 3
BATCH_SIZE = 4

dataSet = ['jingCheng'] ## 'um', 'jingCheng'

MODEL = 'KittiSegWCNN' # 'KittiSeg7', 'KittiSeg5', 'KittiSegCNN','KittiSegCNN1','KittiSegWCNN'
FINETUNE = False
CROPSIZE = (704,1280)#(352,1024)
###################################################################################


dataDir = './data'
if FINETUNE:
    EPOCH = 250
    MILESTONES = [100, 150, 200]  # 100,150,200
    #LR = LR
else:
    MILESTONES = [100,150,200] # 100,150,200
SAVEPATH='./save'



SPARSE = False
if 'KITTI' in dataSet:
    SPARSE = True



pause = False
def onclick(event):
    global pause
    pause = not pause


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

def train(train_loader, model, optimizer, epoch, multiscale_weights, train_writer=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_EPEs = AverageMeter()
    n_iter=0

    #DispNetS
    # if epoch<10:
    #     MY_MULTISCALE_WEIGHTS=[1.0/32, 1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0/2]
    # elif epoch>=10 and epoch<15:
    #     MY_MULTISCALE_WEIGHTS = [0.0, 1.0 / 16, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 2]
    # elif epoch >=15 and epoch<20:
    #     MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 1.0 / 8, 1.0 / 8, 1.0 / 4, 1.0 / 2]
    # elif epoch >=25 and epoch<30:
    #     MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 0.0, 1.0 / 4, 1.0 / 4, 1.0 / 2]
    # elif epoch >=30 and epoch<35:
    #     MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 1.0 / 2, 1.0 / 2]
    # else:
    #     MY_MULTISCALE_WEIGHTS = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


    model.train()

    for step, (input, target, info) in enumerate(train_loader):

        n_iter = len(train_loader)*epoch + step
        start = time.clock()
        target = target.cuda(GPU)  # ,async=True
        input = [j.cuda(GPU) for j in input]

        input_var = torch.autograd.Variable(torch.cat(input, 1))
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        loss = multiscaleCrossentropyloss(mulweights=None, sparse=False)(output, target_var.squeeze())
        flow2_EPE = crossentropyloss()(output, target_var.squeeze(dim=1))

        losses.update(loss.data[0], target.size(0))
        losses_EPEs.update(flow2_EPE.data[0], target.size(0))

        if train_writer is not None:
            train_writer.add_scalar('train_loss', loss.data[0], n_iter)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = (time.clock() - start)
        batch_time.update(elapsed)

        print 'In train-- ', 'EPOCH: ', epoch, ' STEP: ', step, ' LR: ', LR, ' TIME: ', elapsed, ' LOSS: ', loss.data[0]


    return losses.avg, losses_EPEs.avg

def validate(val_loader, model, epoch, val_writers=None):

    fps=3
    fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter('FreeSpace.avi', fourcc, fps, (1280, 704))  # (1360,480)


    fig = plt.figure()
    plt.ion()

    fig.canvas.mpl_connect('button_press_event', onclick)

    batch_time = AverageMeter()
    losses_EPEs = AverageMeter()

    model.eval()
    for step, (oriImage, input, target, info) in enumerate(val_loader):

        target = target.cuda(GPU, async=True)
        input_var = torch.autograd.Variable(torch.cat(input,1).cuda(GPU), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        start = time.clock()
        output = model(input_var)
        elapsed = (time.clock() - start)
        batch_time.update(elapsed)

        preci, rcal = PR(output, target, ci=1)

        for imInd in range(oriImage[0].shape[0]):

            plt.subplot(2, 1, 1)
            # plt.imshow(oriImage[0][imInd].permute(2,0,1)/255.0)

            preShowImg = getMergeFreeSpace(oriImage[0][imInd].numpy(), output[imInd].cpu().data.permute(1,2,0).numpy())
            plt.imshow(preShowImg/255)

            videoWriter.write(np.uint8(preShowImg))

            # plt.imsave('1.png', oriImage[0][imInd].numpy()/255)
            # plt.imsave('1t.png',preShowImg/255)



            plt.subplot(2, 1, 2)
            #plt.imshow(oriImage[1][imInd].permute(2,0,1)/255.0)
            targetShowImg = getMergeFreeSpace(oriImage[0][imInd].numpy(), oriImage[1][imInd].numpy())
            plt.imshow(targetShowImg/255)

            plt.show()
            fig.canvas.get_tk_widget().update()
            time.sleep(5)




        losses_EPE = crossentropyloss( )(output, target_var.squeeze(dim=1))

        # record EPE
        losses_EPEs.update(losses_EPE.data[0], target.size(0))


        # if val_writers is not None:
        #     val_writers.add_image('GroundTruth', flow2rgb(target[0].cpu().numpy(),MAXDISP), 0) # has problem
        #     val_writers.add_image('Inputs', input[0][0].numpy().transpose(1, 2, 0) + np.array([0.411, 0.432, 0.45]),0)
        #     val_writers.add_image('Inputs', input[1][0].numpy().transpose(1, 2, 0) + np.array([0.411, 0.432, 0.45]), 1)
        #     val_writers.add_image('FlowNet Outputs', flow2rgb(output.data[0].cpu().numpy(),MAXDISP), epoch)


        print 'In val-- ', 'EPOCH: ', epoch, ' STEP: ', step, ' LR: ', LR, ' TIME: ', elapsed, ' LOSS: ', losses_EPE.data[0],\
            'PRECISION: ', preci, 'RECALL: ',rcal

    videoWriter.release()

    return losses_EPEs.avg



def main():

    #def dataGenerator(data_dir,  batch_size, shuffle, nem_workers, disp, data_name='KITTI', split=None):



    # train_writer = SummaryWriter(os.path.join(SAVEPATH, 'train'))
    # val_writer = SummaryWriter(os.path.join(SAVEPATH, 'val'))



    dataloaders = dataGenerator(data_dir=dataDir, batch_size=BATCH_SIZE, data_name=dataSet,
                                shuffle=True, num_workers=4, split = 0.1, crop_size=CROPSIZE)

    kittiSeg = torch.load(MODEL + '.pkl')
    kittiSeg.cuda(GPU)

    optimizer = torch.optim.Adam(kittiSeg.parameters(), lr=LR, betas=(0.9, 0.999))

    best_EPE = -10000;

    traLoss = []
    valLoss = []
    plt.ion()
    for epoch in range(EPOCH):

        val_EPE = validate(val_loader = dataloaders['val'],
                           model = kittiSeg,
                           epoch = epoch)

        # val_writer.add_scalar('mean EPE', val_EPE, epoch)




if __name__ == "__main__":
    main()