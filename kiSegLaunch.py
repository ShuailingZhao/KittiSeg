import os
import torch
from kiSegDataGenerator import dataGenerator
from kiSegModule import KittiSeg7, KittiSeg5, KittiSegCNN,KittiSegWCNN
from kiSegLoss import multiscaleCrossentropyloss, crossentropyloss, AverageMeter, getMultiscale_weights, weightcrossentropyloss, Myweightcrossentropyloss,PR,checkRandomWeightIsLossOk
from balance_data_parallel import BalancedDataParallel
import time
import torch.optim
from tensorboardX import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import csv
####################################################################################
# Some parameters can change
LR = 0.001
EPOCH = 200
BATCH_SIZE = 4
useGPUs='0,1'
os.environ['CUDA_VISIBLE_DEVICES'] = useGPUs
gpu0batchSize = BATCH_SIZE/len(useGPUs.split(','))
GPU = 0


dataSet = ['jingCheng'] ## 'um', 'jingCheng'

MODEL = 'KittiSegWCNN' # 'KittiSeg7', 'KittiSeg5','KittiSegCNN','KittiSegWCNN'
FINETUNE = False
CROPSIZE = (352,1024)
showLossCurve = False
###################################################################################


dataDir = './data'
if FINETUNE:
    EPOCH = 250
    MILESTONES = [100, 150, 200]  # 100,150,200
    #LR = LR
else:
    MILESTONES = [70,130,150,170,190] # 100,150,200
SAVEPATH='./save'



SPARSE = False
if 'KITTI' in dataSet:
    SPARSE = True


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

    for step, (inputs, target, info) in enumerate(train_loader):

        n_iter = len(train_loader)*epoch + step
        start = time.clock()
#        target = target.cuda(GPU)  # ,async=True
#        inputs = [j.cuda(GPU) for j in inputs]
        inputs = [j for j in inputs]

#        input_var = torch.autograd.Variable(torch.cat(inputs, 1))
        input_var = torch.autograd.Variable(torch.cat(inputs, 1).cuda())
        target_var = torch.autograd.Variable(target.cuda())

        output = model(input_var)

        # loss = multiscaleCrossentropyloss(mulweights=None, sparse=False)(output, target_var.squeeze(dim=1)) # KittiSegCNN
        # flow2_EPE = crossentropyloss()(output, target_var.squeeze(dim=1))

        loss = Myweightcrossentropyloss(kernelSize=15, gpu=GPU)(output, target_var)
        flow2_EPE = Myweightcrossentropyloss(kernelSize=15, gpu=GPU)(output, target_var)


        losses.update(loss.data.cpu().numpy(), target.size(0))
        losses_EPEs.update(flow2_EPE.data.cpu().numpy(), target.size(0))

        # if train_writer is not None:
        #     train_writer.add_scalar('train_loss', loss.data[0], n_iter)
        if 0 == step and train_writer is not None:
            for name, param in model.named_parameters():
                train_writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        elapsed = (time.clock() - start)
        batch_time.update(elapsed)

        preci, rcal = PR(output, target, ci=1)
        print 'In train-- ', 'EPOCH: ', epoch, ' STEP: ', step, ' LR: ', LR, ' TIME: ', elapsed, ' LOSS: ', loss.data.cpu().numpy(),\
            'PRECISION: ', preci.data.cpu().numpy(), 'RECALL: ',rcal.data.cpu().numpy(),'debugLoss: ', checkRandomWeightIsLossOk(output).data.cpu().numpy()


    return losses.avg, losses_EPEs.avg

def validate(val_loader, model, epoch, val_writers=None):
    batch_time = AverageMeter()
    losses_EPEs = AverageMeter()

    model.eval()
    for step, (input, target, info) in enumerate(val_loader):
        start = time.clock()
        target = target.cuda(GPU, async=True)
#        input_var = torch.autograd.Variable(torch.cat(input,1).cuda(GPU), volatile=True)
        input_var = torch.autograd.Variable(torch.cat(input,1).cuda(), volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(input_var)

        losses_EPE = Myweightcrossentropyloss(kernelSize=15, gpu=GPU)(output, target_var)

        # record EPE
        losses_EPEs.update(losses_EPE.data.cpu().numpy(), target.size(0))

        elapsed = (time.clock() - start)
        batch_time.update(elapsed)
        # if val_writers is not None:
        #     val_writers.add_image('GroundTruth', flow2rgb(target[0].cpu().numpy(),MAXDISP), 0) # has problem
        #     val_writers.add_image('Inputs', input[0][0].numpy().transpose(1, 2, 0) + np.array([0.411, 0.432, 0.45]),0)
        #     val_writers.add_image('Inputs', input[1][0].numpy().transpose(1, 2, 0) + np.array([0.411, 0.432, 0.45]), 1)
        #     val_writers.add_image('FlowNet Outputs', flow2rgb(output.data[0].cpu().numpy(),MAXDISP), epoch)
       
        
        preci, rcal = PR(output, target, ci=1)


        print 'In val-- ', 'EPOCH: ', epoch, ' STEP: ', step, ' LR: ', LR, ' TIME: ', elapsed, ' LOSS: ', losses_EPE.data.cpu().numpy(),\
            'PRECISION: ', preci.data.cpu().numpy(), 'RECALL: ', rcal.data.cpu().numpy()

    return losses_EPEs.avg



def main():

    #def dataGenerator(data_dir,  batch_size, shuffle, nem_workers, disp, data_name='KITTI', split=None):



    train_writer = SummaryWriter(os.path.join(SAVEPATH, 'train'))



    dataloaders = dataGenerator(data_dir=dataDir, batch_size=BATCH_SIZE, data_name=dataSet,
                                shuffle=True, num_workers=4, split = 0.9, crop_size=CROPSIZE)


    if FINETUNE:
        dispNet = torch.load(MODEL+'.pkl')
        for param in dispNet.parameters():
            param.requires_grad = True
    else:
        if MODEL == 'KittiSeg7':
            kittiSeg = KittiSeg7(isBatchNorm=True, gpu=GPU)
        elif MODEL == 'KittiSeg5':
            kittiSeg = KittiSeg5(isBatchNorm=True, gpu=GPU)
        elif MODEL == 'KittiSegCNN':
            kittiSeg = KittiSegCNN(isBatchNorm=True, gpu=GPU)
        else:
            kittiSeg = KittiSegWCNN(isBatchNorm=True, gpu=GPU)

    kittiSeg = BalancedDataParallel(gpu0batchSize, kittiSeg,dim=0)
#    dispNet = torch.nn.DataParallel(dispNet)
    kittiSeg = kittiSeg.cuda()
    #print (dispNet)

    optimizer = torch.optim.Adam(kittiSeg.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.5)

    best_EPE = -10000;

    traLoss = []
    valLoss = []
    if showLossCurve:
        plt.ion()
    for epoch in range(EPOCH):
        scheduler.step()

        multiscaleWeights = getMultiscale_weights(MODEL, FINETUNE, epoch)

        train_loss, train_EPE = train(train_loader = dataloaders['train'],
              model = kittiSeg,
              optimizer = optimizer,
              epoch = epoch,
              multiscale_weights = multiscaleWeights,
              train_writer = train_writer)



        val_EPE = validate(val_loader = dataloaders['val'],
                           model = kittiSeg,
                           epoch = epoch)
        
        train_writer.add_scalars('mean EPE loss', {'trainLoss':train_EPE,'valLoss':val_EPE}, epoch)

        if best_EPE<0:
            best_EPE = val_EPE

        is_best = val_EPE < best_EPE
        best_EPE = min(val_EPE, best_EPE)

        saveNet = 'KittiSeg' + '_' + str(best_EPE) +'_'+ str(epoch) + '.pkl'
        if best_EPE<0.01:
            torch.save(kittiSeg, saveNet)

        traLoss.append(train_EPE)
        valLoss.append(val_EPE)

        if showLossCurve:
            plt.cla()
            plt.plot(traLoss, 'r-', label='traLoss')
            plt.plot(valLoss, 'g-', label='valLoss')
            plt.legend(('traLoss', 'valLoss'), loc='upper right')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.pause(0.3)




    with open('traLoss.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in traLoss:
            writer.writerow([val])

    with open('valLoss.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        for val in valLoss:
            writer.writerow([val])

    # plt.plot(traLoss, 'r-', label='traLoss')
    # plt.plot(valLoss, 'g-', label='valLoss')
    # plt.legend(('traLoss', 'valLoss'), loc='upper right')
    # plt.xlabel('Steps')
    # plt.ylabel('Loss')
    # plt.show()


    #raw_input()
    if showLossCurve:
        plt.savefig('learncurve.png')
        plt.ioff()
    train_writer.close()




if __name__ == "__main__":
    main()
