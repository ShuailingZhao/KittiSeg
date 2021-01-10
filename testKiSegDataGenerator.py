import os
import torch
from torchvision import transforms
import numpy as np
import numbers
import random
from torch.utils.data import Dataset, DataLoader
import glob
#rom PIL import Image
from scipy.ndimage import imread
import re



class myArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __init__(self, type='float32'):
        self.type = type

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))
        array = np.transpose(array, (2, 0, 1))
        # handle numpy array
        tensor = torch.from_numpy(array)
        # put it from HWC to CHW format
        if('float32' ==self.type):
            return tensor.float()
        else:
            return tensor.long()



class myCompose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target


class myRandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x1 = random.randint(0, w - tw)
        y1 = 16#random.randint(0, h - th)
        inputs[0] = inputs[0][y1: y1 + th,x1: x1 + tw]# h x w x _ the last dementation can not write explicitly
        return inputs, target[y1: y1 + th,x1: x1 + tw]



class myRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.fliplr(inputs[0]))
            inputs[1] = np.copy(np.fliplr(inputs[1]))
            target = np.copy(np.fliplr(target))
            target[:,:,0] *= -1
        return inputs,target


class myRandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            inputs[0] = np.copy(np.flipud(inputs[0]))
            inputs[1] = np.copy(np.flipud(inputs[1]))
            target = np.copy(np.flipud(target))
            #target[:,:,1] *= -1
        return inputs,target



class myRandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target


class myRandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, inputs, target):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        inputs[0] *= (1 + random_std)
        inputs[0] += random_mean

        inputs[1] *= (1 + random_std)
        inputs[1] += random_mean

        inputs[0] = inputs[0][:,:,random_order]
        inputs[1] = inputs[1][:,:,random_order]

        return inputs, target


def load_pfm(fileName):
	color = None
	width = None
	height = None
	scale = None
	endian = None

	file = open(fileName)
	header = file.readline().rstrip()
	if header == 'PF':
		color = True
	elif header == 'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip())
	if scale < 0: # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>' # big-endian

	data = np.fromfile(file, endian + 'f')
	shape = (height, width, 3) if color else (height, width)
	return np.flip(np.reshape(data, shape),0).copy(), scale

def split2list(images, split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert(len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0,1,len(images)) > split
    else:
        print "Wrong input split"

    train_images = [sample for sample, split in zip(images, split_values) if not split]
    test_images = [sample for sample, split in zip(images, split_values) if split]
    return train_images, test_images

def getFile_name(root_dir, ext='.png'):
    L = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if -1 == dirpath.find('right') and os.path.splitext(file)[1] == ext:# only using the left disparity
                L.append(os.path.join(dirpath, file))
    return L

def getImgList(root_dir, dataset='um', split=0.9):
    imgList = []
    if 'um' in dataset:
        imglist = getFile_name(os.path.join(root_dir, 'um', 'image_2'), ext='.png')
        for imgfile in imglist:
            labelfile = imgfile.replace('image_2', 'gt_image_2_grey_label')
            labelfile = labelfile.replace('_0', '_road_0')

            imgList.append([imgfile, labelfile])

    if 'jingCheng' in dataset:
        labellist = getFile_name(os.path.join(root_dir, 'jingCheng', 'gt_image_2_grey_label'), ext='.png')
        for labelfile in labellist:
            imgfile = labelfile.replace('gt_image_2_grey_label', 'image_2')

            imgList.append([imgfile, labelfile])

    imgList.sort()
    train_list, test_list = split2list(imgList, split=split)
    return train_list, test_list



def flow2rgb(flow_map, max_value):
    _, h, w = flow_map.shape
    rgb_map = np.ones((h,w,3)).astype(np.float32)
    normalized_flow_map = flow_map/max_value
    rgb_map[:,:,0] += normalized_flow_map[0]
    rgb_map[:,:,1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[0])
    rgb_map[:,:,2] += normalized_flow_map[0]
    return rgb_map.clip(0,1)


def getMergeFreeSpace(oriImg, target):
    assert (isinstance(oriImg, np.ndarray))
    assert (isinstance(target, np.ndarray))

    mergetImg = np.copy(oriImg)
    g = mergetImg[:,:,1]

    if(2 == target.shape[2]):
        g[np.argmax(target, axis=2) == 1] = 255
    else:
        g[target[:,:,0]== 1] = 255


    return mergetImg


def getShowImg(info, output):
    lImgs = []
    rImgs = []
    tImgs = []
    pImgs = []
    for ind in len(info):
        lImg = imread(info[ind]['limgName'])
        rImg = imread(info[ind]['rimgName'])
        tImg = imread(info[ind]['gimgName'])
        lImgs.append(lImg)
        rImgs.append(rImg)
        tImgs.append(tImg)



class LandmarksDataset(Dataset):
    """Stereo Landmarks dataset."""

    def __init__(self, img_list=None, transform = None, ltransform = None, co_transform = None):
        """
        Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
        on a sample.
        """
        self.img_list = img_list
        self.transform = transform
        self.target_transform = ltransform
        self.co_transform = co_transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        iimg_name = self.img_list[idx][0]
        gimg_name = self.img_list[idx][1]
        iimage = imread(iimg_name).astype(np.float32)
        gimage = imread(gimg_name).astype(np.float32)

        gimage = np.expand_dims(gimage, axis=2)


        # CROPSIZE = (256, 256)
        # # sample = {'limage': limage, 'rimage': rimage, 'groudtruth': gimage}
        # w, h = limage.size
        #
        # top = np.random.randint(0, h - CROPSIZE[1] + 1)
        # left = np.random.randint(0, w - CROPSIZE[0] + 1)
        #
        # limageROI = {'image': limage, 'leftTop': (left, top)}
        # rimageROI = {'image': rimage, 'leftTop': (left, top)}
        # gimageROI = {'image': gimage, 'leftTop': (left, top)}
        inputs = [iimage]

        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, gimage)
            oriInputs = [np.copy(inputs[0]), np.copy(target)]

        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
        if self.target_transform is not None:
            target = self.target_transform(target)

        info = {'limgName': iimg_name, 'gimgName': gimg_name}

        return oriInputs, inputs, target, info


def dataGenerator(data_dir, batch_size, data_name = 'KITTI', shuffle=False, num_workers=4, split=0.9, crop_size=(320,448)):

    input_transform = transforms.Compose([
        myArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])
    target_transform = transforms.Compose([
        myArrayToTensor('long'),
        #transforms.Normalize(mean=[0], std=[maxDisp])
    ])

    #if 'KITTI' == data_name:
    if 'KITTI' in data_name:
        co_transform = myCompose([
            myRandomCrop(crop_size),#(384,1024),(320,448)
            myRandomVerticalFlip(),
            #myRandomHorizontalFlip(),
            myRandomColorWarp(30, 0.3)
        ])
    else:
        co_transform = myCompose([
            #myRandomTranslate(10),     can do
            #myRandomRotate(10, 5),
            myRandomCrop(crop_size),#(384,1024),(320,448)
            #myRandomVerticalFlip(),
            #myRandomHorizontalFlip(),
            #myRandomColorWarp(0, 0)
        ])

    train_list, test_list = getImgList(root_dir =data_dir, dataset=data_name, split=split)



    # listNum = len(train_list)//batch_size*batch_size
    # train_list = train_list[0:listNum]
    #
    # listNum = len(test_list)//batch_size*batch_size
    # test_list = test_list[0:listNum]
    #
    # print '-------------------------'
    # print len(train_list)
    # print len(test_list)


    image_datasets = {'train':LandmarksDataset(img_list=train_list, transform = input_transform, ltransform = target_transform, co_transform = co_transform),
                      'val': LandmarksDataset(img_list=test_list, transform = input_transform, ltransform = target_transform, co_transform = co_transform)}
    dataloaders = {'train':torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,shuffle=shuffle, num_workers=num_workers, pin_memory=True),
                   'val':torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,shuffle=False, num_workers=num_workers, pin_memory=True)}

    return dataloaders

