import torch
import torch.cuda
import argparse
import os
import sys
import cv2
import re
import math
import model.network as net
import model.eval as eva
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import time
import visdom
import numpy as np
from PIL import Image
from torchvision import transforms as T 

def parse_args():
    parser = argparse.ArgumentParser(description='train a network')
    parser.add_argument('--dataset',help='training set config file',default='dataset/kitti.data',type=str)
    parser.add_argument('--netcfg',help='the network config file',default='cfg/complex-yolo.cfg',type=str)
    parser.add_argument('--weight',help='the network weight file',default='backup/complex-yolo.backup',type=str)
    parser.add_argument('--vis',help='visdom the training process',default=1,type=int)
    parser.add_argument('--img',help='the input file for detection',default='007000.png',type=str)
    parser.add_argument('--thresh',help='the input file for detection',default=0.3,type=float)
    parser.add_argument('--cuda',help='use the GPU',default=1,type=int)
    args = parser.parse_args()
    return args

def parse_dataset_cfg(cfgfile):
    with open(cfgfile,'r') as fp:
        p1 = re.compile(r'classes=\d')
        p2 = re.compile(r'train=')
        p3 = re.compile(r'names=')
        p4 = re.compile(r'backup=')
        for line in fp.readlines():
            a = line.replace(' ','').replace('\n','')
            if p1.findall(a):
                classes = re.sub('classes=','',a)
            if p2.findall(a):
                trainlist = re.sub('train=','',a)
            if p3.findall(a):
                namesdir = re.sub('names=','',a)
            if p4.findall(a):
                backupdir = re.sub('backup=','',a)
    return int(classes),trainlist,namesdir,backupdir

def parse_network_cfg(cfgfile):
    with open(cfgfile,'r') as fp:
        layerList = []
        layerInfo = ''
        p = re.compile(r'\[\w+\]')
        p1 = re.compile(r'#.+')
        for line in fp.readlines():
            if p.findall(line):
                if layerInfo:
                    layerList.append(layerInfo)
                    layerInfo = ''
            if line == '\n' or p1.findall(line):
                continue
            line = line.replace(' ','')
            layerInfo += line
        layerList.append(layerInfo)
    print('layer number is %d'%(layerList.__len__() - 1) )
    return layerList

def get_names(nameFile):
    with open(nameFile,'r') as fp:
        names = []
        for line in fp.readlines():
            line = line.replace(' ','').replace('\n','')
            if line != '':
                names.append(line)
    return names

def plot_boxes_cv2(image, boxes, class_names=None, color=None, fps=None):
    img = image
    colors = torch.FloatTensor([[1,0,1],[0,0,1],[0,1,1],[0,1,0],[1,1,0],[1,0,0]])
    def get_color(c, x, max_val):
        ratio = float(x)/max_val * 5
        i = int(math.floor(ratio))
        j = int(math.ceil(ratio))
        ratio = ratio - i
        r = (1-ratio) * colors[i][c] + ratio*colors[j][c]
        return int(r*255)
    width = img.shape[1]
    height = img.shape[0]
    num, _ = boxes.size()
    print(width,height)
    for i in range(num):
        box = boxes[i, :]
        im = box[5]#sin(ang)
        re = box[6]#cos(ang)
        ang = math.atan2(im, re)
        
        x = int(box[1] * width)
        y = int(box[2] * height)
        w = int(box[3] * width)
        h = int(box[4] * height)

        theta = math.atan2(w, h)
        l = math.sqrt(w**2 + h**2)

        x1 = x + l/2*(math.cos(ang - theta))
        y1 = y + l/2*(math.sin(ang - theta))
        x2 = x - l/2*(math.cos(ang - theta))
        y2 = y - l/2*(math.sin(ang - theta))
        x3 = x + l/2*(math.cos(ang + theta))
        y3 = y + l/2*(math.sin(ang + theta))
        x4 = x - l/2*(math.cos(ang + theta))
        y4 = y - l/2*(math.sin(ang + theta))
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        point3 = (int(x3), int(y3))
        point4 = (int(x4), int(y4))
        if color:
            rgb = color
        else:
            rgb = (255, 0, 0)
        if len(box) >= 9 and class_names:
            cls_conf = box[8]
            cls_id = int(box[7])
            print('%s: %f' % (class_names[cls_id], cls_conf))
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red   = get_color(2, offset, classes)
            green = get_color(1, offset, classes)
            blue  = get_color(0, offset, classes)
            str_prob = '%.2f'%cls_conf
            info = class_names[cls_id] + str_prob
            if color is None:
                rgb = (red, green, blue)
            img = cv2.putText(img, info, point1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, rgb, 1)
            img = cv2.line(img, point1, point3, rgb, 1)
            img = cv2.line(img, point1, point4, rgb, 1)
            img = cv2.line(img, point2, point3, rgb, 1)
            img = cv2.line(img, point2, point4, rgb, 1)
    if fps is None:
        savename = 'prediction.png'
        print("save plot results to %s" %savename)
        cv2.imwrite(savename, img)
    else:
        fps_info = 'fps:' + '%.2f'%fps
        img = cv2.putText(img, fps_info, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, rgb, 1)
    return img

def detect_image(image, network, thresh, names):
    pil_img = Image.open(image)
    w_im, h_im = pil_img.size
    transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    img = pil_img.resize( (network.width, network.height) )
    img = transform(img).cuda()
    img = img.view(1,3,network.height,network.width)
    pred = network(img)
    evaluator = eva.evalYolov2(network.layers[-1].flow[0], class_thresh=thresh)
    result = evaluator.forward(pred, w_im=w_im, h_im=h_im)
    print(result)
    image1 = cv2.imread(image)
    im = plot_boxes_cv2(image1, result, names)
    cv2.imshow('prediction',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 

def detect_vedio(image, network, thresh, names):
    transform = T.Compose([T.ToTensor(),T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    evaluator = eva.evalYolov2(network.layers[-1].flow[0], class_thresh=thresh)
    if image == '0':
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(image)
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
    fps = 0.0
    while(cap.isOpened()):  
        t0 = time.time()
        ret, img_raw = cap.read()
        h_im , w_im, _ = img_raw.shape
        img = cv2.resize(img_raw,(network.width, network.height))
        if ret == True:
            img = transform(img).view(1,3,network.height,network.width).cuda()
            pred = network(img)
            result = evaluator.forward(pred, w_im=w_im, h_im=h_im)
            t1 = time.time()
            fps = 1/(t1-t0)
            if result is not None:
                im = plot_boxes_cv2(img_raw, result, names, fps=fps)
            else:
                im = img_raw
            cv2.imshow('prediction',im)
            print('fps: %f'%fps)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break
    cap.release()  
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    args = parse_args()
    print(args)
    classes, trainlist, namesdir, backupdir = parse_dataset_cfg(args.dataset)
    print('%d classes in dataset'%classes)
    print('trainlist directory is ' + trainlist)
    names = get_names(namesdir)
    #step 1: parse the network
    layerList = parse_network_cfg(args.netcfg)
    netname = args.netcfg.split('.')[0].split('/')[-1]
    layer = []
    print('the depth of the network is %d'%(layerList.__len__()-1))
    network = net.network(layerList)
    #step 2: load network parameters
    network.load_weights(args.weight)
    layerNum = network.layerNum
    if args.cuda:
        network = network.cuda()
    #step 3: load data and test
    network = network.eval()
    image = args.img
    img_tail =  image.split('.')[-1] 
    if img_tail == 'jpg' or img_tail =='jpeg' or img_tail == 'png':
        detect_image(image, network, args.thresh, names)   
    elif img_tail == 'mp4' or img_tail =='mkv' or img_tail == 'avi' or img_tail =='0':
        detect_vedio( image, network, args.thresh, names)
       