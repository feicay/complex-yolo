import torch
from torch import nn
from torch.autograd import Variable

class evalYolov2(nn.Module):
    def __init__(self, RegionLayer, nms_thresh=0.3, obj_thresh=0.4, class_thresh=0.2):
        super(evalYolov2, self).__init__()
        self.classes = RegionLayer.classes
        self.coords = RegionLayer.coords
        self.num = RegionLayer.num
        self.anchor_len = self.classes + self.coords + 1
        self.anchors = RegionLayer.anchors
        self.count = 0
        self.nms_thresh = nms_thresh
        self.obj_thresh = obj_thresh
        self.class_thresh = class_thresh
    def forward(self, pred, truth=None, w_im=1, h_im=1):
        self.AP = 0.0
        self.Recall = 0.0
        self.TP = 0
        self.TN = 0
        self.FP = 0
        batch, channels, in_height, in_width = pred.size()
        zeros = Variable(torch.zeros(1))
        one = Variable(torch.ones(1))
        width_v = torch.arange(0,in_width).view(1,in_width)
        height_v = torch.arange(0,in_height).view(in_height,1)
        if pred.is_cuda:
            width_v = width_v.cuda()
            height_v = height_v.cuda()
            zeros = zeros.cuda()
            one = one.cuda()
        i_list = width_v.expand(in_height,in_width).contiguous().view(in_height*in_width)
        j_list = height_v.expand(in_height,in_width).contiguous().view(in_height*in_width)
        result_list = []
        for b in range(batch):
            x = pred[b,:,:,:]
            box_list = []
            obj_list = []
            cls_list = []
            ang_list = []
            prob_cls_list = []
            for n in range(self.num):
                idx = n * self.anchor_len
                obj = x[idx+6, :, :].clone()
                ang = x[(idx+4):(idx+6), :, :].clone()
                box = x[idx:(idx+4), :, :].clone()
                classes = x[(idx+7):(idx+self.anchor_len), :, :].clone()
                #convert the box format to image coords
                box_im = box.permute(1,2,0).contiguous().view(in_height*in_width, 4)
                box_im[:,0] = box_im[:,0].add(i_list).div(in_width)
                box_im[:,1] = box_im[:,1].add(j_list).div(in_height)
                box_im[:,2] = torch.exp(box_im[:,2]).mul(self.anchors[n*2]/in_width)
                box_im[:,3] = torch.exp(box_im[:,3]).mul(self.anchors[n*2+1]/in_height)
                obj_im = obj.view(1,in_height*in_width)
                classes_im = classes.permute(1,2,0).contiguous().view(in_height*in_width, (self.anchor_len-7))
                prob_cls, idx_cls = classes_im.max(dim=1)
                prob_cls = prob_cls.view(1,in_height*in_width).mul(obj_im)
                box_list.append(box_im.view(in_height*in_width,4))
                obj_list.append(obj_im.view(in_height*in_width,1))
                cls_list.append(idx_cls.view(in_height*in_width,1))
                prob_cls_list.append(prob_cls.view(in_height*in_width,1))
                ang_list.append(ang.view(in_height*in_width,2))
            obj_total = torch.cat(obj_list, 0)
            box_total = torch.cat(box_list, 0)
            cls_total = torch.cat(cls_list, 0).float()
            ang_total = torch.cat(ang_list, 0)
            prob_cls_total = torch.cat(prob_cls_list, 0)
            #if the w/h of the image and the net is different, correct the region box
            #if (in_height*w_im) != (in_width*h_im):
                #box_total = correct_region_box(box_total, in_width*32, in_height*32, w_im, h_im)
            detection = torch.cat((obj_total,box_total,ang_total,cls_total,prob_cls_total), 1)
            result = nms_obj(detection, self.obj_thresh, self.nms_thresh)
            if truth is None:
                result = result_prob_fliter(result, self.class_thresh)
                result_list.append(result)
            else:
                truth_im = truth[b,:,:]
                truth_num_obj = truth_im[:,0].sign().sum()
                truth_im = truth_im[0:truth_num_obj, :]
                truth_box = truth_im[:, 0:4]
                truth_cls = truth_im[:, 4].view(1)
                result_box = result[:, 1:5].clone()
                iou, idx = box_iou_eval(result_box, truth_box).max(dim=1)
                n_pred, _ = result.size()
                mask_iou =  iou.sub(self.obj_thresh).sign()
                cls_cmp_truth = torch.index_select(truth_cls, 0, idx)
                mask_cls = result_box[:, 5].eq(cls_cmp_truth)
                mask_truth = mask_iou.mul(mask_cls)
                mask_truth = torch.max(mask_truth, zeros)
                n_truth = int(mask_truth.sum())
                n_TP = int(mask_truth.sum())
                n_FP = n_pred - n_TP
                n_TN = truth_num_obj - n_TP
                AP = float(n_TP) / n_pred
                Recall = float(n_TP)/truth_num_obj
                self.TP += n_TP
                self.FP += n_FP
                self.TN += n_TN
                result = result[0:n_pred, :]
                result_list.append(result)
        if truth is not None:
            self.AP = self.TP / (self.TP + self.FP)
            self.Recall = self.TP / (self.TP + self.TN)
            print('AP: %f, Recall: %f'%(self.AP,self.Recall))
        if result is not None:
            result = torch.cat(result_list, 0)
        return result
    def set_object_thresh(self, thresh):
        self.obj_thresh = thresh

def correct_region_box(boxes, net_w, net_h, im_w, im_h):
    print(net_w, net_h, im_w, im_h)
    if((net_w/net_h) < (im_w/im_h)):
        new_w = net_w
        new_h = net_w * im_h / im_w
    else:
        new_h = net_h
        new_w = im_w * net_h / im_h
    delta_x = (net_w - new_w)/2/net_w
    delta_y = (net_h - new_h)/2/net_h
    co_w = float(net_w)/new_w
    co_h = float(net_h)/new_h
    print(delta_x, delta_y, co_w, co_h)
    boxes[:, 0] = boxes[:, 0].sub(delta_x).mul(co_w)
    boxes[:, 1] = boxes[:, 1].sub(delta_y).mul(co_h)
    boxes[:, 2] = boxes[:, 2].mul(co_w)
    boxes[:, 3] = boxes[:, 3].mul(co_h)
    return boxes

def nms_obj(pred, obj_thresh, nms_thresh):
    zeros = torch.zeros(1)
    if pred.is_cuda:
        zeros = zeros.cuda()
    num, l = pred.size()
    p_obj = pred[:, 0]
    p_obj, idx = p_obj.sort(descending=True)
    pred_o = pred.index_select(0, idx).clone()
    print(pred_o)
    mask = p_obj.sub(obj_thresh).sign()
    mask = torch.max(mask, zeros)
    num_obj = int(mask.sum())
    if num_obj == 0:
        return None
    pred_o = pred_o[0:num_obj, :]
    out_list = []
    for i in range(num_obj-1):
        if pred_o[i,0] > 0.01:
            out_list.append(pred_o[i,:].view(1,l))
            pred_box = pred_o[i,1:5].view(1,4)
            truth_box = pred_o[(i+1):num_obj, 1:5].view((num_obj-i-1),4)
            iou = box_iou_eval(pred_box, truth_box).view(num_obj-i-1)
            mask = iou.sub(nms_thresh).sign() * (-1)
            mask = torch.max(mask, zeros)
            pred_o[(i+1):num_obj, 0] = pred_o[(i+1):num_obj, 0].view(num_obj-i-1).mul(mask)
        else:
            pass
    if out_list.__len__() > 0:
        out = torch.cat(out_list,0)
        return out
    else:
        return None

def result_prob_fliter(result, class_thresh):
    if result is None:
        return None
    zeros = torch.zeros(1)
    if result.is_cuda:
        zeros = zeros.cuda()
    mask_pred = result[:, 8].sub(class_thresh).sign()
    mask_pred = torch.max(mask_pred, zeros)
    result[:, 8] = torch.mul(result[:, 8],  mask_pred)
    outlist = []
    num, l = result.size()
    for i in range(num):
        if result[i, 8] > 0.01 and result[i,3]*result[i,4] > 0.0001:
            outlist.append(result[i, :].view(1,l))
        else:
            pass
    if outlist.__len__() > 0:
        out = torch.cat(outlist, 0)
        return out
    else:
        return None

#pred_box is (1,4) tensor and truth_box is (n,4) tensor
def box_iou_eval(pred_box, truth_box):
    num_pred, num_coords = pred_box.size()
    assert(num_coords == 4)
    num_truth, num_coords = truth_box.size()
    assert(num_coords == 4)
    w = Variable(torch.Tensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]))
    zero = Variable(torch.zeros(num_truth))
    iou = Variable(torch.zeros(num_pred))
    if pred_box.is_cuda:
        w = w.cuda()
        zero = zero.cuda()
        iou = iou.cuda()
    truth_box_ex = truth_box.mm(w)
    pred_box_ex = pred_box.mm(w)
    pred_box_ex2 = pred_box_ex.view(1,4*num_pred).expand(num_truth,4*num_pred)
    pred_box_ex3 = pred_box_ex2.view(num_truth,num_pred,4).permute(1,0,2)
    truth_box_ex3 = truth_box_ex.expand(num_pred, num_truth, 4)
    left = torch.max(pred_box_ex3[:,:,0], truth_box_ex3[:,:,0])
    right = torch.min(pred_box_ex3[:,:,2], truth_box_ex3[:,:,2])
    up = torch.max(pred_box_ex3[:,:,1], truth_box_ex3[:,:,1])
    down = torch.min(pred_box_ex3[:,:,3], truth_box_ex3[:,:,3])
    intersection_w = torch.max( right.sub(left), zero)
    intersection_h = torch.max( down.sub(up), zero)
    intersection = intersection_w.mul(intersection_h)
    w_truth = truth_box[:,2].view(1,num_truth).expand(num_pred, num_truth)
    h_truth = truth_box[:,3].view(1,num_truth).expand(num_pred, num_truth)
    w_pred = pred_box[:,2].view(num_pred,1).expand(num_pred, num_truth)
    h_pred = pred_box[:,3].view(num_pred,1).expand(num_pred, num_truth)
    union = torch.add(w_truth.mul(h_truth), w_pred.mul(h_pred)).sub(intersection)
    iou= intersection.div(union)
    return iou