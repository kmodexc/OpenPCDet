import torch
import torch.nn as nn
from ..ops.iou3d_nms import iou3d_nms_utils


def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
    if 'gt_boxes' not in data_dict:
        return recall_dict

    rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
    gt_boxes = data_dict['gt_boxes'][batch_index]

    if recall_dict.__len__() == 0:
        recall_dict = {'gt': 0}
        for cur_thresh in thresh_list:
            recall_dict['roi_%s' % (str(cur_thresh))] = 0
            recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

    cur_gt = gt_boxes
    k = cur_gt.__len__() - 1
    while k >= 0 and cur_gt[k].sum() == 0:
        k -= 1
    cur_gt = cur_gt[:k + 1]

    if cur_gt.shape[0] > 0:
        if box_preds.shape[0] > 0:
            iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
        else:
            iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

        if rois is not None:
            iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

        for cur_thresh in thresh_list:
            if iou3d_rcnn.shape[0] == 0:
                recall_dict['rcnn_%s' % str(cur_thresh)] += 0
            else:
                rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
            if rois is not None:
                roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

        recall_dict['gt'] += cur_gt.shape[0]
    else:
        gt_iou = box_preds.new_zeros(box_preds.shape[0])
    return recall_dict


def generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list, threshold=0.5):
    dece_data = []
    for index in range(len(pd_boxes_list)):
        pd_boxes = pd_boxes_list[index]
        pd_scores = pd_scores_list[index]
        gt_boxes = gt_boxes_list[index]

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k >= 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if pd_boxes.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(pd_boxes[:, 0:7], cur_gt[:, 0:7]).detach()
                pd_class = pd_boxes[:,-1].int().detach()
                gt_class = cur_gt[:,-1].int().detach()
                gt_mask = pd_class.unsqueeze(1) & gt_class.unsqueeze(0)
                dets     = iou3d_rcnn.detach() > threshold
                tps      = (dets & gt_mask).sum(1)
                fps      = (dets & torch.logical_not(gt_mask)).sum(1)
                dece_data.append((tps,fps,pd_scores))
    return dece_data


def merge_dece_records(last_data, current_data):
    dece_data = []
    for tps,fps,pd_scores in last_data:
        dece_data.append((tps.detach(),fps.detach(),pd_scores.detach()))
    dece_data += current_data
    return dece_data

def calc_dece(dece_data, bins=15):
    if dece_data is None or len(dece_data) <= 0:
        return 0, None
    dev = dece_data[0][0].device
    tps = torch.zeros(bins).to(device=dev)
    fps = torch.zeros(bins).to(device=dev)
    avg_scores = torch.zeros(bins).to(device=dev)
    for i in range(len(dece_data)):
        cur_data = dece_data[i]
        _tps,_fps,pd_scores = cur_data
        bins_ind = (pd_scores.detach() * bins).clamp(0,bins-1).int()
        for i in range(bins):
            filter_bin = bins_ind == i
            tps[i] += (torch.logical_and(filter_bin, _tps)).sum()
            fps[i] += (torch.logical_and(filter_bin, _fps)).sum()
            all_active_mask = tps[i]+fps[i]
            if all_active_mask.sum() > 0:
                avg_scores[i] += pd_scores[all_active_mask.nonzero()].mean()
    bin_size = tps+fps
    total_size = bin_size.sum()
    if total_size == 0:
        return 0, None
    mask = (bin_size != 0)
    prec = torch.zeros_like(avg_scores)
    prec[mask] = tps[mask].float() / bin_size[mask].float()
    bin_weights = bin_size[mask].float() / total_size.float()
    dece = bin_weights * (avg_scores[mask] - prec[mask])
    dece_item = torch.abs(dece).sum()
    print("prec",prec)
    print("bw",bin_weights)
    print("avg",avg_scores)
    print("dece",dece)
    return dece_item, dece


class DECELoss(nn.Module):
    def __init__(self):
        super(DECELoss, self).__init__()
        self.last_dece = []
    
    def forward(self, pd_boxes_list, pd_scores_list, gt_boxes_list):

        print(pd_scores_list)

        dece_data = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list)

        merged_dece_data = merge_dece_records(self.last_dece,dece_data)

        dece_loss, _ = calc_dece(merged_dece_data)

        self.last_dece = dece_data

        return dece_loss


def adaptive_focal_loss(gamma, dece_raw, pd_scores_list):
    bins = gamma.shape[0]
    loss = 0
    PAR_GAMMA  = 1
    GAMMA_MAX  = 20
    GAMMA_MIN  = -2
    GAMMA_SW   = 0.2

    gamma = gamma.detach()
    dece_raw = dece_raw.detach()

    neg_gamma  = gamma.lt(0)
    pos_gamma  = torch.logical_not(neg_gamma)

    new_gamma  = torch.clamp(gamma * torch.exp( - PAR_GAMMA * dece_raw), min=GAMMA_MIN, max=GAMMA_MAX) * neg_gamma.float()
    new_gamma += torch.clamp(gamma * torch.exp(   PAR_GAMMA * dece_raw), min=GAMMA_MIN, max=GAMMA_MAX) * pos_gamma.float()

    below_thr  = torch.abs(gamma).lt(GAMMA_SW)

    gamma = new_gamma

    gamma[below_thr & neg_gamma] =   GAMMA_SW
    gamma[below_thr & pos_gamma] = - GAMMA_SW

    for pd_score in pd_scores_list:
        
        bins_ind = (pd_score.detach() * bins).clamp(0,bins-1).int()
        
        gammas = gamma[bins_ind]
        
        neg_gammas = gammas.lt(0)
        pos_gammas = torch.logical_not(neg_gammas)

        loss -= (torch.pow(1+pd_score,torch.abs(gammas)) * torch.log(pd_score) * neg_gammas.float()).sum()
        loss -= (torch.pow(1-pd_score,          gammas)  * torch.log(pd_score) * pos_gammas.float()).sum()

    return loss, gamma


class AdaptiveFocalLoss(nn.Module):

    def __init__(self):
        super(AdaptiveFocalLoss, self).__init__()
        self.last_dece = []
        self.gamma = None
        self.number_of_bins = 15

    def forward(self, pd_boxes_list, pd_scores_list, gt_boxes_list):

        if self.gamma is None:
            if len(pd_scores_list) == 0:
                return 0
            else:
                self.gamma = torch.ones(self.number_of_bins).to(device=pd_scores_list[0].device)

        dece_data = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list)

        merged_dece_data = merge_dece_records(self.last_dece, dece_data)

        self.last_dece = dece_data

        _, dece_raw = calc_dece(merged_dece_data, self.number_of_bins)

        loss, new_gamma = adaptive_focal_loss(self.gamma, dece_raw, pd_scores_list)

        self.gamma = new_gamma

        return loss

