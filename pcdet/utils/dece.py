import torch
import torch.nn as nn
from ..ops.iou3d_nms import iou3d_nms_utils



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
    for j in range(len(dece_data)):
        cur_data = dece_data[j]
        _tps,_fps,pd_scores = cur_data
        bins_ind = (pd_scores.detach() * bins).clamp(0,bins-1).int()
        for i in range(bins):
            filter_bin = bins_ind == i
            filter_tps = torch.logical_and(filter_bin, _tps)
            filter_fps = torch.logical_and(filter_bin, _fps)
            tps[i] += filter_tps.sum()
            fps[i] += filter_fps.sum()
            all_active_mask = torch.logical_or(filter_tps,filter_fps)
            if all_active_mask.sum() > 0:
                # if pd_scores.isnan().sum() > 0:
                #     print("scores",pd_scores)
                avg_scores[i] += pd_scores[all_active_mask.nonzero()].sum()
                # if avg_scores[i].isnan().sum() > 0:
                #     print("scores",pd_scores)
                #     print("mask",all_active_mask)
                #     print("selection",pd_scores[all_active_mask.nonzero()])
                #     print("avg_score nan at",i)
                #     print("avg_score",avg_scores)
    bin_size = tps+fps
    total_size = bin_size.sum()
    if total_size == 0:
        return 0, None
    mask = (bin_size != 0)
    avg_scores[mask] /= bin_size[mask]
    prec = torch.zeros_like(avg_scores)
    prec[mask] = tps[mask].float() / bin_size[mask].float()
    bin_weights = bin_size[mask].float() / total_size.float()
    dece = bin_weights * (avg_scores[mask] - prec[mask])
    dece_item = torch.abs(dece).sum()
    # if dece.isnan().sum() > 0:
    #     print("prec",prec)
    #     print("bw",bin_weights)
    #     print("avg",avg_scores)
    #     print("dece",dece)
    return dece_item, dece


class DECELoss(nn.Module):
    def __init__(self):
        super(DECELoss, self).__init__()
        self.last_dece = []
    
    def forward(self, pd_boxes_list, pd_scores_list, gt_boxes_list):

        dece_data = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list)

        merged_dece_data = merge_dece_records(self.last_dece,dece_data)

        dece_loss, _ = calc_dece(merged_dece_data)

        self.last_dece = dece_data

        return dece_loss


def adaptive_focal_loss(gamma, dece_raw, pd_scores_list):
    if dece_raw is None:
        return 0, gamma

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


def test_calc_dece_val():
    tps = torch.tensor([0,1,0])
    fps = torch.tensor([1,0,1])
    scores = torch.tensor([0.1,0.5,0.9])
    data = [(tps,fps,scores)]
    dece,raw = calc_dece(data,3)
    assert dece > 0
    assert dece < 1
    assert (raw > -1).all(), f"raw={raw}"
    assert (raw < 1).all(), f"raw={raw}"
    assert raw.nonzero().shape[0] == 3, f"raw={raw}"


def test_calc_dece_none():
    tps = torch.tensor([0,0,0])
    fps = torch.tensor([0,0,0])
    scores = torch.tensor([0.1,0.5,0.9])
    data = [(tps,fps,scores)]
    dece,raw = calc_dece(data,3)
    assert dece == 0
    assert raw is None, f"raw={raw}"
    dece,raw = calc_dece(None,3)
    assert dece == 0
    assert raw is None, f"raw={raw}"

def test_adafocal_none():
    bins = 15
    gammas = torch.ones(bins)
    loss, new_gamma = adaptive_focal_loss(gammas, None, None)
    assert loss == 0
    assert (new_gamma == gammas).all()

def test_adafocal_val():
    bins = 15
    gammas = torch.ones(bins)
    dece = torch.arange(bins).float()/bins
    scores = [torch.rand(100)]
    loss, new_gamma = adaptive_focal_loss(gammas, dece, scores)
    assert loss > 0
    assert (new_gamma != gammas).all()

if __name__ == "__main__":
    test_calc_dece_val()
    test_calc_dece_none()
    test_adafocal_none()
    test_adafocal_val()

