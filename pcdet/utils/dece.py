import torch
import torch.nn as nn
from ..ops.iou3d_nms import iou3d_nms_utils




def calc_iou_testing(boxesa,boxesb):
    dist = torch.zeros((boxesa.shape[0], boxesb.shape[0]))
    for i in range(boxesa.shape[0]):
        for j in range(boxesb.shape[0]):
            assert len(list(boxesa.shape)) == 2, f"shape is {boxesa.shape}"
            assert len(list(boxesb.shape)) == 2, f"shape is {boxesb.shape}"
            assert boxesa.shape[1] == 7, f"shape is {boxesa.shape}"
            assert boxesb.shape[1] == 7, f"shape is {boxesb.shape}"
            dist[i,j] = ((boxesa[i]*boxesa[i]) + (boxesb[j]*boxesb[j]))[0:3].sum().sqrt()
    ious = torch.clamp(1-dist,min=0)
    print("ious",ious)
    return ious


def generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list, threshold=0.5, testing=False, full_scores=False):
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
                if not testing:
                    iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(pd_boxes[:, 0:7], cur_gt[:, 0:7]).detach()
                else:
                    iou3d_rcnn = calc_iou_testing(pd_boxes[:, 0:7], cur_gt[:, 0:7])
                gt_class = cur_gt[:,-1].int().detach()
                dets     = iou3d_rcnn > threshold
                if full_scores:
                    tps = torch.zeros((dets.shape[0],dets.shape[1],pd_scores.shape[-1]),dtype=torch.bool,device=dets.device)
                    fps = torch.zeros((dets.shape[0],dets.shape[1],pd_scores.shape[-1]),dtype=torch.bool,device=dets.device)
                    dets_ind = torch.where(dets)
                    if len(dets_ind) != 2 or dets_ind[0].shape[0] == 0:
                        continue
                    assert len(dets_ind) == 2
                    assert dets_ind[0].shape[0] != 0
                    # assert ((dets_ind[0] >= 0) & (dets_ind[0] < tps.shape[0])).all(), f"dets={dets} \n\n det_inds={dets_ind} \n\n dets.shape={dets.shape} \n\n torch.nonzero={torch.nonzero(dets)} \n\n torch.where={torch.where(dets)}"
                    # assert ((dets_ind[1] >= 0) & (dets_ind[1] < tps.shape[1])).all(), f"dets={dets} \n\n det_inds={dets_ind} \n\n dets.shape={dets.shape} \n\n torch.nonzero={torch.nonzero(dets)} \n\n torch.where={torch.where(dets)}"
                    tps_ind = gt_class[dets_ind[1]] - 1
                    # assert ((tps_ind >= 0) & (tps_ind < tps.shape[2])).all(), f"tps_ind={tps_ind}"
                    fps[dets_ind[0],dets_ind[1],:] = True
                    tps[dets_ind[0],dets_ind[1],tps_ind] = True
                    fps[dets_ind[0],dets_ind[1],tps_ind] = False
                    tps = tps.sum(1)
                    fps = fps.sum(1)
                else:
                    pd_class = pd_boxes[:,-1].int().detach()
                    gt_mask  = pd_class.unsqueeze(1) & gt_class.unsqueeze(0)
                    tps      = (dets & gt_mask).sum(1)
                    fps      = torch.logical_not(tps)
                dece_data.append((tps,fps,pd_scores))
    return dece_data


def merge_dece_records(last_data, current_data):
    dece_data = []
    for l in last_data:
        for tps,fps,pd_scores in l:
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
        assert list(pd_scores.shape) == list(_tps.shape), f"pdscores shape={pd_scores.shape} tps shape={_tps.shape}"
        for i in range(bins):
            filter_bin = bins_ind == i
            filter_tps = torch.logical_and(filter_bin, _tps)
            filter_fps = torch.logical_and(filter_bin, _fps)
            tps[i] += filter_tps.sum()
            fps[i] += filter_fps.sum()
            all_active_mask = torch.logical_or(filter_tps,filter_fps)
            if all_active_mask.sum() > 0:
                avg_scores[i] += pd_scores[all_active_mask].sum()
    bin_size = tps+fps
    total_size = bin_size.sum()
    if total_size == 0:
        return 0, None
    mask = (bin_size != 0)
    avg_scores[mask] /= bin_size[mask]
    prec = torch.zeros_like(avg_scores)
    prec[mask] = tps[mask].float() / bin_size[mask].float()
    bin_weights = bin_size[mask].float() / total_size.float()
    dece = torch.zeros_like(avg_scores)
    dece[mask] = avg_scores[mask] - prec[mask]
    dece_item = torch.abs(dece)
    dece_item[mask] *= bin_weights
    dece_item = dece_item.sum()
    return dece_item, dece


class DECELoss(nn.Module):
    def __init__(self,num_last_dece=1):
        super(DECELoss, self).__init__()
        self.last_dece = []
        self.last_pointer = 0
        self.num_last_dece = num_last_dece
        self.use_full_scores = True
        for _ in range(num_last_dece):
            self.last_dece.append([])
    
    def forward(self, pd_boxes_list, pd_scores_list, gt_boxes_list):

        dece_data = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list, full_scores=self.use_full_scores)

        merged_dece_data = merge_dece_records(self.last_dece,dece_data)

        dece_loss, _ = calc_dece(merged_dece_data)

        self.last_dece[self.last_pointer] = dece_data
        self.last_pointer = (self.last_pointer + 1) % self.num_last_dece

        return dece_loss


class FullDECELoss(nn.Module):
    def __init__(self,num_last_dece=1):
        super(FullDECELoss, self).__init__()
        self.last_dece = []
        self.last_pointer = 0
        self.num_last_dece = num_last_dece
        self.use_full_scores = True
        for _ in range(num_last_dece):
            self.last_dece.append([])

    def forward(self, pd_boxes_list, pd_scores_list, gt_boxes_list):

        dece_data = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list, full_scores=self.use_full_scores)

        merged_dece_data = merge_dece_records(self.last_dece,dece_data)

        dece_loss, _ = calc_dece(merged_dece_data)

        self.last_dece[self.last_pointer] = dece_data
        self.last_pointer = (self.last_pointer + 1) % self.num_last_dece

        return dece_loss


def adaptive_focal_loss(gamma, dece_raw, pd_scores_list):
    if dece_raw is None:
        return 0, gamma

    bins = gamma.shape[0]
    loss = 0
    LAMBDA  = 1
    GAMMA_MAX  = 20
    GAMMA_MIN  = -2
    GAMMA_SW   = 0.2

    gamma = gamma.detach()
    dece_raw = dece_raw.detach()

    neg_gamma  = gamma.lt(0)
    pos_gamma  = torch.logical_not(neg_gamma)

    new_gamma  = torch.clamp(gamma * torch.exp( - LAMBDA * dece_raw), min=GAMMA_MIN, max=GAMMA_MAX) * neg_gamma.float()
    new_gamma += torch.clamp(gamma * torch.exp(   LAMBDA * dece_raw), min=GAMMA_MIN, max=GAMMA_MAX) * pos_gamma.float()

    below_thr  = torch.abs(gamma).lt(GAMMA_SW)

    gamma = new_gamma

    gamma[below_thr & neg_gamma] =   GAMMA_SW
    gamma[below_thr & pos_gamma] = - GAMMA_SW

    for pd_score in pd_scores_list:
        pd_score_nonzero = pd_score[pd_score > 0]
        
        bins_ind = (pd_score_nonzero.detach() * bins).clamp(0,bins-1).int()
        
        gammas = gamma[bins_ind]
        
        neg_gammas = gammas.lt(0)
        pos_gammas = torch.logical_not(neg_gammas)

        loss -= (torch.pow(1+pd_score_nonzero,torch.abs(gammas)) * torch.log(pd_score_nonzero) * neg_gammas.float()).sum()
        loss -= (torch.pow(1-pd_score_nonzero,          gammas)  * torch.log(pd_score_nonzero) * pos_gammas.float()).sum()

    return loss, gamma


class AdaptiveFocalLoss(nn.Module):

    def __init__(self,num_last_dece=1):
        super(AdaptiveFocalLoss, self).__init__()
        self.last_dece = []
        self.last_pointer = 0
        self.num_last_dece = num_last_dece
        self.use_full_scores = False
        for _ in range(num_last_dece):
            self.last_dece.append([])
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

        _, dece_raw = calc_dece(merged_dece_data, self.number_of_bins)

        loss, new_gamma = adaptive_focal_loss(self.gamma, dece_raw, pd_scores_list)

        self.last_dece[self.last_pointer] = dece_data
        self.last_pointer = (self.last_pointer + 1) % self.num_last_dece

        self.gamma = new_gamma

        return loss



def test_generate_dece_record_1():
    box_a = torch.tensor([0,0,0,1,1,1,0,0])
    box_b = torch.tensor([0,0,0,1,1,1,0,0])
    pd_boxes_list = [box_a.view(1,8)]
    pd_scores_list = [torch.tensor([1.0]).view(1)]
    pd_boxes_list.append(pd_boxes_list[0])
    pd_scores_list.append(pd_scores_list[0])
    gt_boxes_list = [box_b.view(1,8)]
    gt_boxes_list.append(gt_boxes_list[0])
    res = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list,testing=True)
    assert res is not None
    assert len(res) != 0
    tps,fps,score = res[0]
    assert list(  tps.shape) == [1]
    assert list(  fps.shape) == [1]
    assert list(score.shape) == [1]

def test_generate_dece_record_2():
    box_a = torch.tensor([0,0,0,1,1,1,0,0])
    box_b = torch.tensor([2,2,0,1,1,1,0,0])
    box_c = torch.tensor([0.1,0,0,1,1,1,0,1])
    pd_boxes_list = [torch.stack((box_a,box_b))]
    pd_scores_list = [torch.tensor([1.0,0.5]).view(2)]
    gt_boxes_list = [torch.stack((box_a,box_b,box_c))]
    res = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list,testing=True)
    assert res is not None
    assert len(res) != 0
    tps,fps,score = res[0]
    assert list(  tps.shape) == [2]
    assert list(  fps.shape) == [2]
    assert list(score.shape) == [2]

def test_generate_dece_record_full_score_1():
    box_a = torch.tensor([0,0,0,1,1,1,0,1]).view(1,8)
    box_b = torch.tensor([0,0,0,1,1,1,0,2]).view(1,8)
    scores = torch.tensor([1.0,0.8,0.7]).view(1,3)
    pd_boxes_list = [box_a]
    pd_scores_list = [scores]
    gt_boxes_list = [torch.cat((box_a,box_b))]
    print("pd_boxes_list",pd_boxes_list)
    print("pd_scores_list",pd_scores_list)
    print("gt_boxes_list",gt_boxes_list)
    res = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list,testing=True,full_scores=True)
    assert res is not None
    assert len(res) != 0
    tps,fps,score = res[0]
    assert list(  tps.shape) == [1, 3]
    assert list(  fps.shape) == [1, 3]
    assert list(score.shape) == [1, 3]

def test_generate_dece_record_full_score_2():
    box_a = torch.tensor([0,0,0,1,1,1,0,1]).view(1,8)
    box_b = torch.tensor([0,0,3,1,1,1,0,2]).view(1,8)
    box_c = torch.tensor([2,2,0,1,1,1,0,2]).view(1,8)
    box_d = torch.tensor([2,0,0,1,1,1,0,2]).view(1,8)
    box_e = torch.tensor([0,0,0,1,1,1,0,2]).view(1,8)
    box_f = torch.tensor([2,2,0,1,1,1,0,2]).view(1,8)
    box_g = torch.tensor([5,2,0,1,1,1,0,2]).view(1,8)
    scores = torch.tensor([[1.0,0.8,0.7]]).view(1,3)
    pd_boxes_list = [torch.cat((box_a,box_c,box_e,box_f,box_g))]
    pd_scores_list = [scores]
    gt_boxes_list = [torch.cat((box_a,box_b,box_c,box_d))]
    print("pd_boxes_list",pd_boxes_list)
    print("pd_scores_list",pd_scores_list)
    print("gt_boxes_list",gt_boxes_list)
    res = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list,testing=True,full_scores=True)
    assert res is not None
    assert len(res) != 0
    tps,fps,score = res[0]
    assert list(  tps.shape) == [pd_boxes_list[0].shape[0], 3]
    assert list(  fps.shape) == [pd_boxes_list[0].shape[0], 3]
    assert list(score.shape) == [1, 3]
    assert (scores == score).all()
    assert (tps[0] == torch.tensor([1,0,0])).all(), f"tps is {tps}"
    assert (tps[1] == torch.tensor([0,0,0])).all(), f"tps is {tps}"

def test_generate_dece_record_full_score_3():
    box_a = torch.tensor([0,0,0,1,1,1,0,1]).view(1,8)
    box_b = torch.tensor([0,0,0,1,1,1,0,2]).view(1,8)
    box_c = torch.tensor([2,2,0,1,1,1,0,1]).view(1,8)
    scores = torch.tensor([[1.0,0.8,0.7],[0.8,0.3,0.5]]).view(2,3)
    pd_boxes_list = [box_c.view(1,8)]
    pd_scores_list = [scores]
    gt_boxes_list = [torch.cat((box_a,box_b))]
    print("pd_boxes_list",pd_boxes_list)
    print("pd_scores_list",pd_scores_list)
    print("gt_boxes_list",gt_boxes_list)
    res = generate_dece_record(pd_boxes_list, pd_scores_list, gt_boxes_list,testing=True,full_scores=True)
    assert res is not None
    assert len(res) == 0

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
    assert list(raw.shape) == [3], f"raw.shape={raw.shape}"

def test_calc_dece_val_filtered():
    tps = torch.tensor([0,1,0,0])
    fps = torch.tensor([1,0,1,0])
    scores = torch.tensor([0.1,0.5,0.9,0])
    data = [(tps,fps,scores)]
    dece,raw = calc_dece(data,3)
    assert dece > 0
    assert dece < 1
    assert (raw > -1).all(), f"raw={raw}"
    assert (raw < 1).all(), f"raw={raw}"
    assert raw.nonzero().shape[0] == 3, f"raw={raw}"
    assert list(raw.shape) == [3], f"raw.shape={raw.shape}"

def test_calc_dece_empty_bins():
    tps = torch.tensor([0,1,0,0])
    fps = torch.tensor([1,0,0,0])
    scores = torch.tensor([0.1,0.5,0,0])
    data = [(tps,fps,scores)]
    n_bins = 10
    dece,raw = calc_dece(data,n_bins)
    assert dece > 0
    assert dece < 1
    assert (raw > -1).all(), f"raw={raw}"
    assert (raw < 1).all(), f"raw={raw}"
    assert raw.nonzero().shape[0] == 2, f"raw={raw}"
    assert list(raw.shape) == [n_bins], f"raw.shape={raw.shape}"

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

def test_calc_dece_val_2():
    tps = torch.tensor([1,0,1])
    fps = torch.tensor([0,1,0])
    scores = torch.tensor([0.8,0.85,0.9])
    data = [(tps,fps,scores)]
    dece,raw = calc_dece(data,3)
    assert 0.25 > dece > 0.18, f"raw={raw}"
    assert raw is not None, f"raw={raw}"
    assert (raw[0:2] == 0).all(), f"raw={raw[0:2]}"
    assert 0.25 > raw[2] > 0.18, f"raw={raw[0:2]}"

def test_calc_dece_val_3():
    tps = torch.tensor([1,0,1,0,1,0,1])
    fps = 1-tps
    scores = torch.tensor([0.9,0.9,0.9,0.1,0.1,0.1,0.1])
    data = [(tps,fps,scores)]
    dece,raw = calc_dece(data,3)
    assert raw is not None, f"raw={raw}"
    assert 0.41 > -raw[0] > 0.39, f"raw={raw}"
    assert raw[1] == 0, f"raw={raw}"
    assert 0.25 > raw[2] > 0.21, f"raw={raw}"
    assert 0.33 > dece > 0.32, f"raw={raw}"

def test_calc_dece_full():
    tps = torch.tensor([[1,0],[0,1],[1,0]])
    fps = 1-tps
    scores = torch.tensor([[0.99,0.01],[0.01,0.99],[0.99,0.01]])
    print("scores",scores)
    data = [(tps,fps,scores)]
    dece, raw = calc_dece(data, 3)
    assert raw is not None, f"raw={raw}"
    assert raw[0] == 0.01, f"raw={raw}"
    assert raw[1] == 0.0, f"raw={raw}"
    assert -0.009 > raw[2] > -0.011, f"raw={raw}"
    assert 0.011 > dece > 0.009, f"raw={raw}"

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
    assert new_gamma.isnan().sum() == 0

def test_adafocal_zero_scores():
    bins = 15
    gammas = torch.ones(bins)
    dece = torch.arange(bins).float()/bins
    scores = [torch.rand(100)]
    scores[0][80:90] = 0
    loss, new_gamma = adaptive_focal_loss(gammas, dece, scores)
    assert loss > 0
    assert new_gamma.isnan().sum() == 0

def test_adafocal_empty():
    bins = 15
    gammas = torch.ones(bins)
    dece = torch.arange(bins).float()/bins
    scores = []
    loss, new_gamma = adaptive_focal_loss(gammas, dece, scores)
    assert loss == 0
    assert new_gamma.isnan().sum() == 0

def test_adafocal_val_neg():
    bins = 15
    gammas = torch.ones(bins)
    dece = (torch.arange(bins).float()/bins)-0.5
    scores = [torch.rand(100)]
    loss, new_gamma = adaptive_focal_loss(gammas, dece, scores)
    assert loss > 0
    assert new_gamma.isnan().sum() == 0

if __name__ == "__main__":
    print("use pytest!")

