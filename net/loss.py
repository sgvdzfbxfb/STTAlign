import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class GeometricReconstructionLoss(nn.Module):
    def __init__(self):
        super(GeometricReconstructionLoss, self).__init__()

    def forward(self, X_v, target_X_v, weights, device: torch.device):
        loss = torch.zeros([X_v.shape[0], X_v.shape[1]]).to(device)
        for bn in range(X_v.shape[0]):
            for idx in range(X_v.shape[1]):
                pred = X_v[bn, idx, :, :]
                tag = target_X_v[bn, idx, :, :]
                pred_ = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
                tag_ = tag.unsqueeze(0).repeat(pred.shape[0], 1,  1)
                diff = torch.sum(torch.pow(torch.sub(pred_, tag_), 2), dim=-1)
                minv = torch.argmin(diff, dim=1)
                minvy = torch.argmin(diff, dim=0)
                tagp = tag[minv]
                predd = pred[minvy]
                tmp1 = F.smooth_l1_loss(pred, tagp, reduction="mean")
                tmp2 = F.smooth_l1_loss(tag, predd, reduction="mean")
                loss[bn, idx] = tmp1 + tmp2
        loss = torch.sum(loss * weights)
        prec = torch.mean(X_v, dim=2)
        tarc = torch.mean(target_X_v, dim=2)
        lossc = F.smooth_l1_loss(prec, tarc, reduction="sum") / (prec.shape[0]*3)
        losstc = F.smooth_l1_loss(torch.mean(prec, dim=1), torch.mean(tarc, dim=1), reduction="mean")
        loss = loss / target_X_v.shape[0]
        return loss, lossc

def symmetric_loss(X_v):
    nums = X_v.shape[1]//2
    rg = X_v[:, 0:nums, :, :]
    lg = X_v[:, nums:, :, :]
    lg = torch.flip(lg, dims=[1])
    rgc = torch.abs(torch.mean(rg, dim=2))
    lgc =  torch.abs(torch.mean(lg, dim=2))
    lossc = F.smooth_l1_loss(rgc[:, :, 0:2], lgc[:, :, 0:2], reduction="sum") / (rgc.shape[0] * 2)
    return lossc

def nearnest_index(pred_, tag_):
    pred = pred_ -torch.mean(pred_, dim=0)
    tag = tag_ -torch.mean(pred_, dim=0)
    pred = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
    tag = tag.unsqueeze(0).repeat(pred.shape[0], 1, 1)
    diff = torch.sqrt(torch.sum(torch.pow(torch.sub(pred, tag), 2), dim=-1))
    min_index = torch.argmin(diff, dim=1)
    minv = torch.min(diff, dim=1)[0]
    return min_index, minv

def nearnest_value(pred_, tag_):
    pred = pred_
    tag = tag_
    pred = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
    tag = tag.unsqueeze(0).repeat(pred.shape[0], 1, 1)
    diff = torch.sqrt(torch.sum(torch.pow(torch.sub(pred, tag), 2), dim=-1))
    min_index = torch.argmin(diff, dim=1)
    nearnestp = tag_[min_index]
    minv = pred_ - nearnestp
    return min_index, minv

def spatial_Relation_Loss(pred, target, weights, device):
    loss = torch.zeros([pred.shape[0], pred.shape[1]]).to(device)
    for bn in range(pred.shape[0]):
        for idx in range(pred.shape[1] -1):
            pred1 = pred[bn, idx, :, :]
            pred2 = pred[bn, idx+1, :, :]
            tag1 = target[bn, idx, :, :]
            tag2 = target[bn, idx+1, :, :]
            min_index1, _ = nearnest_index(pred1, tag1)
            min_index2, _ = nearnest_index(pred2, tag2)
            tag1_ = tag1[min_index1]
            tag2_ = tag2[min_index2]
            min_indexpp, minvp1 = nearnest_value(pred1, pred2)
            min_indextp, minvpt1= nearnest_value(tag1_, tag2)
            min_indexpp, minvp2 = nearnest_value(pred2, pred1)
            min_indextp, minvpt2= nearnest_value(tag2_, tag1)
            minvp_mask1 = minvp1
            minvpt_mask1 = minvpt1
            minvp_mask2 = minvp2
            minvpt_mask2 = minvpt2
            lossc1 = F.smooth_l1_loss(minvp_mask1, minvpt_mask1, reduction="mean")
            lossc2 = F.smooth_l1_loss(minvp_mask2, minvpt_mask2, reduction="mean")
            loss[bn, idx] = (lossc1 + lossc2)*0.5
    loss = torch.sum(loss) / weights.shape[0]
    return  loss

def interdental_occlusion_loss(assm_a, lab_a, assm_b, lab_b, device: torch.device):
    cox_coincide_loss_save = torch.zeros(size=[lab_a.shape[0], lab_a.shape[1]], device=device)
    sigma_groove_loss_save = torch.zeros(size=[lab_a.shape[0], lab_a.shape[1]], device=device)
    sigma_incisors_loss_save = torch.zeros(size=[lab_a.shape[0], lab_a.shape[1]], device=device)
    for bn in range(lab_a.shape[0]):
        for idx in range(lab_a.shape[1]):
            pot_a = lab_a[bn][idx]
            pot_a_z = pot_a.clone()[:, 2:]
            pot_a = pot_a[:, :2]
            tt = lab_a.shape[1] - 1
            pot_b = lab_b[bn][tt - idx]
            pot_b_alone = pot_b
            pot_b_z_alone = pot_b_alone.clone()[:, 2:]
            if idx == 0:
                pot_b = torch.cat([pot_b, lab_b[bn][tt - (idx + 1)]], dim=0)
            elif idx == lab_a.shape[1] - 1:
                pot_b = torch.cat([lab_b[bn][tt - (idx - 1)], pot_b], dim=0)
            else:
                pot_b = torch.cat([lab_b[bn][tt - (idx - 1)], pot_b], dim=0)
                pot_b = torch.cat([pot_b, lab_b[bn][tt - (idx + 1)]], dim=0)
            pot_b = pot_b[:, :2]
            pot_a_ = pot_a.unsqueeze(1).repeat(1, pot_b.shape[0], 1)
            pot_b_ = pot_b.unsqueeze(0).repeat(pot_a.shape[0], 1, 1)
            diff = torch.sum(torch.pow(torch.sub(pot_a_, pot_b_), 2), dim=-1)
            minvx = torch.argmin(diff, dim=1)
            is_have_cos_label = []
            for id in range(minvx.shape[0]):
                if torch.norm(pot_a[id] - pot_b[minvx[id]]) < 0.7:
                    is_have_cos_label.append(True)
                else:
                    is_have_cos_label.append(False)
            as_pot_a = assm_a[bn][idx]
            as_pot_a_z = as_pot_a.clone()[:, 2:]
            as_pot_a = as_pot_a[:, :2]
            tt = assm_a.shape[1] - 1
            as_pot_b = assm_b[bn][tt - idx]
            as_pot_b_alone = as_pot_b
            as_pot_b_z_alone = as_pot_b_alone.clone()[:, 2:]
            if idx == 0:
                as_pot_b = torch.cat([as_pot_b, assm_b[bn][tt - (idx + 1)]], dim=0)
            elif idx == assm_a.shape[1] - 1:
                as_pot_b = torch.cat([assm_b[bn][tt - (idx - 1)], as_pot_b], dim=0)
            else:
                as_pot_b = torch.cat([assm_b[bn][tt - (idx - 1)], as_pot_b], dim=0)
                as_pot_b = torch.cat([as_pot_b, assm_b[bn][tt - (idx + 1)]], dim=0)
            as_pot_b_z = as_pot_b.clone()[:, 2:]
            as_pot_b = as_pot_b[:, :2]
            as_pot_a_ = as_pot_a.unsqueeze(1).repeat(1, as_pot_b.shape[0], 1)
            as_pot_b_ = as_pot_b.unsqueeze(0).repeat(as_pot_a.shape[0], 1, 1)
            as_diff = torch.sum(torch.pow(torch.sub(as_pot_a_, as_pot_b_), 2), dim=-1)
            as_minvx = torch.argmin(as_diff, dim=1)
            is_have_cos_assem = []
            not_coincide_num = 0
            zvalue_dis = []
            for id in range(as_minvx.shape[0]):
                flgg = False
                if torch.norm(as_pot_a[id] - as_pot_b[as_minvx[id]]) < 0.7:
                    flgg = True
                is_have_cos_assem.append(flgg)
                if flgg != is_have_cos_label[id]:
                    not_coincide_num += 1
                if is_have_cos_label[id]:
                    zvalue_dis.append(as_pot_a_z[id] - as_pot_b_z[as_minvx[id]])
            cox_coinc_lo = not_coincide_num
            cox_coincide_loss_save[bn][idx] = cox_coinc_lo
            if idx >= 5 and idx <= 10:
                canine_p1_ass = torch.mean(as_pot_a)
                a_xft2 = torch.argmin(as_pot_a_z, dim=0)
                canine_p2_ass = as_pot_a[a_xft2]
                canine_p3_ass = torch.mean(as_pot_b_alone)
                a_xft4 = torch.argmax(as_pot_b_z_alone, dim=0)
                canine_p4_ass = as_pot_b_alone[a_xft4]
                canine_p1_lab = torch.mean(pot_a)
                xft2 = torch.argmin(pot_a_z, dim=0)
                canine_p2_lab = pot_a[xft2]
                canine_p3_lab = torch.mean(pot_b_alone)
                xft4 = torch.argmax(pot_b_z_alone, dim=0)
                canine_p4_lab = pot_b_alone[xft4]
                gap_1 = torch.norm(canine_p1_ass - canine_p1_lab)
                gap_2 = torch.norm(canine_p2_ass - canine_p2_lab)
                gap_3 = torch.norm(canine_p3_ass - canine_p3_lab)
                gap_4 = torch.norm(canine_p4_ass - canine_p4_lab)
                sigma_z_loss = gap_1 + gap_2 + gap_3 + gap_4
                sigma_incisors_loss_save[bn][idx] = sigma_z_loss
            else:
                if len(zvalue_dis) > 0:
                    zvalue_dis = torch.stack(zvalue_dis)
                    sigma_z_loss = torch.var(zvalue_dis, unbiased=False)
                    sigma_groove_loss_save[bn][idx] = sigma_z_loss
    cox_coincide_loss = torch.mean(torch.mean(cox_coincide_loss_save, dim=1))
    sigma_groove_loss = torch.mean(torch.mean(sigma_groove_loss_save, dim=1))
    sigma_incisors_loss = torch.mean(torch.mean(sigma_incisors_loss_save, dim=1))
    return cox_coincide_loss, sigma_groove_loss, sigma_incisors_loss
