from __future__ import print_function
import os
import sys
sys.path.append("/home/charon/codeGala/dzx/orth-tooth")
import config.config as cfg
import time
import argparse
import math
import torch
import numpy as np
from net.arangementnet import teeth_arangement_model
from util import IOStream, Tooth_Assembler
from data.utils import get_files,walkFile
from data.load_test_data import get_test_data, mapping_output
from net.loss import GeometricReconstructionLoss, interdental_occlusion_loss
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_axis_angle

def model_initial(model, model_name):
    pretrained_dict = torch.load(model_name)["model"]
    model_dict = model.state_dict()
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dictf)
    model.load_state_dict(model_dict)

    print("model initial over")


def walkFileType(path_root, file_list, type_):

    for root, dirs, files in os.walk(path_root):
        for d in dirs:
            path_file = os.path.join(root, d)
            if type_ in path_file:
                corrs_up_dir = path_file.replace("_down", "_up")
                if os.path.exists(corrs_up_dir):
                    file_list.append(path_file)


def test(iteration_con, device, model_u, model_l, where_read, where_write, is_the_last):
    tooth_assembler_u = Tooth_Assembler()
    tooth_assembler_l = Tooth_Assembler()
    reconl1_loss = GeometricReconstructionLoss()

    root_file_path = where_read
    if not os.path.exists(where_write):
        os.mkdir(where_write)
    dir_list_l = []
    walkFileType(root_file_path, dir_list_l, "down_end")


    save_root = "/devdata/dzx_data/tadpmData/singleMesh/res_stl"
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    dir_list_u = []
    for fl in dir_list_l:
        dir_list_u.append(fl.replace("_down", "_up"))

    all_tooth_x = []
    all_me_rotate = []
    all_me_translate = []
    for fi in range(0, len(dir_list_l)):
        file_list_l = []
        file_path_l = dir_list_l[fi]
        print(fi, '/', len(dir_list_l), '  ', file_path_l)
        file_list_start_l = []
        file_path_start_l = file_path_l.replace("end", "start")
        get_files(file_path_l, file_list_l, ".stl")
        get_files(file_path_start_l, file_list_start_l, ".stl")
        dir_name_l = os.path.split(file_path_l)[-1]

        file_list_u = []
        file_path_u = dir_list_u[fi]
        print(fi, '/', len(dir_list_l), '  ', file_path_u)
        file_list_start_u = []
        file_path_start_u = file_path_u.replace("end", "start")
        get_files(file_path_u, file_list_u, ".stl")
        get_files(file_path_start_u, file_list_start_u, ".stl")
        dir_name_u = os.path.split(file_path_u)[-1]

        train_data_u, train_label_u, teeth_center_u, Rweights_u, Tweights_u, gr_matrix_u, gdofs_u, gtrans_u, Gmove_bias_u, Rmove_bias_u = get_test_data(file_path_start_u, file_list_u, file_list_start_u)
        train_data_u = train_data_u.cuda().float()
        train_label_u = train_label_u.cuda().float()
        teeth_center_u = teeth_center_u.cuda().float()
        gdofs_u = torch.unsqueeze(torch.tensor(np.array(gdofs_u)), dim=0)
        gtrans_cp_u = torch.unsqueeze(torch.tensor(np.array(gtrans_u)), dim=0)
        gdofs_u = gdofs_u.cuda().float()
        gtrans_cp_u = gtrans_cp_u.cuda().float()
        Rweights_u = Rweights_u.cuda().float()
        Tweights_u = Tweights_u.cuda().float()
        weights_u = Rweights_u - 1 + Tweights_u
        with torch.no_grad():
            pdofs_u, ptrans_u = model_u(train_data_u, teeth_center_u)



        train_data_l, train_label_l, teeth_center_l, Rweights_l, Tweights_l, gr_matrix_l, gdofs_l, gtrans_l, Gmove_bias_l, Rmove_bias_l = get_test_data(file_path_start_l, file_list_l, file_list_start_l)
        train_data_l = train_data_l.cuda().float()
        train_label_l = train_label_l.cuda().float()
        teeth_center_l = teeth_center_l.cuda().float()
        gdofs_l = torch.unsqueeze(torch.tensor(np.array(gdofs_l)), dim=0)
        gtrans_cp_l = torch.unsqueeze(torch.tensor(np.array(gtrans_l)), dim=0)
        gdofs_l = gdofs_l.cuda().float()
        gtrans_cp_l = gtrans_cp_l.cuda().float()
        Rweights_l = Rweights_l.cuda().float()
        Tweights_l = Tweights_l.cuda().float()
        weights_l = Rweights_l - 1 + Tweights_l
        with torch.no_grad():
            pdofs_l, ptrans_l = model_l(train_data_l, teeth_center_l)

        # data = torch.squeeze(data)
        with torch.no_grad():
            assembled_u = tooth_assembler_u(train_data_u, teeth_center_u, pdofs_u, ptrans_u, device)
            assembled_l = tooth_assembler_l(train_data_l, teeth_center_l, pdofs_l, ptrans_l, device)

            recon_loss__u, c_loss__u = reconl1_loss(assembled_u, train_label_u, weights_u, device)
            recon_loss__l, c_loss__l = reconl1_loss(assembled_l, train_label_l, weights_l, device)

            cox_coincide_loss, sigma_groove_loss, sigma_incisors_loss = interdental_occlusion_loss(assembled_u, train_label_u, assembled_l, train_label_l, device)

            angle_pre_u = torch.norm(quaternion_to_axis_angle(pdofs_u), dim=-1) / math.pi * 180.0
            angle_pre_l = torch.norm(quaternion_to_axis_angle(pdofs_l), dim=-1) / math.pi * 180.0
            angle_gro_u = torch.norm(quaternion_to_axis_angle(gdofs_u), dim=-1) / math.pi * 180.0
            angle_gro_l = torch.norm(quaternion_to_axis_angle(gdofs_l), dim=-1) / math.pi * 180.0
            me_rotate__u = torch.mean(torch.abs(torch.sub(angle_pre_u, angle_gro_u)))
            me_rotate__l = torch.mean(torch.abs(torch.sub(angle_pre_l, angle_gro_l)))
            me_translate__u = torch.mean(torch.norm(torch.sub(ptrans_u, gtrans_cp_u), dim=1))
            me_translate__l = torch.mean(torch.norm(torch.sub(ptrans_l, gtrans_cp_l), dim=1))

            all_me_rotate.append(me_rotate__u)
            all_me_rotate.append(me_rotate__l)
            all_me_translate.append(me_translate__u)
            all_me_translate.append(me_translate__l)
            
            dof_loss__u = torch.sum(torch.sum(F.smooth_l1_loss(pdofs_u, gdofs_u, reduction= "none"), dim=-1) * Rweights_u) / pdofs_u.shape[0]
            dof_loss__l = torch.sum(torch.sum(F.smooth_l1_loss(pdofs_l, gdofs_l, reduction= "none"), dim=-1) * Rweights_l) / pdofs_l.shape[0]
            trans_loss__u = torch.sum(torch.sum(F.smooth_l1_loss(ptrans_u, gtrans_cp_u, reduction= "none"), dim=-1) * Tweights_u) / ptrans_u.shape[0]
            trans_loss__l = torch.sum(torch.sum(F.smooth_l1_loss(ptrans_l, gtrans_cp_l, reduction= "none"), dim=-1) * Tweights_l) / ptrans_l.shape[0]
            angle_loss__u = torch.sum(1-torch.sum(pdofs_u*gdofs_u, dim=-1)) / pdofs_u.shape[0]
            angle_loss__l = torch.sum(1-torch.sum(pdofs_l*gdofs_l, dim=-1)) / pdofs_l.shape[0]

            sym_loss__u = dof_loss__u
            sym_loss__l = dof_loss__l
            spl_loss__u = dof_loss__u
            spl_loss__l = dof_loss__l

            loss_u = recon_loss__u + c_loss__u * 1 + dof_loss__u * 10 + angle_loss__u + trans_loss__u * 1
            loss_l = recon_loss__l + c_loss__l * 1 + dof_loss__l * 10 + angle_loss__l + trans_loss__l * 1

            outstr = 'loss: %.6f, dof_loss: %.6f, trans_loss: %.6f, me_rotate: %.6f, me_translate: %.6f, ccoin_loss: %.6f, sg_loss: %.6f, si_loss: %.6f, angle_loss: %.6f, recon_loss: %.6f, sym_loss: %.6f, spl_loss: %.6f' % (
                (loss_u.item() + loss_l.item()) / 2,
                (dof_loss__u.item() + dof_loss__l.item()) / 2, (trans_loss__u.item() + trans_loss__l.item()) / 2,
                (me_rotate__u.item() + me_rotate__l.item()) / 2, (me_translate__u.item() + me_translate__l.item()) / 2,
                cox_coincide_loss.item(), sigma_groove_loss.item(), sigma_incisors_loss.item(),
                (angle_loss__u.item() + angle_loss__l.item()) / 2, ((recon_loss__u.item() + recon_loss__l.item())) / 2,
                (sym_loss__u.item() + sym_loss__l.item()) / 2, (spl_loss__u.item() + spl_loss__l.item()) / 2)
            print("iteration num:", iteration_con)
            print(outstr)
        
        dir_name_u = dir_name_u.replace("_end", "")
        dir_name_l = dir_name_l.replace("_end", "")
        ths_tooth_x_u = mapping_output(where_write, is_the_last, file_list_u, pdofs_u, ptrans_u, Gmove_bias_u, Rmove_bias_u, save_root + "/" + dir_name_u, dir_name_u)
        ths_tooth_x_l = mapping_output(where_write, is_the_last, file_list_l, pdofs_l, ptrans_l, Gmove_bias_l, Rmove_bias_l, save_root + "/" + dir_name_l, dir_name_l)
        print("up tooth num:", len(ths_tooth_x_u))
        print("down tooth num:", len(ths_tooth_x_l))
        all_tooth_x.append(ths_tooth_x_u)
        all_tooth_x.append(ths_tooth_x_l)
    
    all_tooth_add_auc = []
    for idd in range(len(all_tooth_x)):
        all_tooth_add_auc = all_tooth_add_auc + all_tooth_x[idd]
    all_me_rotate = torch.stack(all_me_rotate, dim=0)
    all_me_translate = torch.stack(all_me_translate, dim=0)

    ave_add = 0
    for i in range(len(all_tooth_add_auc)):
        ave_add += all_tooth_add_auc[i]
    ave_add /= len(all_tooth_add_auc)

    ADD_AUC = 0
    for sli in range(0, int(cfg.AUC_K / cfg.AUC_piece) + 1):
        cont = 0
        for i in range(len(all_tooth_add_auc)):
            if all_tooth_add_auc[i] <= sli * cfg.AUC_piece:
                cont = cont + 1
        print(sli * cfg.AUC_piece, ",", cont / len(all_tooth_add_auc))
        ADD_AUC = ADD_AUC + cfg.AUC_piece * (cont / len(all_tooth_add_auc))
    me_r = torch.mean(all_me_rotate).item()
    me_t = torch.mean(all_me_translate).item()
    return ADD_AUC, me_r, me_t, ave_add


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    model_u = teeth_arangement_model()
    model_l = teeth_arangement_model()
    model_path_u = "/home/charon/codeGala/dzx/orth-tooth/outputs/save_model/orth_model_u_550.pth"
    model_path_l = "/home/charon/codeGala/dzx/orth-tooth/outputs/save_model/orth_model_l_550.pth"
    model_initial(model_u, model_path_u)
    model_initial(model_l, model_path_l)
    model_u.cuda()
    model_l.cuda()
    model_u.eval()
    model_l.eval()

    iteration_con = 1

    start_dir = "/devdata/dzx_data/tadpmData/singleMesh/test_stl"
    iteration_dir = "/devdata/dzx_data/tadpmData/singleMesh/test_stl_iteration"
    
    for idx in range(iteration_con):
        if idx == 0:
            if iteration_con == 1:
                auc, me_r, me_t, ave_add = test(idx + 1, device, model_u, model_l, start_dir, iteration_dir, True)
            else:
                auc, me_r, me_t, ave_add = test(idx + 1, device, model_u, model_l, start_dir, iteration_dir, False)
            print("\n", "ADD_AUC: ", auc, "AVE_ADD: ", ave_add, "percentage: ", auc / cfg.AUC_K, "ME_rotate: ", me_r, "ME_trans: ", me_t)
        elif idx == iteration_con - 1:
            auc, me_r, me_t, ave_add = test(idx + 1, device, model_u, model_l, iteration_dir, iteration_dir, True)
            print("\n", "ADD_AUC: ", auc, "AVE_ADD: ", ave_add, "percentage: ", auc / cfg.AUC_K, "ME_rotate: ", me_r, "ME_trans: ", me_t)
        else:
            auc, me_r, me_t, ave_add = test(idx + 1, device, model_u, model_l, iteration_dir, iteration_dir, False)
            print("\n", "ADD_AUC: ", auc, "AVE_ADD: ", ave_add, "percentage: ", auc / cfg.AUC_K, "ME_rotate: ", me_r, "ME_trans: ", me_t)
    print("over")