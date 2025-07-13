from __future__ import print_function
import os
import config.config as cfg
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_divice_id
import time
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.load_train_data import TrainData, train_data_load
from torch.utils.data import DataLoader
import sklearn.metrics as metrics
from net.arangementnet import teeth_arangement_model
from util import IOStream, Tooth_Assembler
from net.loss import GeometricReconstructionLoss, interdental_occlusion_loss, symmetric_loss, spatial_Relation_Loss
from pytorch3d.transforms import *
from pytorch3d.transforms import quaternion_to_axis_angle
import math

def model_initial(model, model_name):
    pretrained_dict = torch.load(model_name)["model"]

    model_dict = model.state_dict()
    pretrained_dictf = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dictf)
    model.load_state_dict(model_dict)

    print("model initial over")


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('./outputs/' + args.exp_name):
        os.makedirs('./outputs/' + args.exp_name)
    if not os.path.exists('./outputs/' + args.exp_name + '/' + 'models'):
        os.makedirs('./outputs/' + args.exp_name + '/' + 'models')


def train(args, io):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_path = "/devdata/dzx_data/tadpmData/singleMesh/train/"
    train_loader = DataLoader(TrainData(file_path), num_workers=0,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    model_u = teeth_arangement_model()
    model_l = teeth_arangement_model()
    tooth_assembler_u = Tooth_Assembler()
    tooth_assembler_l = Tooth_Assembler()
    reconl1_loss = GeometricReconstructionLoss()
    model_path_u = "./outputs/save_model/orth_model_u_550.pth"
    model_path_l = "./outputs/save_model/orth_model_l_550.pth"
    model_initial(model_u, model_path_u)
    model_initial(model_l, model_path_l)

    if args.use_sgd:
        print("Use SGD")
        opt_u = optim.SGD([{'params': model_u.local_fea.parameters(), 'lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
        opt_l = optim.SGD([{'params': model_l.local_fea.parameters(), 'lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=1e-4)

    else:
        print("Use Adam")
        opt_u = optim.Adam(model_u.parameters(), lr=args.lr, weight_decay=1e-4)
        opt_l = optim.Adam(model_l.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.scheduler == 'cos':
        scheduler_u = CosineAnnealingLR(opt_u, args.epochs, eta_min=1e-6, last_epoch = -1)
        scheduler_l = CosineAnnealingLR(opt_l, args.epochs, eta_min=1e-6, last_epoch = -1)
    elif args.scheduler == 'step':
        scheduler_u = StepLR(opt_u, step_size=20, gamma=0.7)
        scheduler_l = StepLR(opt_l, step_size=20, gamma=0.7)
    model_u.cuda()
    model_l.cuda()
    model_u.train()
    model_l.train()
    scaler_u = GradScaler()
    scaler_l = GradScaler()
    best_test_acc = 0
    inter_nums = len(train_loader)

    for epoch in range(args.epochs):
        ####################
        # Train
        ####################

        if args.scheduler == 'cos':
            scheduler_u.step()
            scheduler_l.step()
        elif args.scheduler == 'step':
            if opt_u.param_groups[0]['lr'] > 1e-5:
                scheduler_u.step()
            if opt_l.param_groups[0]['lr'] > 1e-5:
                scheduler_l.step()
            if opt_u.param_groups[0]['lr'] < 1e-5:
                for param_group in opt_u.param_groups:
                    param_group['lr'] = 1e-5
            if opt_l.param_groups[0]['lr'] < 1e-5:
                for param_group in opt_l.param_groups:
                    param_group['lr'] = 1e-5

        train_loss_u = 0.0
        train_loss_l = 0.0
        count = 0.0
        recon_loss_u = 0
        recon_loss_l = 0
        c_loss_u = 0
        c_loss_l = 0
        dof_loss_u = 0
        dof_loss_l = 0
        trans_loss_u = 0
        trans_loss_l = 0
        angle_loss_u = 0
        angle_loss_l = 0
        sym_loss_u = 0
        sym_loss_l = 0
        spl_loss_u = 0
        spl_loss_l = 0
        me_rotate_u = 0
        me_rotate_l = 0
        me_translate_u = 0
        me_translate_l = 0

        ccoin_loss = 0
        sig_loss = 0
        sii_loss = 0
        nums = 0
        tic = time.time()
        train_data_u, train_label_u, teeth_center_u, dof_u = [], [], [], []
        train_data_l, train_label_l, teeth_center_l, dof_l = [], [], [], []
        nnums = 0
        for cafh_l in train_loader:
            cafh_u = []
            for fl in cafh_l:
                cafh_u.append(fl.replace("_down", "_up"))
            train_data_u, train_label_u, teeth_center_u, gdofs_u, gtrans_u, tweights_u, rweights_u, mask_index_u =  train_data_load(cafh_u)
            train_data_l, train_label_l, teeth_center_l, gdofs_l, gtrans_l, tweights_l, rweights_l, mask_index_l =  train_data_load(cafh_l)
            train_data_u = train_data_u.cuda().float()
            train_data_l = train_data_l.cuda().float()
            train_label_u = train_label_u.cuda().float()
            train_label_l = train_label_l.cuda().float()
            teeth_center_u = teeth_center_u.cuda().float()
            teeth_center_l = teeth_center_l.cuda().float()
            gdofs_u = gdofs_u.cuda().float()
            gdofs_l = gdofs_l.cuda().float()
            gtrans_u = gtrans_u.cuda().float()
            gtrans_l = gtrans_l.cuda().float()
            tweights_u = tweights_u.cuda().float()
            tweights_l = tweights_l.cuda().float()
            rweights_u = rweights_u.cuda().float()
            rweights_l = rweights_l.cuda().float()

            mask_index_u = mask_index_u.cuda().long()
            mask_index_l = mask_index_l.cuda().long()

            weights_u = rweights_u - 1 + tweights_u
            weights_l = rweights_l - 1 + tweights_l

            gdofs_u = gdofs_u
            gdofs_l = gdofs_l

            nums = nums + 1
            batch_size = train_data_u.size()[0]
            opt_u.zero_grad()
            opt_l.zero_grad()
            with autocast():
                pdofs_u, ptrans_u = model_u(train_data_u, teeth_center_u)
                pdofs_l, ptrans_l = model_l(train_data_l, teeth_center_l)
                assembled_u = tooth_assembler_u(train_data_u, teeth_center_u, pdofs_u, ptrans_u, device)
                assembled_l = tooth_assembler_l(train_data_l, teeth_center_l, pdofs_l, ptrans_l, device)

                nnums = nnums + 1
                recon_loss__u, c_loss__u = reconl1_loss(assembled_u, train_label_u, weights_u, device)
                recon_loss__l, c_loss__l = reconl1_loss(assembled_l, train_label_l, weights_l, device)
                cox_coincide_loss, sigma_groove_loss, sigma_incisors_loss = interdental_occlusion_loss(assembled_u, train_label_u, assembled_l, train_label_l, device)

                angle_pre_u = torch.norm(quaternion_to_axis_angle(pdofs_u[mask_index_u]), dim=-1) / math.pi * 180.0
                angle_pre_l = torch.norm(quaternion_to_axis_angle(pdofs_l[mask_index_l]), dim=-1) / math.pi * 180.0
                angle_gro_u = torch.norm(quaternion_to_axis_angle(gdofs_u[mask_index_u]), dim=-1) / math.pi * 180.0
                angle_gro_l = torch.norm(quaternion_to_axis_angle(gdofs_l[mask_index_l]), dim=-1) / math.pi * 180.0
                me_rotate__u = torch.mean(torch.abs(torch.sub(angle_pre_u, angle_gro_u)))
                me_rotate__l = torch.mean(torch.abs(torch.sub(angle_pre_l, angle_gro_l)))
                me_translate__u = torch.mean(torch.norm(torch.sub(ptrans_u[mask_index_u], gtrans_u[mask_index_u]), dim=1))
                me_translate__l = torch.mean(torch.norm(torch.sub(ptrans_l[mask_index_l], gtrans_l[mask_index_l]), dim=1))

                dof_loss__u = torch.sum(torch.sum(F.smooth_l1_loss(pdofs_u[mask_index_u], gdofs_u[mask_index_u], reduction= "none"), dim=-1) * rweights_u[mask_index_u]) / pdofs_u[mask_index_u].shape[0]
                dof_loss__l = torch.sum(torch.sum(F.smooth_l1_loss(pdofs_l[mask_index_l], gdofs_l[mask_index_l], reduction= "none"), dim=-1) * rweights_l[mask_index_l]) / pdofs_l[mask_index_l].shape[0]
                trans_loss__u = torch.sum(torch.sum(F.smooth_l1_loss(ptrans_u[mask_index_u], gtrans_u[mask_index_u], reduction= "none"), dim=-1) * tweights_u[mask_index_u]) / ptrans_u[mask_index_u].shape[0]
                trans_loss__l = torch.sum(torch.sum(F.smooth_l1_loss(ptrans_l[mask_index_l], gtrans_l[mask_index_l], reduction= "none"), dim=-1) * tweights_l[mask_index_l]) / ptrans_l[mask_index_l].shape[0]
                angle_loss__u = torch.sum(1-torch.sum(pdofs_u[mask_index_u]*gdofs_u[mask_index_u], dim=-1)) / pdofs_u[mask_index_u].shape[0]
                angle_loss__l = torch.sum(1-torch.sum(pdofs_l[mask_index_l]*gdofs_l[mask_index_l], dim=-1)) / pdofs_l[mask_index_l].shape[0]

                sym_loss__u = dof_loss__u
                sym_loss__l = dof_loss__l
                spl_loss__u = dof_loss__u
                spl_loss__l = dof_loss__l

                loss_u = recon_loss__u + c_loss__u * 1 + dof_loss__u * 10 + angle_loss__u + trans_loss__u
                loss_l = recon_loss__l + c_loss__l * 1 + dof_loss__l * 10 + angle_loss__l + trans_loss__l

            scaler_u.scale(loss_u + cox_coincide_loss.detach() + sigma_groove_loss.detach() + sigma_incisors_loss.detach()).backward()
            scaler_l.scale(loss_l + cox_coincide_loss.detach() + sigma_groove_loss.detach() + sigma_incisors_loss.detach()).backward()
            scaler_u.step(opt_u)
            scaler_l.step(opt_l)
            scaler_u.update()
            scaler_l.update()

            count += batch_size
            train_loss_u += loss_u.item()
            train_loss_l += loss_l.item()
            recon_loss_u += recon_loss__u.item()
            recon_loss_l += recon_loss__l.item()
            c_loss_u += c_loss__u.item()
            c_loss_l += c_loss__l.item()
            dof_loss_u += dof_loss__u.item()
            dof_loss_l += dof_loss__l.item()
            trans_loss_u += trans_loss__u.item()
            trans_loss_l += trans_loss__l.item()
            angle_loss_u += angle_loss__u.item()
            angle_loss_l += angle_loss__l.item()
            sym_loss_u += sym_loss__u.item()
            sym_loss_l += sym_loss__l.item()
            spl_loss_u += spl_loss__u.item()
            spl_loss_l += spl_loss__l.item()
            me_rotate_u += me_rotate__u.item()
            me_rotate_l += me_rotate__l.item()
            me_translate_u += me_translate__u.item()
            me_translate_l += me_translate__l.item()

            ccoin_loss += cox_coincide_loss.item()
            sig_loss += sigma_groove_loss.item()
            sii_loss += sigma_incisors_loss.item()

            if nums % cfg.VIEW_NUMS == 0:
                toc = time.time()
                train_loss_u = train_loss_u/(cfg.VIEW_NUMS)
                train_loss_l = train_loss_l/(cfg.VIEW_NUMS)
                recon_loss_u = recon_loss_u/(cfg.VIEW_NUMS)
                recon_loss_l = recon_loss_l/(cfg.VIEW_NUMS)
                c_loss_u = c_loss_u/(cfg.VIEW_NUMS)
                c_loss_l = c_loss_l/(cfg.VIEW_NUMS)
                dof_loss_u = dof_loss_u/(cfg.VIEW_NUMS)
                dof_loss_l = dof_loss_l/(cfg.VIEW_NUMS)
                trans_loss_u = trans_loss_u/(cfg.VIEW_NUMS)
                trans_loss_l = trans_loss_l/(cfg.VIEW_NUMS)
                angle_loss_u = angle_loss_u/(cfg.VIEW_NUMS)
                angle_loss_l = angle_loss_l/(cfg.VIEW_NUMS)
                sym_loss_u = sym_loss_u/(cfg.VIEW_NUMS)
                sym_loss_l = sym_loss_l/(cfg.VIEW_NUMS)
                spl_loss_u = spl_loss_u/(cfg.VIEW_NUMS)
                spl_loss_l = spl_loss_l/(cfg.VIEW_NUMS)
                me_rotate_u = me_rotate_u/(cfg.VIEW_NUMS)
                me_rotate_l = me_rotate_l/(cfg.VIEW_NUMS)
                me_translate_u = me_translate_u/(cfg.VIEW_NUMS)
                me_translate_l = me_translate_l/(cfg.VIEW_NUMS)

                ccoin_loss = ccoin_loss/(cfg.VIEW_NUMS)
                sig_loss = sig_loss/(cfg.VIEW_NUMS)
                sii_loss = sii_loss/(cfg.VIEW_NUMS)

                print("Orth-Tooth lr = ", (opt_u.param_groups[0]['lr'] + opt_u.param_groups[0]['lr']) / 2)
                outstr = 'Orth-Tooth epoch %d /%d,epoch %d /%d, loss: %.6f, recon_loss: %.6f, c_loss: %.6f, dof_loss: %.6f, trans_loss: %.6f, me_rotate: %.6f, me_trans: %.6f, ccoin_loss: %.6f, sg_loss: %.6f, si_loss: %.6f, sym_loss: %.6f, spl_loss: %.6f, angle_loss: %.6f, const time: %.6f' % (
                 epoch,args.epochs, nums, inter_nums, (train_loss_u + train_loss_l) / 2, (recon_loss_u + recon_loss_l) / 2, (c_loss_u + c_loss_l) / 2,
                 (dof_loss_u + dof_loss_l) / 2, (trans_loss_u + trans_loss_l) / 2, (me_rotate_u + me_rotate_l) / 2, (me_translate_u + me_translate_l) / 2,
                 ccoin_loss, sig_loss, sii_loss,
                 (sym_loss_u + sym_loss_l) / 2, (spl_loss_u + spl_loss_l) / 2, (angle_loss_u + angle_loss_l) / 2, toc - tic)

                io.cprint(outstr)

                train_loss_u = 0.0
                train_loss_l = 0.0
                count = 0.0
                recon_loss_u = 0
                recon_loss_l = 0
                c_loss_u = 0
                c_loss_l = 0
                dof_loss_u = 0
                dof_loss_l = 0
                trans_loss_u = 0
                trans_loss_l = 0
                angle_loss_u = 0
                angle_loss_l = 0
                sym_loss_u = 0
                sym_loss_l = 0
                spl_loss_u = 0
                spl_loss_l = 0
                me_rotate_u = 0
                me_rotate_l = 0
                me_translate_u = 0
                me_translate_l = 0

                ccoin_loss = 0
                sig_loss = 0
                sii_loss = 0
                tic = time.time()

        save_model_path = "outputs/save_model"
        if not os.path.exists(save_model_path):
            os.mkdir(save_model_path)
        if epoch > 0 and (epoch) % cfg.SAVE_MODEL == 0:
            torch.save({'model': model_u.state_dict(), 'epoch': epoch}, 'outputs/save_model/orth_model_u_' + str(epoch) + '.pth')
            torch.save({'model': model_l.state_dict(), 'epoch': epoch}, 'outputs/save_model/orth_model_l_' + str(epoch) + '.pth')


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser(description='Orth-Tooth')
    parser.add_argument('--exp_name', type=str, default='exp_stand1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='swin', metavar='N',
                        choices=['vision', 'swin'],
                        help='Model to use, [vision, swin]')
    parser.add_argument('--dataset', type=str, default='hsf', metavar='N',
                        choices=['hsf'])
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1001, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=1.5*1e-4, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='initial dropout rate')
    parser.add_argument('--emb_dims', type=int, default=2048, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=40, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    runlog_path = 'outputs/' + args.exp_name + '/run_.log'
    io = IOStream(runlog_path)
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)