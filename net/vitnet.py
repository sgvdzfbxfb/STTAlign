from functools import partial
import torch
import torch.nn as nn
import  torch.nn.functional as F
from image_models.timm.models.vision_transformer import PatchEmbed, Block
from image_models.timm.models.swin_transformer import SwinTransformer, SwinTransformerBlock
from net.pos_embed import PositionalEncoding
from config import config as cfg

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cplinear1 = nn.Linear(3, embed_dim)
        self.plinear1 = nn.Linear(3, embed_dim)
        self.encoder_embed = nn.Parameter(torch.zeros(1, cfg.teeth_nums, embed_dim), requires_grad=False)
        self.decoder_embed = nn.Parameter(torch.zeros(1, cfg.teeth_nums, embed_dim*5), requires_grad=False)
        self.teeth1 = nn.Parameter(torch.zeros(cfg.teeth_nums, 1, 1),requires_grad=True)
        self.teeth_blocks = nn.ModuleList([
            SwinTransformer(img_size=(cfg.teeth_nums, cfg.sam_points), patch_size=1, in_chans=3, window_size=8, num_classes=embed_dim, embed_dim=embed_dim, depths=(2, 2, 6, 2), qkv_bias=True, norm_layer=norm_layer)
            for i in range(1)])
        self.cp_blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, input_resolution=(cfg.teeth_nums, 1), num_heads=num_heads, window_size=8, shift_size=4, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim*2)
        self.tb_props = nn.ModuleList([
            SwinTransformerBlock(embed_dim*2, input_resolution=(cfg.teeth_nums, 1), num_heads=num_heads, window_size=8, shift_size=4, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.linear21 = nn.Linear(embed_dim*2, embed_dim)
        self.linear22 = nn.Linear(embed_dim, 3)
        self.linear23 = nn.Linear(embed_dim, 4)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))
        pos_embed2 = PositionalEncoding(self.decoder_embed.shape[1], self.decoder_embed.shape[2], self.device)
        self.decoder_embed.data.copy_(pos_embed2.float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        TB, C, N = x.shape
        x = x.permute(0, 2, 1)
        xc = self.cplinear1(x[:, 0:1, 3:]).permute(1, 0, 2)
        self.decoder_r = torch.tensor(cfg.decoder_r)
        self.decoder_r = self.decoder_r.to(torch.device('cuda', 0))
        self.decoder_t = torch.tensor(cfg.decoder_t)
        self.decoder_t = self.decoder_t.to(torch.device('cuda', 0))
        x = x[:, :, :3]
        xc = self.encoder_embed + xc
        xc = xc.permute(1, 0, 2)
        xc = torch.unsqueeze(xc, dim=0)
        for cpb in self.cp_blocks:
            xc = cpb(xc)
        xc = torch.squeeze(xc, dim=0)
        xc = xc.permute(1, 0, 2)
        teeths = []
        x = x.permute(2, 0, 1)
        x = torch.unsqueeze(x, dim=0)
        for blk in self.teeth_blocks:
            x = blk(x)
            t_x = torch.squeeze(x, dim=0)
            teeths.append(t_x)
        x = torch.squeeze(x, dim=0)
        x = x.permute(1, 2, 0)
        x = torch.cat(teeths, dim=-1)
        x = torch.mean(x, dim=1, keepdim=True)
        x = x.permute(1, 0, 2)
        x = torch.cat([x, xc],  dim=-1)
        x_ = x.clone().permute(1, 0, 2)
        x = x.permute(1, 0, 2)
        x = torch.unsqueeze(x, dim=0)
        for blk in self.tb_props:
            x = blk(x)
        x = torch.squeeze(x, dim=0)
        x = self.teeth1 * x + x_
        x = torch.mean(x, dim=1)
        x = F.relu(self.linear21(x))
        if torch.count_nonzero(self.decoder_t).item() < 5:
            transv = 10*F.tanh(self.linear22(x))
        else:
            transv = 10*F.tanh(self.linear22(x)) + self.decoder_t
        x = F.tanh(self.linear23(x))
        if torch.count_nonzero(self.decoder_r).item() < 5:
            dofx = torch.nn.functional.normalize(x, dim=-1)
        else:
            dofx = torch.nn.functional.normalize(x, dim=-1) + self.decoder_r
        return dofx, transv

    def forward(self, imgs, mask_ratio=0.75):
        dofx, transv = self.forward_encoder(imgs)
        return dofx, transv

def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=256, depth=4, num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model