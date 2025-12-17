import numpy as np
import torch
from einops import rearrange
from torch import nn
import torch.nn.functional as F
from model.cvt import ConvEmbed, Block

class MaskedAutoencoderCvT(nn.Module):
    def __init__(self, img_size=(512,512), patch_size=16, in_chans=9, out_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 use_only_masked_tokens_ab=False, abnormal_score_func='L1', masking_method="random_masking",
                 grad_weighted_loss=False): 
        super().__init__()
        
        # --------------------------------------------------------------------------
        # Abnormal specifics
        self.use_only_masked_tokens_ab = use_only_masked_tokens_ab
        self.abnormal_score_func = abnormal_score_func[0] if isinstance(abnormal_score_func, list) else abnormal_score_func
        self.masking = getattr(self, masking_method)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans, # Defaults to 9 (3x3 frames)
            stride=patch_size,
            padding=0,
            embed_dim=embed_dim,
            norm_layer=norm_layer
        )
        self.patch_size = patch_size
        self.num_patches = img_size[0]//patch_size*img_size[1]//patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(embed_dim, embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)
        self.out_chans = out_chans
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

    def freeze_backbone(self):
        # Unused, kept for interface compatibility
        pass

    def patchify(self, imgs):
        """
        imgs: (N, C, H, W) where C is self.out_chans (3)
        x: (N, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.out_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.out_chans))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = 20 # Note: Ensure this matches input size in real deployment
        w = 40 
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.out_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.out_chans, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio, grad_mask=None):
        N, D, H, W = x.shape
        L = H*W
        x = rearrange(x, 'b c h w -> b (h w) c')
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        self.masked_H = H
        self.masked_W = int(W*(1.-mask_ratio))
        self.H = H
        self.W = W
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, grad_mask=None):
        x = self.patch_embed(x)
        x, mask, ids_restore = self.masking(x, mask_ratio, grad_mask)
        cls_token = self.cls_token
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x, self.masked_H, self.masked_W)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        for blk in self.decoder_blocks:
            x = blk(x, self.H, self.W)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        Pure MSE Reconstruction Loss (No Gradient Weighting)
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        # Loss on all patches or just masked? Standard MAE usually does masked only.
        # However, for anomaly detection, minimizing reconstruction error on all patches is common.
        # Based on "Sum ||P - Y||", it implies summing over L.
        # We will compute loss over ALL patches here to align with "Sum" over all L.
        # If masked only was desired: loss = (loss * mask).sum() / mask.sum()
        loss = loss.mean() 
        return loss

    def forward(self, imgs, targets, grad_mask=None, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, grad_mask)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(targets, pred, mask)
        
        if self.training:
            return loss, pred, mask
        else:
            return loss, pred, mask, self.abnormal_score(targets, pred, mask)

    def abnormal_score(self, imgs, pred, mask):
        # Computes s_t
        imgs = self.patchify(imgs)
        return ((imgs - pred) ** 2).mean(-1)