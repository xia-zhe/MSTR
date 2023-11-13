#!/usr/bin/python3
# -*- coding: utf-8 -*-

# system, numpy
import os
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional

import tensorly as tl
from tensorly.tenalg import inner as tl_inner
from tensorly.decomposition import tucker
import math


torch.pi = math.pi
tl.set_backend('pytorch')

# user defined
import src.utils_improvements

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        # functional.reset_net(self.layers)
        return x

class TRL(nn.Module):
    def __init__(self, input_size, ranks, output_size, verbose=1, **kwargs):
        super(TRL, self).__init__(**kwargs)
        self.ranks = list(ranks)
        self.verbose = verbose

        if isinstance(output_size, int):
            self.input_size = [input_size]
        else:
            self.input_size = list(input_size)

        if isinstance(output_size, int):
            self.output_size = [output_size]
        else:
            self.output_size = list(output_size)

        self.n_outputs = int(np.prod(output_size[1:]))

        # Core of the regression tensor weights
        self.core = nn.Parameter(tl.zeros(self.ranks), requires_grad=True)
        #self.core = nn.Parameter(torch.tensor(tl.zeros(self.ranks)), requires_grad=True)
        self.bias = nn.Parameter(tl.zeros(1), requires_grad=True)
        #self.bias = nn.Parameter(torch.tensor(tl.zeros(1)), requires_grad=True)
        weight_size = list(self.input_size[1:]) + list(self.output_size[1:])

        # Add and register the factors
        self.factors = []
        for index, (in_size, rank) in enumerate(zip(weight_size, ranks)):
            #self.factors.append(nn.Parameter(tl.zeros((in_size, rank)), requires_grad=True))
            self.factors.append(nn.Parameter(torch.tensor(tl.zeros((in_size, rank))), requires_grad=True))

            self.register_parameter('factor_{}'.format(index), self.factors[index])

        # FIX THIS
        self.core.data.uniform_(-0.1, 0.1)
        for f in self.factors:
            f.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        tucker_tensor = (self.core, self.factors)
        regression_weights = tl.tucker_to_tensor(tucker_tensor)
        return tl_inner(x, regression_weights, n_modes=tl.ndim(x) - 1) + self.bias

    def penalty(self, order=2):
        penalty = tl.norm(self.core, order)
        for f in self.factors:
            penalty = penalty + tl.norm(f, order)
        return penalty




class EmbeddingNet(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, momentum,hidden_size=None):
        super(EmbeddingNet, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            if use_bn:
                modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        else:
            modules.append(nn.Linear(in_features=input_size, out_features=output_size))
            modules.append(nn.BatchNorm1d(num_features=output_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)

class SNNBranch(nn.Module):
    def __init__(self, input_size, output_size, dropout, use_bn, thd, momentum,hidden_size=None):
        super(SNNBranch, self).__init__()
        modules = []
        if hidden_size:
            modules.append(nn.Linear(in_features=input_size, out_features=hidden_size))
            # if use_bn:
            #     modules.append(nn.BatchNorm1d(num_features=hidden_size))
            modules.append(neuron.IFNode((v_threshold=thd)))
            # modules.append(nn.Dropout(dropout))
            modules.append(nn.Linear(in_features=hidden_size, out_features=output_size))
            # modules.append(nn.BatchNorm1d(num_features=output_size, momentum=momentum))
            modules.append(neuron.IFNode((v_threshold=thd)))
            # modules.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*modules)

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)



class AVCA(nn.Module):
    def __init__(self, params_model, input_size_audio, input_size_video):
        super(AVCA, self).__init__()

        print('Initializing model variables...', end='')
        # Dimension of embedding
        self.dim_out = params_model['dim_out']
        # Number of classes
        self.hidden_size_encoder=params_model['encoder_hidden_size']
        self.hidden_size_decoder=params_model['decoder_hidden_size']
        self.r_enc=params_model['dropout_encoder']#0.2 0.3
        self.r_proj=params_model['dropout_decoder']#0.3 0.1
        self.depth_transformer=params_model['depth_transformer']
        self.additional_triplets_loss=params_model['additional_triplets_loss']
        self.reg_loss=params_model['reg_loss']
        self.r_dec=params_model['additional_dropout']#0.5 0.15
        self.momentum=params_model['momentum']
        self.thd=params_model['thd']

        self.first_additional_triplet=params_model['first_additional_triplet']
        self.second_additional_triplet=params_model['second_additional_triplet']

        print('Initializing trainable models...', end='')

        self.A_enc = EmbeddingNet(
            input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )
        self.V_enc = EmbeddingNet(
            input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            use_bn=True
        )

        self.trl_v = TRL(ranks=(400, 1, 1, 300), input_size=(256, 512, 1, 1), output_size=(256, 300))
        self.trl_a = TRL(ranks=(400, 1, 1, 300), input_size=(256, 512, 1, 1), output_size=(256, 300))
        self.cross_attention=Transformer(300, self.depth_transformer, 3, 100, 64, dropout=self.r_enc)

        self.W_proj= EmbeddingNet(
            input_size=300,
            output_size=self.dim_out,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )

        self.D = EmbeddingNet(
            input_size=self.dim_out,
            output_size=300,
            dropout=self.r_dec,
            momentum=self.momentum,
            use_bn=True
        )


        self.SNNbranchaudio = SNNBranch(input_size=input_size_audio,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            thd=self.thd,
            use_bn=True)
        self.SNNbranchvideo = SNNBranch(input_size=input_size_video,
            hidden_size=self.hidden_size_encoder,
            output_size=300,
            dropout=self.r_enc,
            momentum=self.momentum,
            thd=self.thd,
            use_bn=True)
        self.A_proj = EmbeddingNet(input_size=300, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.V_proj = EmbeddingNet(input_size=300, hidden_size=self.hidden_size_decoder, output_size=self.dim_out, dropout=self.r_proj, momentum=self.momentum,use_bn=True)

        self.A_rec = EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.V_rec = EmbeddingNet(input_size=self.dim_out, output_size=300, dropout=self.r_dec, momentum=self.momentum, use_bn=True)

        self.pos_emb1D = torch.nn.Parameter(torch.randn(2, 300))
        self.pos_emb1D_t = torch.nn.Parameter(torch.randn(2, 300))
        self.T = 10
        # Optimizers
        print('Defining optimizers...', end='')
        self.lr = params_model['lr']
        self.optimizer_gen = optim.Adam(list(self.A_proj.parameters()) + list(self.V_proj.parameters()) +
                                        list(self.A_rec.parameters()) + list(self.V_rec.parameters()) +
                                        list(self.V_enc.parameters()) + list(self.A_enc.parameters()) +
                                        list(self.cross_attention.parameters()) + list(self.D.parameters()) +
                                        list(self.W_proj.parameters())+list(self.SNNbranchaudio.parameters())+list(self.SNNbranchvideo.parameters())
                                        +list(self.trl_a.parameters())+list(self.trl_v.parameters()),
                                        lr=self.lr, weight_decay=1e-5)

        self.scheduler_gen =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, 'max', patience=3, verbose=True)

        print('Done')

        # Loss function
        print('Defining losses...', end='')
        self.criterion_reg = nn.MSELoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        print('Done')

    def optimize_scheduler(self, value):
        self.scheduler_gen.step(value)

    def forward(self, audio, image, negative_audio, negative_image, word_embedding, negative_word_embedding):

        self.phi_a = self.A_enc(audio)
        self.phi_v = self.V_enc(image)
        phi_a1 = self.SNNbranchaudio(audio)
        for t in range(1, self.T):
            phi_a1 += self.SNNbranchaudio(audio)
        self.phi_a1 = phi_a1/self.T

        phi_v1= self.SNNbranchvideo(image)
        for t in range(1, self.T):
            phi_v1 += self.SNNbranchaudio(image)
        self.phi_v1 = phi_v1/self.T

        # self.phi_a = 0.5*self.phi_a + 0.5*self.phi_a*nn.functional.softmax(self.phi_a1)
        self.phi_input = torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_a*nn.functional.softmax(self.phi_a1, dim=1) + self.pos_emb1D[0, :]), dim=1)
        self.phi_a= self.cross_attention(self.phi_input)[:, 0, :]
        # self.phi_v = 0.5*self.phi_v + 0.5*self.phi_v*nn.functional.softmax(self.phi_v1)
        self.phi_vinput = torch.stack((self.phi_v + self.pos_emb1D[1, :], self.phi_v*nn.functional.softmax(self.phi_v1, dim=1) + self.pos_emb1D[1, :]), dim=1)
        self.phi_v = self.cross_attention(self.phi_vinput)[:, 1, :]
        self.phi_a_neg=self.A_enc(negative_audio)
        self.phi_v_neg=self.V_enc(negative_image)

        phi_a_neg1=self.SNNbranchaudio(negative_audio)
        for t in range(1, self.T):
            phi_a_neg1 += self.SNNbranchaudio(negative_audio)
        self.phi_a_neg1 = phi_a_neg1/self.T
        phi_v_neg1=self.SNNbranchvideo(negative_image)
        for t in range(1, self.T):
            phi_v_neg1 += self.SNNbranchaudio(negative_image)
        self.phi_v_neg1 = phi_v_neg1/self.T
        # self.phi_a_neg = 0.5*self.phi_a_neg + 0.5*self.phi_a_neg*nn.functional.softmax(self.phi_a_neg1)
        self.phi_a_neg_input = torch.stack((self.phi_a_neg + self.pos_emb1D[0, :], self.phi_a_neg*nn.functional.softmax(self.phi_a_neg1, dim=1) + self.pos_emb1D[0, :]), dim=1)
        self.phi_a_neg= self.cross_attention(self.phi_a_neg_input)[:, 0, :]
        # self.phi_v_neg = 0.5*self.phi_v_neg + 0.5*self.phi_v_neg*nn.functional.softmax(self.phi_v_neg1)
        self.phi_v_neg_input = torch.stack((self.phi_v_neg + self.pos_emb1D[1, :], self.phi_v_neg*nn.functional.softmax(self.phi_v_neg1, dim=1) + self.pos_emb1D[1, :]), dim=1)
        self.phi_v_neg= self.cross_attention(self.phi_v_neg_input)[:, 1, :]
        functional.reset_net(self.SNNbranchvideo)
        functional.reset_net(self.SNNbranchaudio)

        self.phi_at = self.trl_a(audio.reshape((256, 512, 1, 1)))
        self.phi_vt = self.trl_v(image.reshape((256, 512, 1, 1)))

        self.phi_at_neg = self.trl_a(negative_audio.reshape((256, 512, 1, 1)))
        self.phi_vt_neg = self.trl_v(negative_image.reshape((256, 512, 1, 1)))

        self.positive_input_t = torch.stack((self.phi_at + self.pos_emb1D_t[0, :], self.phi_vt + self.pos_emb1D_t[1, :]), dim=1)
        self.negative_input_t = torch.stack((self.phi_at_neg + self.pos_emb1D_t[0, :], self.phi_vt_neg + self.pos_emb1D_t[1, :]),dim=1)

        self.phi_attn_t= self.cross_attention(self.positive_input_t)
        self.phi_attn_neg_t = self.cross_attention(self.negative_input_t)

        self.audio_fe_attn_t = self.phi_at + self.phi_attn_t[:, 0, :]
        self.video_fe_attn_t = self.phi_vt + self.phi_attn_t[:, 1, :]
        self.audio_fe_neg_attn_t = self.phi_at_neg + self.phi_attn_neg_t[:, 0, :]
        self.video_fe_neg_attn_t = self.phi_vt_neg + self.phi_attn_neg_t[:, 1, :]


        self.w=word_embedding
        self.w_neg=negative_word_embedding

        self.theta_w = self.W_proj(word_embedding)
        self.theta_w_neg=self.W_proj(negative_word_embedding)
        # functional.reset_net(self.W_proj)
        self.rho_w=self.D(self.theta_w)
        self.rho_w_neg=self.D(self.theta_w_neg)

        self.positive_input=torch.stack((self.phi_a + self.pos_emb1D[0, :], self.phi_v + self.pos_emb1D[1, :]), dim=1)
        self.negative_input=torch.stack((self.phi_a_neg + self.pos_emb1D[0, :], self.phi_v_neg + self.pos_emb1D[1, :]), dim=1)

        self.phi_attn= self.cross_attention(self.positive_input)

        self.phi_attn_neg = self.cross_attention(self.negative_input)

        self.audio_fe_attn1 = self.phi_a + self.phi_attn[:, 0, :]
        self.video_fe_attn1= self.phi_v + self.phi_attn[:, 1, :]

        self.audio_fe_attn=torch.stack((self.audio_fe_attn1 + self.pos_emb1D[0, :],self.video_fe_attn_t + self.pos_emb1D_t[0, :]),dim=1)
        self.audio_fe_attn=self.cross_attention(self.audio_fe_attn)[:, 0, :]
        self.video_fe_attn=torch.stack((self.video_fe_attn1 + self.pos_emb1D[1, :],self.audio_fe_attn_t + self.pos_emb1D_t[1, :]),dim=1)
        self.video_fe_attn=self.cross_attention(self.video_fe_attn)[:, 0, :]

        self.audio_fe_neg_attn1 = self.phi_a_neg + self.phi_attn_neg[:, 0, :]
        self.video_fe_neg_attn1 = self.phi_v_neg + self.phi_attn_neg[:, 1, :]

        self.audio_fe_neg_attn=torch.stack((self.audio_fe_neg_attn1 + self.pos_emb1D[0, :],self.video_fe_neg_attn_t + self.pos_emb1D_t[0, :]),dim=1)
        self.audio_fe_neg_attn=self.cross_attention(self.audio_fe_neg_attn)[:, 0, :]
        self.video_fe_neg_attn=torch.stack((self.video_fe_neg_attn1 + self.pos_emb1D[1, :],self.audio_fe_neg_attn_t + self.pos_emb1D_t[1, :]),dim=1)
        self.video_fe_neg_attn=self.cross_attention(self.video_fe_neg_attn)[:, 0, :]

        self.theta_v = self.V_proj(self.video_fe_attn)
        self.theta_v_neg=self.V_proj(self.video_fe_neg_attn)
        self.theta_a = self.A_proj(self.audio_fe_attn)
        self.theta_a_neg=self.A_proj(self.audio_fe_neg_attn)

        self.phi_v_rec = self.V_rec(self.theta_v)
        self.phi_a_rec = self.A_rec(self.theta_a)
        self.se_em_hat1 = self.A_proj(self.phi_a_rec)
        self.se_em_hat2 = self.V_proj(self.phi_v_rec)


        self.rho_a=self.D(self.theta_a)
        self.rho_a_neg=self.D(self.theta_a_neg)
        self.rho_v=self.D(self.theta_v)
        self.rho_v_neg=self.D(self.theta_v_neg)
        # functional.reset_net(self.D)
        # functional.reset_net(self.V_rec)
        # functional.reset_net(self.A_rec)
        # functional.reset_net(self.A_proj)
        # functional.reset_net(self.V_proj)

    def backward(self, optimize):

        if self.additional_triplets_loss==True:
            first_pair = self.first_additional_triplet*(self.triplet_loss(self.theta_a, self.theta_w, self.theta_a_neg) + \
                                                        self.triplet_loss(self.theta_v, self.theta_w, self.theta_v_neg))
            second_pair=self.second_additional_triplet*(self.triplet_loss(self.theta_w, self.theta_a, self.theta_w_neg) + \
                                                        self.triplet_loss(self.theta_w, self.theta_v, self.theta_w_neg))

            l_t=first_pair+second_pair

        if self.reg_loss==True:
            l_r = (self.criterion_reg(self.phi_v_rec, self.phi_v) + \
                            self.criterion_reg(self.phi_a_rec, self.phi_a) + \
                            self.criterion_reg(self.theta_v, self.theta_w) + \
                            self.criterion_reg(self.theta_a, self.theta_w))


        l_rec= self.criterion_reg(self.w, self.rho_v) + \
                  self.criterion_reg(self.w, self.rho_a) + \
                  self.criterion_reg(self.w, self.rho_w)

        l_ctv=self.triplet_loss(self.rho_w, self.rho_v, self.rho_v_neg)
        l_cta=self.triplet_loss(self.rho_w, self.rho_a, self.rho_a_neg)
        l_ct=l_cta+l_ctv
        l_cmd=l_rec+l_ct

        l_tv = self.triplet_loss(self.theta_w, self.theta_v, self.theta_v_neg)
        l_ta = self.triplet_loss(self.theta_w, self.theta_a, self.theta_a_neg)
        l_at = self.triplet_loss(self.theta_a, self.theta_w, self.theta_w_neg)
        l_vt = self.triplet_loss(self.theta_v, self.theta_w, self.theta_w_neg)

        l_w=l_ta+l_at+l_tv+l_vt

        loss_gen=l_cmd + l_w
        if self.additional_triplets_loss==True:
           loss_gen+=l_t
        if self.reg_loss==True:
            loss_gen+=l_r

        if optimize == True:
            self.optimizer_gen.zero_grad()
            loss_gen.backward()
            self.optimizer_gen.step()

        loss = {'aut_enc': 0,  'gen_cyc': 0,
                'gen_reg': 0, 'gen': loss_gen}

        loss_numeric = loss['gen_cyc'] + loss['gen']

        return loss_numeric, loss

    def optimize_params(self, audio, video, cls_numeric, cls_embedding,audio_negative, video_negative, negative_cls_embedding,optimize=False):

        self.forward(audio, video, audio_negative, video_negative, cls_embedding, negative_cls_embedding)

        loss_numeric, loss = self.backward(optimize)

        return loss_numeric, loss

    def get_embeddings(self, audio, video, embedding):
        # print('audio size = {}'.format(audio.size()))
        # print('video size {}'.format(video.size()))
        # print('embedding size {}'.format(embedding.size()))
        # print("********************")
        # audio= torch.Size([256, 512])
        # video    torch.Size([256, 512])
        # embedding   torch.Size([256, 300])

        phi_a = self.A_enc(audio)
        phi_v = self.V_enc(video)
        phi_a1 = self.SNNbranchaudio(audio)
        for t in range(1, self.T):
            phi_a1 += self.SNNbranchaudio(audio)
        phi_a1 = phi_a1/self.T

        phi_v1= self.SNNbranchvideo(video)
        for t in range(1, self.T):
            phi_v1 += self.SNNbranchaudio(video)
        phi_v1 = phi_v1/self.T

        audio = torch.unsqueeze(audio, dim=-1)
        phi_at = torch.unsqueeze(audio, dim=-1)
        video = torch.unsqueeze(video, dim=-1)
        phi_vt = torch.unsqueeze(video, dim=-1)

        # phi_at=audio.reshape((256, 512, 1, 1))
        phi_at = self.trl_a(phi_at)
        # phi_vt=video.reshape((256, 512, 1, 1))
        phi_vt = self.trl_v(phi_vt)

        input_concatenated_t = torch.stack((phi_at + self.pos_emb1D_t[0, :], phi_vt + self.pos_emb1D_t[1, :]), dim=1)

        phi_attn_t= self.cross_attention(input_concatenated_t)

        phi_at = phi_at + phi_attn_t[:, 0, :]
        phi_vt = phi_vt + phi_attn_t[:, 1, :]

        # phi_a = 0.5*phi_a + 0.5*phi_a*nn.functional.softmax(phi_a1)
        # phi_v = 0.5*phi_v + 0.5*phi_v*nn.functional.softmax(phi_v1)
        phi_input = torch.stack((phi_a + self.pos_emb1D[0, :], phi_a*nn.functional.softmax(phi_a1, dim=1) + self.pos_emb1D[0, :]), dim=1)
        phi_a= self.cross_attention(phi_input)[:, 0, :]
        # self.phi_v = 0.5*self.phi_v + 0.5*self.phi_v*nn.functional.softmax(self.phi_v1)
        phi_vinput = torch.stack((phi_v + self.pos_emb1D[1, :], phi_v*nn.functional.softmax(phi_v1, dim=1) + self.pos_emb1D[1, :]), dim=1)
        phi_v = self.cross_attention(phi_vinput)[:, 1, :]
        functional.reset_net(self.SNNbranchvideo)
        functional.reset_net(self.SNNbranchaudio)
        theta_w=self.W_proj(embedding)

        input_concatenated=torch.stack((phi_a+self.pos_emb1D[0,:], phi_v+self.pos_emb1D[1,:]), dim=1)

        phi_attn= self.cross_attention(input_concatenated)

        phi_a = phi_a + phi_attn[:,0,:]
        phi_v = phi_v + phi_attn[:,1,:]

        phi_a=torch.stack((phi_a + self.pos_emb1D[0, :],phi_vt + self.pos_emb1D_t[0, :]),dim=1)
        phi_a = self.cross_attention(phi_a)[:, 0, :]
        phi_v=torch.stack((phi_v + self.pos_emb1D[1, :],phi_at + self.pos_emb1D_t[1, :]),dim=1)
        phi_v = self.cross_attention(phi_v)[:, 0, :]

        theta_v = self.V_proj(phi_v)
        theta_a = self.A_proj(phi_a)
        # functional.reset_net(self.A_proj)
        # functional.reset_net(self.V_proj)
        return theta_a, theta_v, theta_w


