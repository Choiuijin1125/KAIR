from models.model_plain import ModelPlain

from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler
from torch.optim import Adam

from models.select_network import define_G
from models.model_base import ModelBase
from models.loss import CharbonnierLoss
from models.loss_ssim import SSIMLoss

from utils.utils_model import test_mode
from utils.utils_regularizers import regularizer_orth, regularizer_clip


class ModelPlain8(ModelPlain):
    """Train with two inputs (L, C) and with pixel loss"""

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data, need_H=True, need_aux=True):
        self.L = data['L'].to(self.device)
        
        
        if need_H:
            self.H = data['H'].to(self.device)
        
        if need_aux:
            self.L_aux = data['L_aux'].to(self.device)
        
        

    # ----------------------------------------
    # feed (L, C) to netG and get E
    # ----------------------------------------
    def netG_forward(self):
        self.E, self.E_aux = self.netG(self.L)
        self.H_hf = self.H - self.gaussian_filter(self.H) 
        self.E_hf = self.E - self.gaussian_filter(self.E)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        G_loss = self.G_lossfn_weight * self.G_lossfn(self.E, self.H)
        G_loss += self.G_lossfn_weight * self.G_lossfn(self.E_aux, self.L_aux)
        G_loss += self.G_lossfn_weight * self.G_lossfn(self.E_hf, self.H_hf)
        
        G_loss.backward()

        # ------------------------------------
        # clip_grad
        # ------------------------------------
        # `clip_grad_norm` helps prevent the exploding gradient problem.
        G_optimizer_clipgrad = self.opt_train['G_optimizer_clipgrad'] if self.opt_train['G_optimizer_clipgrad'] else 0
        if G_optimizer_clipgrad > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.opt_train['G_optimizer_clipgrad'], norm_type=2)

        self.G_optimizer.step()

        # ------------------------------------
        # regularizer
        # ------------------------------------
        G_regularizer_orthstep = self.opt_train['G_regularizer_orthstep'] if self.opt_train['G_regularizer_orthstep'] else 0
        if G_regularizer_orthstep > 0 and current_step % G_regularizer_orthstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_orth)
        G_regularizer_clipstep = self.opt_train['G_regularizer_clipstep'] if self.opt_train['G_regularizer_clipstep'] else 0
        if G_regularizer_clipstep > 0 and current_step % G_regularizer_clipstep == 0 and current_step % self.opt['train']['checkpoint_save'] != 0:
            self.netG.apply(regularizer_clip)

        # self.log_dict['G_loss'] = G_loss.item()/self.E.size()[0]  # if `reduction='sum'`
        self.log_dict['G_loss'] = G_loss.item()

        if self.opt_train['E_decay'] > 0:
            self.update_E(self.opt_train['E_decay'])

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # test / inference x8
    # ----------------------------------------
    def testx8(self, aux=False):
        self.netG.eval()
        with torch.no_grad():
            self.E = test_mode(self.netG, self.L, mode=3, sf=self.opt['scale'], modulo=1, aux=aux)
        self.netG.train()
        

