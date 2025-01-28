
import os
import sys
import numpy as np
from torch import nn
import torch



class Model(nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        #output_dir
        self.output_dir = args.output_dir #default
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = nn.Parameter(in_m, requires_grad=False)
        self.input_std = nn.Parameter(in_s, requires_grad=False)
        self.output_mean = nn.Parameter(out_m, requires_grad=False)
        self.output_std = nn.Parameter(out_s, requires_grad=False)
        
        
        self.len_limit_unit = 16 # base on the configuration of network.

        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000


        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        # LFCC dim (base component)
        self.lfcc_dim = [20]
        self.lfcc_with_delta = True

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # no truncation
        self.v_truncate_lens = [None for x in self.frame_hops]


        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        self.v_emd_dim = -2

        # output classes
        self.v_out_class = 2

        #squeeze-ro-extication
        self.reduction = 2 
        self.m_se_pooling = []
        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        ## 
        ## 2nd part of the classifier
        self.m_angle_scales = nn.ModuleDict()

        #weight for hidden_states:
        self.weight_hidd = nn.Parameter(torch.zeros(30, device='cuda'))
        self.hidden_features_dim = hidd_dims[ssl_model] #768

        self.extracter = getattr(hub, ssl_model)(ssl_ckpt)

        self.ssl_finetune = args.ssl_finetune
        self.multi_scale_active = args.multi_scale_active
        self.multi_branch_fix = args.multi_branch_fix

        self.data_type = args.data_type

        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            self.m_transform.append(
                    MaxPool1dLin_gmlp_scales(num_scale = len(Frame_shifts), feature_F_dim = self.hidden_features_dim, 
                        emb_dim=self.v_emd_dim, seq_len=2001,
                        gmlp_layers=5, batch_first=True, flag_pool='ap')
            )

            if(self.v_emd_dim > 0):
                for fs_i in Frame_shifts:
                    self.m_angle_scales[f"{fs_i}"]= nii_p2sgrad.P2SActivationLayer(self.v_emd_dim, self.v_out_class)
            elif(self.v_emd_dim < 0):
                for i, fs_i in enumerate(Frame_shifts):  # i here indicate how many downsample will happen
                    self.m_angle_scales[f"{fs_i}"]= nii_p2sgrad.P2SActivationLayer(
                            int(self.hidden_features_dim/pow(abs(self.v_emd_dim),i+1)), self.v_out_class)

        self.m_debugger_segscore_ali = []
        for i in Frame_shifts:
            self.m_debugger_segscore_ali.append(nii_debug.data_probe())

        self.m_transform = nn.ModuleList(self.m_transform)
        return
