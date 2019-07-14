from .ops import *


class Generator(nn.Module):
    def __init__(self, first_dim=32, isCheck=False, device=None):
        super(Generator, self).__init__()
        self.isCheck = isCheck
        self.device = device
        self.stage_1 = CoarseNet(5, first_dim, device=device)
        self.stage_2 = RefinementNet(5, first_dim, device=device)

    def forward(self, masked_img, mask, small_mask): # mask : 1 x 1 x H x W
        # border, maybe
        mask = mask.expand(masked_img.size(0),1,masked_img.size(2),masked_img.size(3))
        small_mask = small_mask.expand(masked_img.size(0), 1, masked_img.size(2) // 8, masked_img.size(3) // 8)
        if self.device:
            ones = to_var(torch.ones(mask.size()), device=self.device)
        else:
            ones = to_var(torch.ones(mask.size()))
        # stage1
        stage1_input = torch.cat([masked_img, ones, ones*mask], dim=1)
        stage1_output, resized_mask = self.stage_1(stage1_input, mask)
        # stage2
        new_masked_img = stage1_output*mask.clone() + masked_img.clone()*(1.-mask.clone())
        stage2_input = torch.cat([new_masked_img, ones.clone(), ones.clone()*mask.clone()], dim=1)
        stage2_output, offset_flow = self.stage_2(stage2_input, small_mask)

        return stage1_output, stage2_output, offset_flow


class CoarseNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''
    def __init__(self, in_ch, out_ch, device=None):
        super(CoarseNet,self).__init__()
        self.down = Down_Module(in_ch, out_ch)
        self.atrous = Dilation_Module(out_ch*4, out_ch*4)
        self.up = Up_Module(out_ch*4, 3)
        self.device=device

    def forward(self, x, mask):
        x = self.down(x)
        resized_mask = down_sample(mask, scale_factor=0.25, mode='nearest', device=self.device)
        x = self.atrous(x)
        x = self.up(x)

        return x, resized_mask


class RefinementNet(nn.Module):
    '''
    # input: B x 5 x W x H
    # after down: B x 128(32*4) x W/4 x H/4
    # after atrous: same with the output size of the down module
    # after up : same with the input size
    '''
    def __init__(self, in_ch, out_ch, device=None):
        super(RefinementNet,self).__init__()
        self.down_conv_branch = Down_Module(in_ch, out_ch, isRefine=True)
        self.down_attn_branch = Down_Module(in_ch, out_ch, activation=nn.ReLU(), isRefine=True, isAttn=True)
        self.atrous = Dilation_Module(out_ch*4, out_ch*4)
        self.CAttn = Contextual_Attention_Module(out_ch*4, out_ch*4, device=device)
        self.up = Up_Module(out_ch*8, 3, isRefine=True)

    def forward(self, x, resized_mask):
        # conv branch
        conv_x = self.down_conv_branch(x)
        conv_x = self.atrous(conv_x)

        # attention branch
        attn_x = self.down_attn_branch(x)

        attn_x, offset_flow = self.CAttn(attn_x, attn_x, mask=resized_mask)

        # concat two branches
        deconv_x = torch.cat([conv_x, attn_x], dim=1) # deconv_x => B x 256 x W/4 x H/4
        x = self.up(deconv_x)

        return x, offset_flow
