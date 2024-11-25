import torch.nn as nn
import torch
import timm
import torch.nn.functional as F
from torchinfo import summary
from torch.nn import Softmax, Parameter


def convblock(in_channels,out_channels,kernel_size=3,stride=1,dilation=1,padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,dilation=dilation,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


class DecoderBlock(nn.Module):
    def __init__(self,
        in_channels,
        skip_channels,
        out_channels,):
        super().__init__()
        self.conv1 = nn.Sequential(
            convblock(in_channels=in_channels + skip_channels,out_channels=out_channels,kernel_size=1,padding=0),
            convblock(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
        )
        self.conv2 = convblock(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1)
    
    def forward(self,x,skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=rate, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class CAM_Module(nn.Module):
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x, fb):
        batch_size, chnnels, width, height = x.shape
        proj_query = fb.view(batch_size, chnnels, -1)
        proj_key = fb.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = fb.view(batch_size, chnnels, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, chnnels, height, width)

        return x + self.gamma * out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).to('cuda').repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class S_Module(nn.Module):
    def __init__(self, in_dim):
        super(S_Module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x) #D
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        
        self.pau_attention = concate
        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        
        return self.gamma * (out_H + out_W) + x


class FeedbackSpatialAttention(nn.Module):
    def __init__(self, in_channel,feedback=False) :
        super().__init__()
        self.x_att = S_Module(in_dim=in_channel)
        self.fb_att = S_Module(in_dim=in_channel)
        self.feedback = feedback
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
    
    def forward(self,x,fb=None):
        x_att = self.gamma*self.x_att(x)
        if fb!=None:
            fb_att = self.fb_att(fb)
            x_att = x_att + self.gamma2*fb_att
        
        output = x + x_att
        return output


class StageAttentionwCAM(nn.Module):
    def __init__(self,in_channel,out_channel,cbam=False):
        super().__init__()
        self.down = nn.MaxPool2d(kernel_size=2,stride=2)
        self.oneconv = convblock(in_channels=in_channel,out_channels=out_channel,kernel_size=1,padding=0)
        self.pam = FeedbackSpatialAttention(in_channel=out_channel)
        self.cam = CAM_Module()
    
    def forward(self,x,fb=None):
        if fb!=None:
            fb = self.down(fb)
            fb = self.oneconv(fb)
        out = self.pam(x,fb)
        out2 = self.cam(x,fb)
        return out+out2



class EfficientFeebackNetwork(nn.Module):
    def __init__(self, in_channels=3,num_class=3,feedback=False):
        super().__init__()
        self.encoder = timm.create_model(model_name='efficientnet_b0.ra_in1k',pretrained=True,features_only=True)
        channel_size = [24,40,112,320]
        skip_channel = [16,24,40,112]
        out_channel = [16,24,40,112]
        
        sa_input = [num_class,16,24,40,112]
        sa_out = [16,24,40,112,320]

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(channel_size, skip_channel, out_channel)
        ]
        self.decoder = nn.ModuleList(blocks)
        attblocks = [
            StageAttentionwCAM(in_channel=in_c,out_channel=out_c) for in_c,out_c in zip(sa_input,sa_out)
        ]
        self.attblocks = nn.ModuleList(attblocks)
        
        rates = [2, 4, 6, 8]
        self.aspp1 = ASPP_module(320, 100, rate=rates[0])
        self.aspp2 = ASPP_module(320, 100, rate=rates[1])
        self.aspp3 = ASPP_module(320, 100, rate=rates[2])
        self.aspp4 = ASPP_module(320, 100, rate=rates[3])
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(320, 100, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(100),
                                             nn.ReLU(),
                                             )
        self.ref_aspp = nn.Conv2d(500, 320, 1, bias=False)

        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True),
            nn.Conv2d(in_channels=16,out_channels=num_class,kernel_size=1)
        )
        self.feedback = feedback
        if feedback:
            strides = [2,4,8,16]
    
    def forward(self,x,fb=None):
        encoder = self.encoder(x)
        if fb!=None:
            for i in range(len(encoder)-1):
                if i==0:
                    encoder[i] = self.attblocks[i](encoder[i],fb)
                else:
                    encoder[i] = self.attblocks[i](encoder[i],encoder[i-1])
        aspp1 = self.aspp1(encoder[-1])
        aspp2 = self.aspp2(encoder[-1])
        aspp3 = self.aspp3(encoder[-1])
        aspp4 = self.aspp4(encoder[-1])
        aspp5 = self.global_avg_pool(encoder[-1])
        aspp5 = F.interpolate(aspp5, size=aspp4.size()[2:], mode='bilinear', align_corners=True)
        aspp_all = torch.cat([aspp1,aspp2,aspp3,aspp4,aspp5],dim=1)

        dec0 = self.ref_aspp(aspp_all)
        if fb!=None:
            dec0 = self.attblocks[-1](dec0,encoder[-2])
        for i in range(len(encoder)-2,-1,-1):
            dec0 = self.decoder[i](dec0,encoder[i])
        head = self.head(dec0)
        return head


if __name__=='__main__':
    model = EfficientFeebackNetwork(num_class=2)
    summary(model,((1,3,512,512),(1,2,512,512)))