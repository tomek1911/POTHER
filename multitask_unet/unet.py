import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

# U-Net: Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham. https://doi.org/10.1007/978-3-319-24574-4_28
# Inception: Szegedy, C., Wei Liu, Yangqing Jia, Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A.: Going deeper with convolutions. In: 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1–9. IEEE, Boston, MA, USA (2015). https://doi.org/10.1109/CVPR.2015.7298594.

def get_backbone(name, pretrained=True):

    # loading backbone model
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    # specifying skip feature and output names
    if name.startswith('resnet'):
        feature_names = [None, 'relu', 'layer1', 'layer2', 'layer3']
        backbone_output = 'layer4'
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))

    return backbone, feature_names, backbone_output

class PytorchMult(nn.Module):
    def __init__(self):
        super(PytorchMult, self).__init__()

    def forward(self, skip_features, psi):
        return skip_features * psi

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)
        self.att_out = PytorchMult()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        g_signal = self.att_out(x,psi)
        return g_signal


class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_block_full(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_block_full, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.25)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
      		nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UpsampleBlock(nn.Module):

    def __init__(self, ch_in, ch_out=None, skip_in=0, use_bn=True, attention=False):
        super(UpsampleBlock, self).__init__()

        self.attention = attention
        ch_out = ch_in//2 if ch_out is None else ch_out

        if attention:
            if skip_in != 0:
                self.up = Up_conv(ch_in, ch_out)
                self.att = Attention_block(ch_out, skip_in, ch_out//2)
            else:
                self.up = Up_conv(ch_in, ch_out)
        else:
            self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                         stride=2, padding=1, output_padding=0, bias=(not use_bn))
            self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        if attention:
            if skip_in != 0:
                conv2_in = ch_out + skip_in
            else:
                conv2_in = ch_out
        else:
            conv2_in = ch_out + skip_in

        self.conv = Conv_block(conv2_in, ch_out)

    def forward(self, x, skip_f=None):
        x_a = None
        if self.attention:
            if skip_f is not None:
                x_g = self.up(x)
                x_a = self.att(x_g, skip_f)
            else:
                x = self.up(x)
        else:
            x = self.up(x)
            x = self.bn1(x) if self.bn1 is not None else x
            x = self.relu(x)

        if skip_f is not None:
            if self.attention:
                x = torch.cat([x_a, x_g], dim=1)
            else:
                x = torch.cat([x, skip_f], dim=1)

        x = self.conv(x)

        return x, x_a


class InceptionA(nn.Module):
    def __init__(self, in_channels, base = 8):
        super().__init__()
    
        conv_block = Conv_block

        out = int(in_channels/base)
        out_conv = int(in_channels/base*2)

        self.branch1x1 = conv_block(in_channels, out, kernel_size=1, padding=0)

        self.branch5x5_1 = conv_block(in_channels, out, kernel_size=1, padding=0)
        self.branch5x5_2 = conv_block(out, out_conv, kernel_size=5, padding=2)

        self.branch3x3_1 = conv_block(in_channels, out, kernel_size=1, padding=0)
        self.branch3x3_2 = conv_block(out, out_conv, kernel_size=3, padding=1)
        self.branch3x3_3 = conv_block(out_conv, out_conv, kernel_size=3, padding=1)
        
        self.branch_pool = conv_block(in_channels, out, kernel_size=1, padding=0)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
    def __init__(self, in_channels, base = 8):
        super().__init__()

        out = int(in_channels/base)
        out_conv = int(in_channels/base*2)
        
        conv_block = Conv_block

        self.branch3x3_1 = conv_block(in_channels, out, kernel_size=1)
        self.branch3x3_2 = conv_block(out, out_conv, kernel_size=3, stride=2)

        self.branch5x5_1 = conv_block(in_channels, out, kernel_size=1, padding=0)
        self.branch5x5_2 = conv_block(out, out_conv, kernel_size=5, stride = 2, padding=3)

        self.branch7x7x3_1 = conv_block(in_channels, out, kernel_size=1)
        self.branch7x7x3_2 = conv_block(out, out, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(out, out, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(out, out, kernel_size=3, stride=2)

        self.branch_pool = conv_block(in_channels, out, kernel_size=1, padding=1)

    def _forward(self, x):

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)
        
        outputs = [branch3x3, branch5x5, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):
    def __init__(self, in_channels, base = 8):
        super().__init__()
        
        conv_block = Conv_block
        out = int(in_channels/base)
        out_conv = int(in_channels/base*2)

        self.branch1x1 = conv_block(in_channels, out, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, out, kernel_size=1)
        self.branch3x3_2a = conv_block(out, out, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(out, out, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, out_conv, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(out_conv, out, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(out, out, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(out, out, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class ClassificatonHead(nn.Module):
    def __init__(self, channels, features, num_classes, unet_params, task2 = False, exp_head = False, norm_features=False):
        super(ClassificatonHead, self).__init__()
        self.exp_head = exp_head
        self.norm_features= norm_features
        
        self.drop = nn.Dropout(p=0.5)
        self.avgpool = nn.AvgPool2d(features, None, 0)
        self.fc = nn.Linear(channels, num_classes)
        self.relu = nn.ReLU(inplace=True)

        if task2:
            #GAP
            self.gap = {}
            self.gap_e3 = nn.AvgPool2d(unet_params['encoder']['features'][-2], None, 0)
            self.gap_d3 = nn.AvgPool2d(16, None, 0)
            self.gap_d2 = nn.AvgPool2d(15, None, 0)
            self.gap_d1 = nn.AvgPool2d(unet_params['decoder']['features'][2], None, 0)
            self.gap_d0 = nn.AvgPool2d(unet_params['decoder']['features'][3], None, 0)
            
            #features reduction - conv1x1
            self.conv1x1_e4 = Conv_block(2048, 1024, kernel_size=1, padding=0)
            self.conv1x1_e3 = Conv_block(832, 512, kernel_size=1, padding=0)
            self.conv1x1_d3 = Conv_block(832, 512, kernel_size=1, padding=0)
            self.conv1x1_d2 = Conv_block(384, 256, kernel_size=1, padding=0)
            self.conv1x1_d1 = Conv_block(192, 128, kernel_size=1, padding=0)
            self.conv1x1_d0 = Conv_block(192, 128, kernel_size=1, padding=0)

            #inception feature extractors
            # self.inception_e4 = Inception(2048, 8)
            self.inception_e3 = InceptionE(1024, 8)
            self.inception_d3 = InceptionE(1024, 8)
            self.inception_d2 = InceptionD(512, 8)
            self.inception_d1 = InceptionA(256, 8)
            self.inception_d0 = InceptionA(128, 4)

            self.fc1 = nn.Linear(896, 256)
            self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x, dec_f=None, enc_f=None, ag_f=None):

        if not self.exp_head:
            #V1
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.drop(x)
            return self.fc(x)
            #V2
            # x = self.f_conv(x)
            # x = self.drop(x)
            # x = self.avgpool(x)
            # x = x.view(x.shape[0], -1)
            # return self.fc(x)

        features_list = []

        #d0 - - upsample_block[3]
        # f = self.inception_d0(dec_f['relu'])
        # f = self.conv1x1_d0(f)
        # f = self.gap_d0(f)
        # f = f.view(x.shape[0], -1)
        # if self.norm_features:
        #     f = nn.functional.normalize(f, p=2, dim=1)
        # features_list.append(f)

        #d1 - - upsample_block[2]
        f = self.inception_d1(dec_f['layer1'])
        f = self.conv1x1_d1(f)
        f = self.gap_d1(f)
        f = f.view(x.shape[0], -1)
        if self.norm_features:
            f = nn.functional.normalize(f, p=2, dim=1)
        features_list.append(f)

        #d2 - - upsample_block[1]
        f = self.inception_d2(dec_f['layer2'])
        f = self.conv1x1_d2(f)
        f = self.gap_d2(f)
        f = f.view(x.shape[0], -1)
        if self.norm_features:
            f = nn.functional.normalize(f, p=2, dim=1)
        features_list.append(f)

        #d3 - upsample_block[0]
        f = self.inception_d3(ag_f['layer3'])
        f = self.conv1x1_d3(f)
        f = self.gap_d3(f)
        f = f.view(x.shape[0], -1)
        if self.norm_features:
            f = nn.functional.normalize(f, p=2, dim=1)
        features_list.append(f)

        # #e3
        # f = self.inception_e3(enc_f['layer3'])
        # f = self.conv1x1_e3(f)
        # f = self.gap_e3(f)
        # f = f.view(x.shape[0], -1)
        # if self.norm_features:
        #     f = nn.functional.normalize(f, p=2, dim=1)
        # features_list.append(f)

        # #e4
        # f = self.conv1x1_e4(x)
        # f = self.avgpool(f)
        # f = f.view(x.shape[0], -1)
        # if self.norm_features:
        #     f = nn.functional.normalize(f, p=2, dim=1)
        # features_list.append(f)

        # norm_vector = [3.1444483475648335, 3.096837966552978, 2.1885827028441716, 1.5230322769520528, 1.5466716038328856, 1.0]
        norm_vector = [3.096837966552978, 2.1885827028441716, 1.5230322769520528] # d1, d2, d3
        features_list = [features / norm for (norm, features) in zip (norm_vector, features_list)]

        features_vector = torch.cat(features_list, dim=1)

        x = self.drop(features_vector)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

#######################################################################################################
##                                                 UNET                                              ##
#######################################################################################################
class Unet(nn.Module):

    def __init__(self,
                 backbone_name='resnet50',
                 pretrained=True,
                 encoder_freeze=False,
                 segmentation_classes=2,
                 classes_encoder=3,
                 decoder_filters=(256, 128, 64, 32, 16),
                 infer_tensor=torch.zeros(1, 3, 224, 224),
                 shortcut_features='default',
                 attention=False,
                 decoder_use_batchnorm=True,
                 full_seg_conv=False,
                 experimental_head=True,
                 norm_features=False,
                 tasks = {'T1':True}):
        super(Unet, self).__init__()

        self.tasks = tasks
        self.exp_head = experimental_head
        self.norm_features = norm_features

        # SETUP ENCODER BACKBONE
        self.backbone_name = backbone_name
        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(
            backbone_name, pretrained=pretrained)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.replaced_conv1 = True

        #CALCULATE FEATURES SIZES OF SKIP CONNECTIONS
        encoder_params, x, features = self.infer_skip_channels(infer_tensor)
        self.unet_params = {'encoder': encoder_params}
        bb_output_channels = self.unet_params['encoder']['channels_out'][-1]
        shortcut_chs =  self.unet_params['encoder']['channels_out'][:-1]

        if shortcut_features != 'default':
            self.shortcut_features = shortcut_features

        #BUILD DECODER I - segmentation task
        if tasks['T2']:
            self.upsample_blocks = nn.ModuleList()
            # avoid having more blocks than skip connections
            decoder_filters = decoder_filters[:len(self.shortcut_features)]
            decoder_filters_in = [bb_output_channels] + list(decoder_filters[:-1])
            num_blocks = len(self.shortcut_features)
            for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
                print('upsample_blocks[{}] in: {}   out: {}'.format(
                    i, filters_in, filters_out))
                self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                        skip_in=shortcut_chs[num_blocks-i-1],
                                                        use_bn=decoder_use_batchnorm,
                                                        attention=attention))           

        #BUILD DECODER II - reconstruction task    
        if tasks['T3']:
            self.upsample_blocks2 = nn.ModuleList()
            # avoid having more blocks than skip connections
            decoder_filters = decoder_filters[:len(self.shortcut_features)]
            decoder_filters_in = [bb_output_channels] + list(decoder_filters[:-1])
            num_blocks = len(self.shortcut_features)
            for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
                print('upsample_blocks[{}] in: {}   out: {}'.format(
                    i, filters_in, filters_out))
                self.upsample_blocks2.append(UpsampleBlock(filters_in, filters_out,
                                                        skip_in=shortcut_chs[num_blocks-i-1],
                                                        use_bn=decoder_use_batchnorm,
                                                        attention=True))

        # last decoder layer
        self.final_conv = nn.Conv2d(decoder_filters[-1], segmentation_classes, kernel_size=(1, 1))
        
        #CALCULATE FEATURES OUTPUT OF DECODER
        if tasks['T2'] or tasks['T3']:
            decoder_params = self.infer_decoder_channels(x, features)
            self.unet_params['decoder'] = decoder_params
        
        print('Unet parameters:')
        print(f" >encoder: {self.unet_params['encoder']}")
        if 'decoder' in self.unet_params:
            print(f" >decoder: {self.unet_params['decoder']}")
        #CLASSIFYING HEAD - ENCODER FEATURES
        if tasks['T1']:
            if self.exp_head:
                if tasks['T2'] and tasks['T3']:
                    in_channels = bb_output_channels + 2 * self.unet_params['decoder']['channels_out'][0]
                elif tasks['T2'] or tasks['T3']:
                    in_channels = bb_output_channels  #+ self.unet_params['decoder']['channels_out'][0]

                self.classification_head = ClassificatonHead(in_channels, self.unet_params['encoder']['features'][-1],
                                                        classes_encoder, self.unet_params, tasks['T2'], self.exp_head, self.norm_features)
            else:
                self.classification_head = ClassificatonHead(bb_output_channels, self.unet_params['encoder']['features'][-1],
                                                        classes_encoder, self.unet_params, tasks['T2'], self.exp_head, self.norm_features)

        if encoder_freeze:
            self.freeze_encoder()

        self.features = None

    def freeze_encoder(self):

        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, *input):

        x_class, x_seg, x_res = None, None, None
        #encoder forward progation 
        x, encoder_features = self.forward_backbone(*input)
        #decoder forward progation - segmentation
        if self.tasks['T2']:
            decoder_features = {}
            ag_features = {}
            x_seg = x
            for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
                skip_features = encoder_features[skip_name]
                x_seg, x_a = upsample_block(x_seg, skip_features)
                if skip_name in ['layer3', 'layer2', 'layer1']:
                    decoder_features[skip_name] = x_seg
                    ag_features[skip_name] = x_a
            x_seg = self.final_conv(x_seg)

        #decoder forward progation - restoration
        if self.tasks['T3']:
            decoder2_features = {}
            x_res = x
            for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks2):
                skip_features = encoder_features[skip_name]
                x_res = upsample_block(x_res, skip_features)
                if skip_name in ['layer3']:
                    decoder2_features[skip_name] = x_res
            x_res = self.final_conv(x_res)

        if self.tasks['T1'] and self.exp_head:
            assert self.tasks['T2'] or self.tasks['T3'], "Experimental head needs at least one of decoders' features!"
            x_class = self.classification_head(x, decoder_features, encoder_features, ag_features)
        else:
            x_class = self.classification_head(x)
        
        return x_class, x_seg, x_res

    def forward_backbone(self, x):

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break

        return x, features

    def infer_decoder_channels(self, x, features):

        decoder_info = {'layers_name': [], 'channels_out': [], 'features': []}
        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x, _ = upsample_block(x, skip_features)
            decoder_info['layers_name'].append(skip_name)
            decoder_info['channels_out'].append(x.shape[1])
            decoder_info['features'].append(x.shape[2])

        x = self.final_conv(x)

        decoder_info['layers_name'].append(skip_name)
        decoder_info['channels_out'].append(x.shape[1])
        decoder_info['features'].append(x.shape[2])
        return decoder_info

    def infer_skip_channels(self, infer_tensor = torch.zeros(1, 3, 224, 224)):

        x = infer_tensor
        has_fullres_features = self.backbone_name.startswith('vgg') or self.backbone_name == 'unet_encoder'
        channels = [] if has_fullres_features else [0]  # only VGG has features at full resolution
        encoder_info = {'layers_name': [], 'channels_out': [], 'features': []}
        features = {None: None} if None in self.shortcut_features else dict()
        
        #first resnet layer has no skip connection
        encoder_info['layers_name'].append('')
        encoder_info['channels_out'].append(0)
        encoder_info['features'].append(0)
        
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                encoder_info['layers_name'].append(name)
                encoder_info['channels_out'].append(x.shape[1])
                encoder_info['features'].append(x.shape[2])
                features[name] = x
            elif name == self.bb_out_name:
                encoder_info['layers_name'].append(name)
                encoder_info['channels_out'].append(x.shape[1])
                encoder_info['features'].append(x.shape[2])
                break

        return encoder_info, x, features

    def get_pretrained_parameters(self):
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                yield param

    def get_random_initialized_parameters(self):
        pretrained_param_names = set()
        for name, param in self.backbone.named_parameters():
            if not (self.replaced_conv1 and name == 'conv1.weight'):
                pretrained_param_names.add('backbone.{}'.format(name))

        for name, param in self.named_parameters():
            if name not in pretrained_param_names:
                yield param
