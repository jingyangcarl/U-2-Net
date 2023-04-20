import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms
from skimage import io, transform
from PIL import Image
import imageio
imageio.plugins.freeimage.download()

import numpy as np
import glob
import os

from data_loader import *
from model import *

from tqdm import tqdm
from transformers import CLIPTextModel, AutoTokenizer
# ------- 1. define loss function --------

mse_loss = nn.MSELoss(size_average=True)
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, albedos, speculs):

    loss0 = mse_loss(d0[:,:3,...],albedos)
    loss1 = mse_loss(d1[:,:3,...],albedos)
    loss2 = mse_loss(d2[:,:3,...],albedos)
    loss3 = mse_loss(d3[:,:3,...],albedos)
    loss4 = mse_loss(d4[:,:3,...],albedos)
    loss5 = mse_loss(d5[:,:3,...],albedos)
    loss6 = mse_loss(d6[:,:3,...],albedos)
 
    loss0_ = bce_loss(d0[:,-1:,...],speculs)
    loss1_ = bce_loss(d1[:,-1:,...],speculs)
    loss2_ = bce_loss(d2[:,-1:,...],speculs)
    loss3_ = bce_loss(d3[:,-1:,...],speculs)
    loss4_ = bce_loss(d4[:,-1:,...],speculs)
    loss5_ = bce_loss(d5[:,-1:,...],speculs)
    loss6_ = bce_loss(d6[:,-1:,...],speculs)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    loss_ = loss0_ + loss1_ + loss2_ + loss3_ + loss4_ + loss5_ + loss6_

    return loss0, loss+loss_, {
        'l0': loss0+loss0_,
        'l1': loss1+loss1_,
        'l2': loss2+loss2_,
        'l3': loss3+loss3_,
        'l4': loss4+loss4_,
        'l5': loss5+loss5_,
        'l6': loss6+loss6_,
    }

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn



# ------- 2. set the directory of training dataset --------

expname = 'test_models_28_normalalbedo2albedospec_masked'
# expname = 'debug'
model_name = 'u2net' #'u2netp'

exp_dir = os.path.join(os.getcwd(), 'logs', expname)
os.makedirs(exp_dir, exist_ok=True)
ngpus = torch.cuda.device_count()

epoch_num = 100000
batch_size_train = 16*ngpus
batch_size_val = 1
train_num = 0
val_num = 0

# train_dir = '/home/ICT2000/jyang/projects/ObjectReal/data/v0.4/pine_env_0/brdf/*/'
train_dir = '/home/ICT2000/jyang/projects/ObjectReal/data/v0/models_env_0/brdf/*/'
tra_normal_dir = os.path.join('normal')
tra_albedo_dir = os.path.join('albedo')
tra_specul_dir = os.path.join('specular')
# get all abs paths
tra_normal_paths = glob.glob(os.path.join(train_dir, tra_normal_dir, '*.exr'))
tra_albedo_paths = glob.glob(os.path.join(train_dir, tra_albedo_dir, '*.exr'))
tra_specul_paths = glob.glob(os.path.join(train_dir, tra_specul_dir, '*.exr'))
tra_mask_paths = [i.replace('normal', 'mask').replace('exr', 'png') for i in tra_normal_paths]
# filter by views
getcamid = lambda x: int(os.path.splitext(x)[0].split('/')[-1].replace('cam', ''))
tra_normal_paths = [i for i in tra_normal_paths if (getcamid(i) <= 16 and getcamid(i) != 13)][:25] # filter cams
tra_albedo_paths = [i for i in tra_albedo_paths if (getcamid(i) <= 16 and getcamid(i) != 13)][:25] # filter cams
tra_specul_paths = [i for i in tra_specul_paths if (getcamid(i) <= 16 and getcamid(i) != 13)][:25] # filter cams
tra_mask_paths = [i for i in tra_mask_paths if (getcamid(i) <= 16 and getcamid(i) != 13)][:25] # filter cams
tra_labels = [os.path.splitext(i)[0].split('/')[-3] for i in tra_normal_paths]
# tokenize text description
tokenizer = AutoTokenizer.from_pretrained(
    'runwayml/stable-diffusion-v1-5',
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tra_tokens = tokenizer(
    tra_labels, 
    max_length=tokenizer.model_max_length, 
    padding="max_length", 
    truncation=True, 
    return_tensors="pt"
)['input_ids'].numpy()

# test
test_normal_paths = tra_normal_paths
test_tokens = tra_labels

print("---")
print("train images: ", len(tra_normal_paths))
print("train labels: ", len(tra_albedo_paths))
print("---")

salobj_dataset = LightStageDataset(
    normal_paths=tra_normal_paths,
    albedo_paths=tra_albedo_paths,
    specul_paths=tra_specul_paths,
    mask_paths=tra_mask_paths,
    tokens = tra_tokens,
    transform=transforms.Compose([
        TextureRandomCrop(288),
        TextureNormalize(),
    ]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16)

test_salobj_dataset = LightStageDataset(
    normal_paths = tra_normal_paths[::7],
    albedo_paths = [],
    specul_paths= [],
    mask_paths=[],
    tokens = tra_tokens[::7],
    transform=transforms.Compose([
        TextureNormalize(half=False),
    ]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=3)

# ------- 3. define model --------
if(model_name=='u2net'):
    net = U2NET(6,4)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.cuda()
    
    # https://discuss.pytorch.org/t/do-dataparallel-and-distributeddataparallel-affect-the-batch-size-and-gpu-memory-consumption/97194
    net = nn.DataParallel(net, device_ids=list(range(ngpus)), dim=0)
    

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 20000 # save the model every 2000 iterations
val_frq = 2000

for epoch in range(0, epoch_num):
    net.train()

    pbar = tqdm(salobj_dataloader, dynamic_ncols=True)
    for i, data in enumerate(pbar):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        normals, albedos, speculs, tokens = data['normal'], data['albedo'], data['specul'], data['token']

        normals = normals.type(torch.FloatTensor)
        albedos = albedos.type(torch.FloatTensor)
        speculs = speculs.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            normals = Variable(normals.cuda(), requires_grad=False)
            albedos = Variable(albedos.cuda(), requires_grad=False)
            speculs = Variable(speculs.cuda(), requires_grad=False)
            tokens = Variable(tokens.cuda(), requires_grad=False)
        else:
            normals = Variable(normals, requires_grad=False)
            albedos = Variable(albedos, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()
        
        # hack
        inputs = torch.cat((normals, albedos**(1/2.2)), dim=1)
        
        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs, tokens)
        loss2, loss, descs = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, albedos, speculs)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        pbar.set_description(f'[epc: {epoch + 1}/{epoch_num}, bch: {(i + 1) * batch_size_train}/{len(salobj_dataset)}, ite: {ite_num}] ' + 
                             f'loss: {running_loss / ite_num4val:.4f}, tar: {running_tar_loss / ite_num4val:.4f} ' + 
                             f'l0: {descs["l0"]:.4f}, l1: {descs["l1"]:.4f}, l2: {descs["l2"]:.4f}, l3: {descs["l3"]:.4f}, l4: {descs["l4"]:.4f}, l5: {descs["l5"]:.4f}, l6: {descs["l6"]:.4f}')

        if ite_num % val_frq == 0:
            
            # evaluate model
            net.eval()
            with torch.no_grad():
                for i_test, data_test in enumerate(tqdm(test_salobj_dataloader, desc='testing', dynamic_ncols=True)):

                    normals_test, albedos_test, tokens_test, npath_test = data_test['normal'], data_test['albedo'], data_test['token'], data_test['npath']
                    normals_test = normals_test.type(torch.FloatTensor)

                    if torch.cuda.is_available():
                        normals_test = Variable(normals_test.cuda())
                        albedos_test = Variable(albedos_test.cuda())
                        tokens_test = Variable(tokens_test.cuda())
                    else:
                        normals_test = Variable(normals_test)

                    inputs_test = torch.cat((normals_test, albedos_test**(1/2.2)), dim=1)
                        
                    d1,d2,d3,d4,d5,d6,d7= net(inputs_test, tokens_test)

                    # normalization
                    pred = d1[0,...] # batch size is 1
                    pred = normPRED(pred)
                    pred = pred.cpu().data.numpy().transpose((1, 2, 0))
                    
                    # save results to test_results folder
                    obj, _, cam = os.path.splitext(npath_test[0])[0].split('/')[-3:]
                    save_path_albedo = os.path.join(exp_dir, 'val', f'{ite_num}', f'{obj}_{cam}_a.png')
                    save_path_specul = os.path.join(exp_dir, 'val', f'{ite_num}', f'{obj}_{cam}_s.png')
                    os.makedirs(os.path.dirname(save_path_albedo), exist_ok=True)
                    imageio.imwrite(save_path_albedo, (pred[...,:3] * 255.).astype(np.uint8))
                    imageio.imwrite(save_path_specul, (pred[...,-1] * 255.).astype(np.uint8))

            # back to train
            net.train()

        if ite_num % save_frq == 0:

            save_path = os.path.join(exp_dir, model_name, "bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(net.state_dict(), save_path)
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()  # resume train
            ite_num4val = 0

