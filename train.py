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

import numpy as np
import glob
import os

from data_loader import *
from model import *

from tqdm import tqdm

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss0, loss, {
        'l0': loss0,
        'l1': loss1,
        'l2': loss2,
        'l3': loss3,
        'l4': loss4,
        'l5': loss5,
        'l6': loss6,
    }

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name, pred, d_dir, step):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    # image = io.imread(image_name)
    # imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    save_path = os.path.join(d_dir, f'{step}', f'{imidx}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    im.save(save_path)

# ------- 2. set the directory of training dataset --------

expname = 'test_pine'
model_name = 'u2net' #'u2netp'

data_dir = '/home/ICT2000/jyang/projects/ObjectReal/data/v0.4/pine_env_0/brdf/pine_static/'
tra_image_dir = os.path.join('normal')
tra_label_dir = os.path.join('specular')
image_ext = 'exr'
label_ext = 'exr'

exp_dir = os.path.join(os.getcwd(), 'logs', expname)
os.makedirs(exp_dir, exist_ok=True)

epoch_num = 100000
batch_size_train = 16
batch_size_val = 1
train_num = 0
val_num = 0

tra_img_name_list = glob.glob(os.path.join(data_dir, tra_image_dir, f'*.{image_ext}'))
tra_lbl_name_list = glob.glob(os.path.join(data_dir, tra_label_dir, f'*.{image_ext}'))
# test_img_name_list = glob.glob(os.path.join(os.getcwd(), 'test_data', 'test_images', f'*'))
test_img_name_list = tra_img_name_list

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

salobj_dataset = LightStageDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        # RescaleT(320),
        RandomCrop(288),
        SpecularNormalNormalize(),
        # ToTensorLab(flag=0)
    ]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=16)

test_salobj_dataset = LightStageDataset(
    img_name_list = tra_img_name_list,
    lbl_name_list = [],
    transform=transforms.Compose([
        # RescaleT(320),
        # RandomCrop(288),
        SpecularNormalNormalize(True),
        # ToTensorLab(flag=0)
    ]))
test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=16)

# ------- 3. define model --------
# define the net
if(model_name=='u2net'):
    net = U2NET(3, 1)
elif(model_name=='u2netp'):
    net = U2NETP(3,1)

if torch.cuda.is_available():
    net.cuda()

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
val_frq = 1000

for epoch in range(0, epoch_num):
    net.train()

    pbar = tqdm(salobj_dataloader, dynamic_ncols=True)
    for i, data in enumerate(pbar):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss, descs = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()

        pbar.set_description(f'[eposh: {epoch + 1}/{epoch_num}, batch: {(i + 1) * batch_size_train}/{len(salobj_dataset)}, ite: {ite_num}] ' + 
                             f'loss: {running_loss / ite_num4val:.4f}, tar: {running_tar_loss / ite_num4val:.4f} ' + 
                             f'l0: {descs["l0"]:.4f}, l1: {descs["l1"]:.4f}, l2: {descs["l2"]:.4f}, l3: {descs["l3"]:.4f}, l4: {descs["l4"]:.4f}, l5: {descs["l5"]:.4f}, l6: {descs["l6"]:.4f}')

        if ite_num % val_frq == 0:
            
            # evaluate model
            net.eval()
            with torch.no_grad():
                for i_test, data_test in enumerate(test_salobj_dataloader):
                    if i_test >= 3: continue
                    print("inferencing:",test_img_name_list[i_test].split(os.sep)[-1])

                    inputs_test = data_test['image']
                    inputs_test = inputs_test.type(torch.FloatTensor)

                    if torch.cuda.is_available():
                        inputs_test = Variable(inputs_test.cuda())
                    else:
                        inputs_test = Variable(inputs_test)

                    d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

                    # normalization
                    pred = d1[:,0,:,:]
                    pred = normPRED(pred)
                    
                    # save results to test_results folder
                    prediction_dir = os.path.join(exp_dir, 'val') + os.sep
                    os.makedirs(prediction_dir, exist_ok=True)
                    save_output(test_img_name_list[i_test], pred, prediction_dir, ite_num)

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

