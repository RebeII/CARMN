import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import os, sys
from config import *
import matplotlib.image as mpimg
import gensim
import numpy as np
import re
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import pickle
import time


'''class mul_dataloader(data.Dataset):
    def __init__(self, text_dir, img_dir, label_dir):
        self.text_dir = text_dir
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.length = len(os.listdir(text_dir))
        self.img_totensor = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
    

    def __getitem__(self, i):
        index = i
        text_path = os.path.join(self.text_dir, str(index+1))
        img_path = os.path.join(self.img_dir, str(index+1)+".jpg")
        label_path = os.path.join(self.label_dir, str(index+1)+".txt")

        text_file = open(text_path, 'rb')
        text = pickle.load(text_file)
        text_file.close()
        text = torch.from_numpy(text)
        text = text.float()

        img = Image.open(img_path)
        img = self.img_totensor(img)

        label_file = open(label_path, 'r')
        label = label_file.read()
        label_file.close()
        label = int(label)

        return text, img, label
    

    def __len__(self):
        return self.length'''

class mul_dataloader(data.Dataset):
    def __init__(self, dir):
        self.dir = dir
    

    def __getitem__(self, i):
        index = i

        text_file = open(self.dir, 'rb')
        file = pickle.load(text_file)
        text_file.close()
        text = file["text_tensor"][i]
        text = text.type(torch.FloatTensor)
        #text = text.float()

        img = file["image"][i]
        label = file["label"][i]

        return text, img, label
    

    def __len__(self):
        text_file = open(self.dir, 'rb')
        file = pickle.load(text_file)
        text_file.close()
        return len(file["label"])


class CARMN(nn.Module):
    def __init__(self):
        super(CARMN, self).__init__()

        vgg_model_1000 = models.vgg19(pretrained=True)
        new_classifier = torch.nn.Sequential(*list(vgg_model_1000.children())[-1][:7])
        vgg_model_1000.classifier = new_classifier
        for i in vgg_model_1000.parameters():
            i.requires_grad=False
        self.vgg_encoding = vgg_model_1000
        self.img_linear = nn.Linear(1000, 300, bias=True)
        self.relu = nn.LeakyReLU()

        self.ln1 = nn.LayerNorm([291,300], elementwise_affine=True)
        self.ln2 = nn.LayerNorm([1,300], elementwise_affine=True)

        self.toqueries_T_P = nn.Linear(300, 300, bias=False)
        self.tokeys_T_P = nn.Linear(300, 300, bias=False)
        self.tovalues_T_P = nn.Linear(300, 300, bias=False)

        self.toqueries_P_T = nn.Linear(300, 300, bias=False)
        self.tokeys_P_T = nn.Linear(300, 300, bias=False)
        self.tovalues_P_T = nn.Linear(300, 300, bias=False)

        self.heads = 10
        self.toqueries_1 = nn.Linear(300, 300, bias=False)
        self.tokeys_1 = nn.Linear(300, 300, bias=False)
        self.tovalues_1 = nn.Linear(300, 300, bias=False)
        self.mul_1 = nn.Linear(300, 300, bias=False)
        self.toqueries_2 = nn.Linear(300, 300, bias=False)
        self.tokeys_2 = nn.Linear(300, 300, bias=False)
        self.tovalues_2 = nn.Linear(300, 300, bias=False)
        self.mul_2 = nn.Linear(300, 300, bias=False)
        self.toqueries_3 = nn.Linear(300, 300, bias=False)
        self.tokeys_3 = nn.Linear(300, 300, bias=False)
        self.tovalues_3 = nn.Linear(300, 300, bias=False)
        self.mul_3 = nn.Linear(300, 300, bias=False)

        self.conv1 = nn.Conv2d(2,25,(1,300))
        self.conv2 = nn.Conv2d(2,25,(2,300))
        self.conv3 = nn.Conv2d(2,25,(3,300))
        self.conv4 = nn.Conv2d(2,25,(4,300))

        self.maxpool1 = nn.MaxPool1d(291)
        self.maxpool2 = nn.MaxPool1d(290)
        self.maxpool3 = nn.MaxPool1d(289)
        self.maxpool4 = nn.MaxPool1d(288)

        self.linear = nn.Linear(100, 300)
        self.last_linear = nn.Linear(600, 2)



    def forward(self, x1, x2):
        x2 = self.vgg_encoding(x2)
        x2.unsqueeze_(dim=1)
        x2 = self.img_linear(x2)
        x2 = self.relu(x2)

        ln_R_P = self.ln2(x2)
        ln_R_T = self.ln1(x1)

        queries_T_P = self.toqueries_T_P(ln_R_P)
        keys_T_P = self.tokeys_T_P(ln_R_T)
        values_T_P = self.tovalues_T_P(ln_R_T)
        att_T_P  = torch.bmm(queries_T_P, keys_T_P.transpose(1,2))
        att_T_P = F.softmax(att_T_P, dim=2)
        att_T_P = torch.bmm(att_T_P, values_T_P)
        R_T_P = att_T_P + ln_R_P

        queries_P_T = self.toqueries_P_T(ln_R_T)
        keys_P_T = self.tokeys_P_T(ln_R_P)
        values_P_T = self.tovalues_P_T(ln_R_P)
        att_P_T = torch.bmm(queries_P_T, keys_P_T.transpose(1,2))
        att_P_T = F.softmax(att_P_T, dim=2)
        att_P_T = torch.bmm(att_P_T, values_P_T)
        R_P_T = att_P_T + ln_R_T

        b1, t1, k1 = ln_R_T.size()
        h = self.heads
        k1 = k1//h
        queries_T_T_1 = self.toqueries_1(ln_R_T).view(b1, t1, h, k1)
        keys_T_T_1 = self.tokeys_1(ln_R_T).view(b1, t1, h, k1)
        values_T_T_1 = self.tovalues_1(ln_R_T).view(b1, t1, h, k1)
        queries_T_T_1 = queries_T_T_1.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        keys_T_T_1 = keys_T_T_1.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        values_T_T_1 = values_T_T_1.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        queries_T_T_1 = queries_T_T_1 / (h ** (1/4))
        keys_T_T_1 = keys_T_T_1 / (h ** (1/4))
        att_T_T_1 = torch.bmm(queries_T_T_1, keys_T_T_1.transpose(1,2))
        att_T_T_1 = F.softmax(att_T_T_1, dim=2)
        att_T_T_1 = torch.bmm(att_T_T_1, values_T_T_1)
        att_T_T_1 = att_T_T_1.transpose(1, 2).contiguous().view(b1, t1, h*k1)
        att_T_T_1 = self.mul_1(att_T_T_1)

        queries_T_T_2 = self.toqueries_2(att_T_T_1).view(b1, t1, h, k1)
        keys_T_T_2 = self.tokeys_2(att_T_T_1).view(b1, t1, h, k1)
        values_T_T_2 = self.tovalues_2(att_T_T_1).view(b1, t1, h, k1)
        queries_T_T_2 = queries_T_T_2.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        keys_T_T_2 = keys_T_T_2.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        values_T_T_2 = values_T_T_2.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        queries_T_T_2 = queries_T_T_2 / (h ** (1/4))
        keys_T_T_2 = keys_T_T_2 / (h ** (1/4))
        att_T_T_2 = torch.bmm(queries_T_T_2, keys_T_T_2.transpose(1,2))
        att_T_T_2 = F.softmax(att_T_T_2, dim=2)
        att_T_T_2 = torch.bmm(att_T_T_2, values_T_T_2)
        att_T_T_2 = att_T_T_2.transpose(1, 2).contiguous().view(b1, t1, h*k1)
        att_T_T_2 = self.mul_2(att_T_T_2)

        queries_T_T_3 = self.toqueries_3(att_T_T_2).view(b1, t1, h, k1)
        keys_T_T_3 = self.tokeys_3(att_T_T_2).view(b1, t1, h, k1)
        values_T_T_3 = self.tovalues_3(att_T_T_2).view(b1, t1, h, k1)
        queries_T_T_3 = queries_T_T_3.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        keys_T_T_3 = keys_T_T_3.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        values_T_T_3 = values_T_T_3.transpose(1, 2).contiguous().view(b1 * h, t1, k1)
        queries_T_T_3 = queries_T_T_3 / (h ** (1/4))
        keys_T_T_3 = keys_T_T_3 / (h ** (1/4))
        att_T_T_3 = torch.bmm(queries_T_T_3, keys_T_T_3.transpose(1,2))
        att_T_T_3 = F.softmax(att_T_T_3, dim=2)
        att_T_T_3 = torch.bmm(att_T_T_3, values_T_T_3)
        att_T_T_3 = att_T_T_3.transpose(1, 2).contiguous().view(b1, t1, h*k1)
        att_T_T_3 = self.mul_3(att_T_T_3)

        R = torch.stack((R_P_T, att_T_T_3), dim=1)
        r1 = self.conv1(R)
        r1 = self.relu(r1)
        r1.squeeze_(dim=3)
        r1 = self.maxpool1(r1)
        r1.squeeze_(dim=2)
        r2 = self.conv2(R)
        r2 = self.relu(r2)
        r2.squeeze_(dim=3)
        r2 = self.maxpool2(r2)
        r2.squeeze_(dim=2)
        r3 = self.conv3(R)
        r3 = self.relu(r3)
        r3.squeeze_(dim=3)
        r3 = self.maxpool3(r3)
        r3.squeeze_(dim=2)
        r4 = self.conv4(R)
        r4 = self.relu(r4)
        r4.squeeze_(dim=3)
        r4 = self.maxpool4(r4)
        r4.squeeze_(dim=2)
        total_r = torch.cat([r1,r2,r3,r4], dim=1)
        total_r = self.linear(total_r)
        total_r = self.relu(total_r)
        R_T_P.squeeze_(dim=1)
        Y = torch.cat((R_T_P, total_r), dim=1)
        Y = self.last_linear(Y)
        return Y



if __name__ == '__main__':
    train_dataset = mul_dataloader(train_dir)
    valiate_dataset = mul_dataloader(valiate_dir)
    test_dataset = mul_dataloader(test_dir)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCHSIZE, shuffle=False)
    valiate_loader = data.DataLoader(dataset=valiate_dataset, batch_size=BATCHSIZE, shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCHSIZE, shuffle=False)
    model = CARMN()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(device)
    for e in range(epoch):
        f = open("./result.txt", "a")
        f.write("epoch: " + str(e+1) + "\n")
        f.close()
        model.train()
        for i, (text,img,label) in enumerate(train_loader):
            epoch_start = time.time()
            model = model.to(device)
            text = text.to(device)
            img = img.to(device)
            label = label.to(device)
            out = model(text, img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if i%20==0:
            #f.write('epoch: {}, batch: {}, loss: {}'.format(e + 1, i + 1, loss.data))
            #print('epoch: {}, batch: {}, loss: {}'.format(e + 1, i + 1, loss.data))
        torch.save(model, './trained_model/'+str(e+1)+'.pth')
        
        model.eval()    # turn the model to test pattern, do some as dropout, batchNormalization
        
        correct = 0
        total = 0
        for (text,img,label) in valiate_loader:
            model = model.to(device)
            text = text.to(device)
            img = img.to(device)
            label = label.to(device)
            out = model(text, img)
            _, pre = torch.max(out.data, 1)
            #_, pred = torch.max(out, 1)     # 返回每一行中最大值和对应的索引
            total += label.size(0)
            correct += (pre == label).sum().item()
        f = open("./result.txt", "a")
        f.write('Valiate Accuracy: '+ str(correct / total) + "\n")
        f.close()
        #print('Test Accuracy: {}'.format(correct / total))


        
        correct = 0
        total = 0
        for (text,img,label) in test_loader:
            model = model.to(device)
            text = text.to(device)
            img = img.to(device)
            label = label.to(device)
            out = model(text, img)
            _, pre = torch.max(out.data, 1)
            #_, pred = torch.max(out, 1)     # 返回每一行中最大值和对应的索引
            total += label.size(0)
            correct += (pre == label).sum().item()
        epoch_end = time.time()
        f = open("./result.txt", "a")
        f.write('Test Accuracy: '+ str(correct / total) + "\t" + "time:" + str(epoch_end-epoch_start) + "\n")
        #print('Test Accuracy: {}'.format(correct / total), "time:", epoch_end-epoch_start)
        f.close()


