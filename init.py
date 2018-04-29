# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:50:22 2018

@author: JAE
"""
from torchvision import models, transforms
from torch.autograd import Variable
import torch
import torchvision
import copy
import torch.nn as nn
import torch.nn.functional as F
import random

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

class network(nn.Module):
    def __init__(self,img_size):
        super(network, self).__init__()
        self.img_size = img_size
        self.ml = nn.ModuleList()
        self.ml.append(nn.Conv2d(3,32,3,padding=1))
        self.ml.append(nn.Conv2d(32,64,3,padding=1))
        self.ml.append(nn.Conv2d(64,64,3,padding=1))
        self.ml.append(nn.Linear(64*img_size*img_size,4)
        

        
    def forward(self, x):
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
            
        x = self.c0(x)
        x = self.c1(x)
        x = self.c2(x)
        return x


model = network()


image = Image.new('RGB', (200, 200))
draw = ImageDraw.Draw(image)
x = 100
y = 100
r = 10
#draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,255))
draw.rectangle((0,0,200,200),fill='white')
#draw.point((100, 100), 'red')

toten = torchvision.transforms.ToTensor()
toimg = torchvision.transforms.ToPILImage()

x = random.randint(1,199)
y = random.randint(1,199)
draw.rectangle((0,0,200,200),fill='white')
draw.ellipse((x-r, y-r, x+r, y+r), fill='black')    

my_x = random.randint(1,199)
my_y = random.randint(1,199)
draw.ellipse((my_x-r, my_y-r, my_x+r, my_y+r), fill='blue')    
    
x_arr = toten(image)
plt.imshow(toimg(x_arr))
plt.pause(0.01)
print('')
    

model.cuda()
x_arr = x_arr.unsqueeze(0).cuda()



pred = model(x_arr)
print(torch.max(pred))

pred_y = pred.squeeze().topk(1,dim=1)
pred_x = pred_y[0].squeeze().topk(1)

xx = pred_x[1].item()
yy = pred_y[1][xx].item()

print(pred[0,0,xx,yy])


print(pred.size())

#global_count = 0
#episode = 0
#while episode < 10:
#    episode += 1
#    T=0
#    state = env.reset()
##    args.epsilon -= 0.8/args.max_episode_length
#    while T < 100:
#        T += 1
#        global_count += 1
#        
#        action = agent.get_action(state)
#       
#        next_state , reward , done, _ = env.step(action)
#        if args.reward_clip > 0:
#            reward = max(min(reward, args.reward_clip), -args.reward_clip)  # Clip rewards
# 
#        memory.push([state, action, reward, next_state, done])
#        state = next_state
#        
#        if global_count % args.replay_interval == 0 :
#            agent.basic_learn(memory)
#        if global_count % args.target_update_interval == 0 :
#            agent.target_dqn_update()
#            
#            
#        if done :
#            break
#    if episode % args.evaluation_interval == 0 :
#        test(episode)
#
