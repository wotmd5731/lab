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


from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

image = Image.new('RGBA', (200, 200))
draw = ImageDraw.Draw(image)
x = 100
y = 100
r = 10
draw.ellipse((x-r, y-r, x+r, y+r), fill=(255,0,0,255))
#draw.point((100, 100), 'red')



toten = torchvision.transforms.ToTensor()
toimg = torchvision.transforms.ToPILImage()

arr = toten(image)
plt.imshow(toimg(arr))