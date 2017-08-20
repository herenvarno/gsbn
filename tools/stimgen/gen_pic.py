#!/usr/bin/env python

import os
import sys
import re
import glob
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc

from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

def crop_center(img, cropx, cropy):
	x = img.shape[0]
	y = img.shape[1]
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)
	return img[starty:starty+cropy,startx:startx+cropx]
	

def find_color(c, color_map):
	a = color_map.index(c)
	if a==None:
		a = color_map[-1]
	return a
	
if len(sys.argv) < 6:
	print("Arguments wrong! Please retry with command :")
	print("python "+os.path.realpath(__file__)+" <image file directory> <Height> <Width> <Channel> <Output filename>")
	exit(-1)

lines = []
with open("colors.map") as f:
	lines = f.read().split()

if(len(lines)!=int(lines[0])+1):
	print("color map error!")
	
color_map = [int(x) for x in lines[1:]]
print(color_map)

#mndata = MNIST('./mnist')
#images, labels = mndata.load_training()

#for i,image in enumerate(images):
#	if i>50 :
#		break
#	img = np.asarray(image).reshape((28,28))
#	misc.imsave("./mnist/"+str(i)+".png", img)
#	
#	
#exit()
files = []
for f in glob.glob( os.path.join(sys.argv[1], '*.png') ):
	files.append(f)

files=sorted(files)

patterns = []
masks = []

H = int(sys.argv[2])
W = int(sys.argv[3])
C = int(sys.argv[4])
DIM_HCU = H*W*C
DIM_MCU = 40

rd = gsbn_pb2.StimRawData()
rd.data_cols = DIM_HCU;
rd.mask_cols = DIM_HCU;

masks.append(np.zeros(DIM_HCU, dtype=np.uint8))
masks.append(np.ones(DIM_HCU, dtype=np.uint8))

patterns.append(np.zeros(DIM_HCU, dtype=np.uint8)+0x7fffffff)

for f in files:
	img0 = misc.imresize(misc.imread(f), (H,W))
	img0 = crop_center(img0, H, W)
#	plt.imshow(img0, interpolation="nearest", cmap='spectral')
#	plt.show()
	img = np.zeros([H,W,C],np.uint32)
	for h in range(H):
		for w in range(W):
			a = 0
			a = a | img0[h][w][0]
			a = a<<8 | img0[h][w][1]
			a = a<<8 | img0[h][w][2]
			
			b = find_color(a, color_map)
			
			for c in range(C):
				img[h,w,c] = b%DIM_MCU
				b = b//DIM_MCU
	img = img.reshape(-1)
	if len(img.shape)==1 and img.shape[0] == DIM_HCU:
		patterns.append(img)

# Compete patterns
# pattern 1 50% vs pattern 2 50%
img = np.zeros(DIM_HCU, dtype=np.uint8)
for h in range(H):
	for w in range(W):
		if w<W*0.5:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[1][(h*W+w)*C+c]
		else:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[2][(h*W+w)*C+c]
patterns.append(img)

# pattern 3 40% vs pattern 4 60%
img = np.zeros(DIM_HCU, dtype=np.uint8)
for h in range(H):
	for w in range(W):
		if w<W*0.4:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[3][(h*W+w)*C+c]
		else:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[4][(h*W+w)*C+c]
patterns.append(img)

# pattern 5 30% vs pattern 6 70%
img = np.zeros(DIM_HCU, dtype=np.uint8)
for h in range(H):
	for w in range(W):
		if w<W*0.3:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[5][(h*W+w)*C+c]
		else:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[6][(h*W+w)*C+c]
patterns.append(img)

# pattern 7 20% vs pattern 8 80%
img = np.zeros(DIM_HCU, dtype=np.uint8)
for h in range(H):
	for w in range(W):
		if w<W*0.2:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[7][(h*W+w)*C+c]
		else:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[8][(h*W+w)*C+c]
patterns.append(img)

# pattern 9 10% vs pattern 10 90%
img = np.zeros(DIM_HCU, dtype=np.uint8)
for h in range(H):
	for w in range(W):
		if w<W*0.1:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[9][(h*W+w)*C+c]
		else:
			for c in range(C):
				img[(h*W+w)*C+c] = patterns[10][(h*W+w)*C+c]
patterns.append(img)


rd.data_rows=len(patterns)
rd.mask_rows=len(masks)

for p in patterns:
	for d in p:
		rd.data.append(int(d))

for m in masks:
	for d in m:
		rd.mask.append(float(d))

with open(sys.argv[5], "wb+") as f:
	f.write(rd.SerializeToString())

