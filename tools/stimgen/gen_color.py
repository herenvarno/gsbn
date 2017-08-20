#!/usr/bin/env python

import os
import sys
import re
import glob
import math
import random
import matplotlib.pyplot as plt
import numpy as np

from google.protobuf import text_format
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../build")
import gsbn_pb2

def img_gen_func_1(h, w, H, W):
	if h<H/2:
		if w<W/2:
			return True
	return False
	
def img_gen_func_2(h, w, H, W):
	if h<H/2:
		if w>=W/2:
			return True
	return False

def img_gen_func_3(h, w, H, W):
	if h>=H/2:
		if w<W/2:
			return True
	return False

def img_gen_func_4(h, w, H, W):
	if h>=H/2:
		if w>=W/2:
			return True
	return False

def img_gen_func_5(h, w, H, W):
	if h==w or h==W-w-1:
			return True
	return False

def img_gen_func_r25(h, w, H, W):
	if random.uniform(0, 1) < 0.25:
			return True
	return False

def img_gen_func_r50(h, w, H, W):
	if random.uniform(0, 1) < 0.50:
			return True
	return False

def img_gen_func_r60(h, w, H, W):
	if random.uniform(0, 1) < 0.60:
			return True
	return False

def img_gen_func_r70(h, w, H, W):
	if random.uniform(0, 1) < 0.70:
			return True
	return False

def img_gen_func_r80(h, w, H, W):
	if random.uniform(0, 1) < 0.80:
			return True
	return False

def img_gen_func_r90(h, w, H, W):
	if random.uniform(0, 1) < 0.90:
			return True
	return False

def img_gen_func_full(h, w, H, W):
	return True;

def main():
	img_generator = {
		1 : img_gen_func_full,
		2 : img_gen_func_full,
		3 : img_gen_func_full,
		4 : img_gen_func_full,
		5 : img_gen_func_full,
		6 : img_gen_func_full,
		7 : img_gen_func_full,
		8 : img_gen_func_full,
		9 : img_gen_func_full,
		10 : img_gen_func_full
	}

	print(sys.argv)
	if len(sys.argv) < 5:
		print("Arguments wrong! Please retry with command :")
		print("python "+os.path.realpath(__file__)+" <Height> <Width> <Color type> <Output filename>")
		exit(-1)

	patterns = []
	masks = []

	H = int(sys.argv[1])
	W = int(sys.argv[2])
	C = int(sys.argv[3])
	DIM_HCU = H*W
	DIM_MCU = C

	rd = gsbn_pb2.StimRawData()
	rd.data_cols = DIM_HCU;
	rd.mask_cols = DIM_HCU;

	masks.append(np.zeros(DIM_HCU, dtype=np.uint8))
	masks.append(np.ones(DIM_HCU, dtype=np.uint8))

	patterns.append(np.zeros(DIM_HCU, dtype=np.uint8)+0x7fffffff)

	for i,g in enumerate(img_generator.keys()):
		img = np.zeros(DIM_HCU, dtype=np.uint8)
		for h in range(H):
			for w in range(W):
				if img_generator[g](h, w, H, W):
					img[h*W+w] = i
				else:
					img[h*W+w] = 0x7fffffff
		patterns.append(img)
	for i,g in enumerate(img_generator.keys()):
		img = np.zeros(DIM_HCU, dtype=np.uint8)
		for h in range(H):
			for w in range(W):
				if img_generator[g](h, w, H, W) and h<5 and w<5:
					img[h*W+w] = i
				else:
					img[h*W+w] = 0x7fffffff
		patterns.append(img)

	rd.data_rows=len(patterns)
	rd.mask_rows=len(masks)

	for p in patterns:
		for d in p:
			rd.data.append(int(d))

	for m in masks:
		for d in m:
			rd.mask.append(float(d))

	with open(sys.argv[4], "wb+") as f:
		f.write(rd.SerializeToString())

if __name__ == '__main__':
	main()

