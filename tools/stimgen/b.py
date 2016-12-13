import os
import sys
import re
import numpy as np

for l in range(2):
	if l<1:
		x = np.arange(l*10, (l+1)*10)
		t = []
		for ss in x:
			t.append([ss*10/10, (ss+1)*10/10])
		for i, tt in enumerate(t):
			s = tt[0]
			e = tt[1]
			string = "\
		mode_param : {\n\
			begin_time : %s\n\
			end_time : %s\n\
			prn : 1\n\
			gain_mask: 0\n\
			plasticity: 1\n\
			stim_index : %s\n\
		}\n\
		" % (str(s), str(e), str(i))
			print(string)
	else:
		x = np.arange(l*10, (l+1)*10)
		t = []
		for ss in x:
			t.append([ss*10/10, (ss+1)*10/10])
		for i, tt in enumerate(t):
			s = tt[0]
			e = tt[1]
			string = "\
	mode_param : {\n\
		begin_time : %s\n\
		end_time : %s\n\
		prn : 0\n\
		gain_mask: 1\n\
		plasticity: 0\n\
		stim_index : %s\n\
	}\n\
		" % (str(s), str(e), str(i+10))
			print(string)

