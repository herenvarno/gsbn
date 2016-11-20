import os
import sys
import re
import numpy as np

x = np.arange(19)
t = []
for x in range(19):
	t.append([x*10/10, (x+1)*10/10])
for i, tt in enumerate(t):
	s = tt[0]
	e = tt[1]
	if(i<9):
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
	else:
		string = "\
	mode_param : {\n\
		begin_time : %s\n\
		end_time : %s\n\
		prn : 0\n\
		gain_mask: 1\n\
		plasticity: 0\n\
		stim_index : %s\n\
	}\n\
" % (str(s), str(e), str(i))
	print(string)

