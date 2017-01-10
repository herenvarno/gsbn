import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

p_frac_bit = [13, 14, 15, 16, 17, 18]
n_frac_bit = ["fix16(other-traces frac bit=2)", "fix16(other-traces frac bit=3)", "fix16(other-traces frac bit=4)", "fix16(other-traces frac bit=5)", "fix16(other-traces frac bit=6)", "fix16(other-traces frac bit=7)", "fix16(other-traces frac bit=8)", "fix16(other-traces frac bit=9)", "fix16(other-traces frac bit=10)"]
c = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '0.8', '0.4']
fp32_5 = np.repeat(2, len(p_frac_bit))
fp16_5 = np.repeat(24, len(p_frac_bit))
fix_5 = [
	[0, 1, 2, 5, 12, 16],
	[3, 8, 15, 30, 34, 39],
	[6, 12, 25, 32, 34, 42],
	[7, 18, 23, 23, 16, 32],
	[8, 17, 24, 19, 26, 23],
	[7, 16, 26, 14, 32, 19],
	[7, 15, 25, 13, 34, 16],
	[7, 16, 22, 13, 32, 15],
	[8, 19, 28, 24, 36, 14]]
fp32_2 = np.repeat(14, len(p_frac_bit))
fp16_2 = np.repeat(43, len(p_frac_bit))
fix_2 = [
	[0, 1, 1, 8, 22, 25],
	[3, 10, 24, 34, 45, 50],
	[7, 20, 32, 43, 50, 50],
	[7, 22, 33, 38, 42, 48],
	[12, 24, 35, 40, 39, 42],
	[11, 25, 32, 37, 38, 35],
	[11, 24, 32, 37, 38, 36],
	[12, 24, 36, 38, 37, 38],
	[9, 22, 33, 44, 38, 33]]
fp32_n1 = np.repeat(32, len(p_frac_bit))
fp16_n1 = np.repeat(44, len(p_frac_bit))
fix_n1 = [
	[7, 8, 8, 15, 27, 29],
	[10, 15, 28, 40, 49, 49],
	[13, 25, 38, 47, 50, 50],
	[18, 29, 40, 45, 46, 47],
	[18, 28, 42, 43, 42, 45],
	[19, 30, 41, 42, 43, 45],
	[19, 29, 42, 42,44, 45],
	[19, 26, 39, 44, 42, 43],
	[19, 30, 42, 48, 45, 44]]
	
fig=plt.figure(1)
fig.suptitle('10x10 Network remembers 50 patterns', fontweight='bold')

ax=fig.add_subplot(221)
ax.set_title("50% HCU need recall")
ax.set_ylim([0,50])
ax.set_xlabel("P-traces frac bit")
ax.set_ylabel("correctly recalled pattern")
ax.plot(p_frac_bit, fp32_5, '--', label="fp32")
ax.plot(p_frac_bit, fp16_5, '--', label="fp16(p-traces are fp32)")
for i,p in enumerate(fix_5):
	ax.plot(p_frac_bit, p, '-o', color=c[i], label=n_frac_bit[i])

ax=fig.add_subplot(222)
ax.set_title("20% HCU need recall")
ax.set_ylim([0,50])
ax.set_xlabel("P-traces frac bit")
ax.set_ylabel("correctly recalled pattern")
ax.plot(p_frac_bit, fp32_2, '--', label="fp32")
ax.plot(p_frac_bit, fp16_2, '--', label="fp16(p-traces are fp32)")
for i,p in enumerate(fix_2):
	ax.plot(p_frac_bit, p, '-o', color=c[i], label=n_frac_bit[i])

ax=fig.add_subplot(223)
ax.set_title("only 1 HCU needs recall")
ax.set_ylim([0,50])
ax.set_xlabel("P-traces frac bit")
ax.set_ylabel("correctly recalled pattern")
ax.plot(p_frac_bit, fp32_n1, '--', label="fp32")
ax.plot(p_frac_bit, fp16_n1, '--', label="fp16(p-traces are fp32)")
for i,p in enumerate(fix_n1):
	ax.plot(p_frac_bit, p, '-o', color=c[i], label=n_frac_bit[i])
	
ax.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))

plt.tight_layout()
plt.show()
