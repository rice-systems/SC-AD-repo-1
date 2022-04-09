import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
import matplotlib.ticker as ticker


plt.rcParams["figure.figsize"] = (8,6)
matplotlib.rcParams.update({'font.size': 14})

benches = ['fir', 'radix-sort', 'hash-join']
allocs=[
	['0', '5857', '7840', '8832'],
	['0', '6435', '8226', '9121'],
	['0', '6282', '8124', '9044']]

prefix=['prefetch', 'discard', 'discardlazy']

data = np.zeros(4)
baseline = np.zeros(4)
for i in np.arange(3):
	print(benches[i] + '\n')
	for k in np.arange(3):
		for j in np.arange(4):
			tmp = np.loadtxt('data/' + benches[i] + '/' + prefix[k] + '-' + allocs[i][j] + '.txt')
			data[j] = np.mean(tmp)
		if (k == 0):
			for kk in np.arange(4):
				baseline[kk] = data[kk]
		for j in np.arange(4):
			print("{:.2f} ".format(data[j] / baseline[j]), end='')
			# print("{:.2f} ".format(data[j]), end='')
		print("\n")