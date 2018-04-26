from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

'''table = fits.open('D:/Kepler/Kepler-387/new lc/longall.fits')
time = table[1].data['TIME']
flux = table[1].data['FLUX']
dis_time = np.zeros(len(time)-1)
for j in range(len(time)-1):
    dis_time[j] = time[j+1] - time[j]'''

array = [1, 2, 3, 3, 3, 5, 9, 9]

test = plt.hist(array,bins=8)
testy = test[0]
testx = test[1]
print(testy)
print(testx)

plt.show()