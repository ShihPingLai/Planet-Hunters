from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np


filename = np.loadtxt('D:/Kepler/Kepler-387/filename lc.txt', dtype = 'str')
n = filename.size
fluxall = np.array([1])
timeall = np.array([0])
k = 10

for i in range(n):
    table = fits.open('D:/Kepler/Kepler-387/new lc/'+filename[i])
    time = table[1].data['TIME']
    flux = table[1].data['FLUX']
    flux[0:k] = 1.0
    flux[(np.size(flux)-k):np.size(flux)] = 1.0
    fluxall = np.concatenate((fluxall,flux),axis=0)
    timeall = np.concatenate((timeall,time),axis=0)
    print(filename[i]+'combined')


c1 = fits.Column(name = 'TIME', array = timeall, format = 'F')
c2 = fits.Column(name = 'FLUX', array = fluxall, format = 'F')
t = fits.BinTableHDU.from_columns([c1,c2])
#t.writeto('D:/Kepler/Kepler-387/new lc/longall.fits')
print('data longall is complete')
print(fluxall)

std_fluxall = np.nanstd(fluxall)
A = np.where(fluxall < 1 - 3*std_fluxall)
timec = timeall[A]
fluxc = fluxall[A]
dis_time = np.zeros(len(timec)-1)
for j in range(len(timec)-1):
    dis_time[j] = timec[j+1] - timec[j]
plt.hist(dis_time,bins=100)
plt.xlim(3,150)
plt.ylim(0,20)
plt.show()
plt.plot(timeall,fluxall,'g.')
plt.plot(timeall[A],fluxall[A],'r.')
plt.show()