from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

number = '14'

filename = np.loadtxt('D:\Kepler\Kepler-' + number + '/filename lc.txt', dtype='str')  # 檔名txt
#f1 = open('D:\Kepler\Kepler-' + number + '/ks.txt', 'w+')

n = filename.size
histime = []

for i in range(n):
    # print(filename[i])
    table = fits.open('D:\Kepler\Kepler-' + number + '\data/' + filename[i])  # 檔案路徑
    time0 = table[1].data['TIME']
    flux0 = table[1].data['PDCSAP_FLUX']
    flux_err0 = table[1].data['PDCSAP_FLUX_ERR']
    notnan = np.where(np.logical_not(np.isnan(flux0)))

    k = 10
    time1 =[]
    flux1 = []
    for ktime in range (2 * k + 1):
        time1 = np.append(time1, [np.nan])
    for kflux in range (2 * k + 1):
        flux1 = np.append(flux1, [np.nan])
    time1 = np.append(time1, [time0])
    flux1 = np.append(flux1, [flux0])
    #flux_err = flux_err0[notnan]

#flus = []
nan = np.where(np.isnan(flux1))
flux = []
time = []
flux = flux1[notnan]
angry = np.where(np.logical_not(np.isnan(time1)))
time = time1[angry]
notnan = np.where(np.logical_not(np.isnan(flux1)))
fluxm = np.array(flux)
avr_flux = np.zeros(len(flux))
g = np.int(len(flux) / 20)
std_flux = np.zeros([g])

# nan surounding deleting
nanp = np.zeros(len(fluxm))

for gan in range(2 * k):
    for fan in range(len(nan) - (gan + 1)):
        if ((nan[fan + gan + 1] - nan[fan]) == (gan + 1)):
            nanp[nan[fan]] = nanp[nan[fan]] + 0.5
            nanp[nan[fan + gan + 1]] = nanp[nan[fan + gan + 1]] + 0.5

nanpremark0 = np.where(nanp == 2 * k)
nanpremark = np.copy(nanpremark0[0][:])
nanmark = np.copy(nanpremark0[0][:])
if (nanmark != []):
    for qan in range(2 * k):
        if ((nanpremark[0] - qan - 1) >= 0):
            nanmark = np.append(nanmark, [nanpremark[0] - qan - 1])
        if ((nanpremark[len(nanpremark) - 1] + qan + 1) < len(nanmark)):
            nanmark = np.append(nanmark, [nanpremark[len(nanpremark) - 1] + qan + 1])

        for han in range(len(nanpremark) - 1):
            if (nanpremark[han + 1] - nanpremark[han] != 1):
                for dan in range(2 * k):
                    nanmark = np.append(nanmark, [nanpremark[han] + dan + 1])
                    nanmark = np.append(nanmark, [nanpremark[han + 1] - dan - 1])

print(nanmark)
y = 0

for c in range(g):
    std_flux[c] = np.std(flux[y:(y + 20)])
    y = y + 20

std = np.median(std_flux)

for a in range(len(flux) - 2 * k):
    avr_flux[a + k] = np.mean(flux[a:(a + 2 * k + 1)])

for b in range(len(flux) - 5 * k):
    if (flux[b + 3 * k] < (avr_flux[b + k] + avr_flux[b + 5 * k]) / 2 - 3 * std and fluxm[
            b + 3 * k] not in nanmark):
        fluxm[b + 3 * k] = np.nan

# transit highlighting
kat0 = np.where(np.isnan(fluxm))  # and np.logical_not(nanmark)
katp = np.zeros(len(fluxm))
kat = np.copy(kat0[0][:])

for cnacer in range(1):
    #x = 25-cnacer
    x = 3  # deleting marks less than 2

    for g in range(x):
        for f in range(len(kat) - (g + 1)):
            if ((kat[f + g + 1] - kat[f]) == (g + 1)):
                katp[kat[f]] = katp[kat[f]] + 0.5
                katp[kat[f + g + 1]] = katp[kat[f + g + 1]] + 0.5

    premark0 = np.where(katp == x)
    premark = np.copy(premark0[0][:])
    mark = np.copy(premark0[0][:])

    if(premark != []):
        for q in range(x):
            if ((premark[0] - q - 1) >= 0):
                mark = np.append(mark, [premark[0] - q - 1])
            if ((premark[len(premark) - 1] + q + 1) < len(mark)):
                mark = np.append(mark, [premark[len(premark) - 1] + q + 1])

        for h in range(len(premark) - 1):
            if (premark[h + 1] - premark[h] != 1):
                for d in range(x):
                    mark = np.append(mark, [premark[h] + d + 1])
                    mark = np.append(mark, [premark[h + 1] - d - 1])

histime = time[mark]
print(histime)
sortime = np.sort(histime)
dis_histime = np.zeros(len(sortime) - 1)
for j in range(len(sortime) - 1):
    dis_histime[j] = sortime[j + 1] - sortime[j]
print(dis_histime)
plt.hist(dis_histime, bins=100)
hist = plt.hist(dis_histime, bins=100)
histy = hist[0]
histx = hist[1]
plt.show()

# ks = np.loadtxt('D:\Kepler\Kepler-30/hist' + n + '.txt', dtype = 'float')
# print(ks)






'''for c in range(len(flux)-2*k):
    avr_flux[c+k] = np.nanmean(fluxm[c:c+2*k+1])

for d in range(len(flux)):
    flus[d] = flux[d]/avr_flux[d]

c1 = fits.Column(name = 'TIME', array = time, format = 'F')
c2 = fits.Column(name = 'FLUX', array = flus, format = 'F')
c3 = fits.Column(name = 'FLUX_ERR', array=flux_err, format='F')
t = fits.BinTableHDU.from_columns([c1,c2,c3])
name = str(k)
t.writeto('D:\專題\Kepler-10/new/'+name+'/'+filename[i])#處理後存檔
print('data',filename[i],'is complete')'''
