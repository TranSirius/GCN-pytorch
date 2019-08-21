import numpy as np

f = open('Res.txt', 'r')
lines = f.readlines()

rec = []
for line in lines:
    rec.append(float(line.replace("\n", "")))

print("min: %5.3f, stdv: %3.2f\nmax: %5.3f, min:  %5.3f" % (np.mean(rec) * 100, np.std(rec) * 100, max(rec) * 100, min(rec) * 100))