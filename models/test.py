import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


rank1= []
eps_list = []
lambda_list = []

with open('ilids-vid_to_prid2011_to_TCLNet.csv', newline='') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    cpt = 0
    for row in csv_reader:
        if cpt % 2 == 1:
            print('\n', row)

            e = row[0].find('eps=')
            l = row[0].find('lambda=')

            eps_list.append(float(row[0][e + 4:]))

            x = row[0][l + 7:].find('eps=')
            lambda_list.append(float(row[0][l + 7:][:x-1]))

            rank1.append(float(row[1]))

        cpt += 1


eps = np.array(eps_list)
lambd = np.array(lambda_list)
rank = np.array(rank1)

x = eps.astype(float)
y = lambd.astype(float)
z = rank.astype(float)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z, "*")
plt.show()
