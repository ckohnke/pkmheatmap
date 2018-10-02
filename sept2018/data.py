# example showing how to plot scattered data with hexbin.
import csv
from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from geopy.geocoders import Here
from pylab import *
from scipy.interpolate import griddata
import sys
import matplotlib.mlab as mlab

config = np.loadtxt("../config",dtype=str)
APP_ID = config[0]
APP_CODE = config[1]

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['axes.titlesize'] = label_size + 4


DF = "sept2018_pc.csv"
HEADER = "Date,City,State,Event Type,TO,Best of,Event Locator Link,Notes,Total Players,Lat,Lon"

# create north polar stereographic basemap
m = Basemap(width=6000000,height=4500000,resolution='c',projection='aea',lat_1=35.,lat_2=45,lon_0=-100,lat_0=40)
geolocator = Here(APP_ID, APP_CODE)

# number of points, bins to plot.
bins = 12

# Read the data
data = np.genfromtxt(DF, dtype=str, delimiter=",", skip_header=1)

city =  data[:,1]
state = data[:,2]
nplayers = []
for ii in data[:,8]:
    if ii == '':
        nplayers.append(0)
    else:
        nplayers.append(int(ii))

lookup = []
for f in data:
    if(not f[10]):
        lookup.append(f[1]+', '+f[2])

lats = []
lons = []
for s in lookup:
    location = geolocator.geocode(s)
    lats.append(location.latitude)
    lons.append(location.longitude)
    print(s,location.latitude,location.longitude)
    time.sleep(1)

ii = 0
for f in data:
    if(not f[10]):
        f[9 ] = lats[ii]
        f[10] = lons[ii]
        ii = ii+1

# save if we had to lookup lat / lon data
if (lookup != []):
    np.savetxt(DF,data,fmt="%s",delimiter=",",header=HEADER)

nplayers=np.array(nplayers)
lats=np.array(data[:, 9], dtype=float)
lons=np.array(data[:,10], dtype=float)

plt.figure(figsize=(10,8))
nhist, bhist, patches = plt.hist(nplayers, density=1, bins=9, alpha=0.75, histtype='stepfilled')
mu = np.mean(nplayers)
sigma = np.std(nplayers)
y = mlab.normpdf(bhist, mu, sigma)
plt.plot(bhist, y, 'r--')
plt.xlabel('Number of Players', fontsize=20)
plt.ylabel('Probability', fontsize=20)
plt.title('Histogram of Players - September 2018',fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(axis='y', alpha=0.75)

print("count:", len(data[:,1]))
print("sum:", np.sum(nplayers))
print("mean:", np.mean(nplayers))
print("median:", np.median(nplayers))
print("std:", np.std(nplayers))
print("var:", np.var(nplayers))

# convert to map projection coordinates.
x, y = m(lons, lats)

# Plot 1 - Number of events
# make plot using hexbin
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('coolwarm',5)
CS = m.hexbin(x,y,gridsize=bins,mincnt=1,cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(ticks=range(6), location="bottom", pad=0.4) # draw colorbar
cbar.set_label("Number of PCs",fontsize=16)
plt.clim(0.5, 5.5)
plt.title('Number of PCs - September 2018', fontsize=20)
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

# Plot 2 - Number of Players
# make plot using hexbin
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('coolwarm',20)
CS = m.hexbin(x,y,C=nplayers,gridsize=bins, mincnt=0,cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Average Number of Players",fontsize=16)
plt.title('Average Number of Players Per PC - September 2018', fontsize=20)
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

# Plot 3 - Heatmap

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('coolwarm',20)
CS1 = m.contourf(x,y,nplayers,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Number of Players")
plt.title('Player Heat Map - September 2018')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

#plt.gcf().set_size_inches(18,10)

plt.show()

