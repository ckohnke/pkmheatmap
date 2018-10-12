# example showing how to plot scattered data with hexbin.
import csv
from numpy.random import uniform
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import maskoceans
from geopy.geocoders import Here
from pylab import *
from scipy.interpolate import griddata
import sys
import matplotlib.mlab as mlab
from here import *
from heapq import nsmallest, nlargest

config = np.loadtxt("../config",dtype=str)
APP_ID = config[0]
APP_CODE = config[1]

label_size = 16
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size 
mpl.rcParams['axes.labelsize'] = label_size
mpl.rcParams['axes.titlesize'] = label_size + 4

DF = "all_regs.csv"
MTX = DF.strip('.csv')+'_time.mtx'
MDX = DF.strip('.csv')+'_dist.mtx'
HEADER = "Date,City,State,Country,Event Type,TO,Best of,Event Locator Link,Notes,Total Players,Latitude,Longitude"

# create north polar stereographic basemap
m = Basemap(width=6000000,height=5000000,resolution='l',projection='aea',lat_1=35.,lat_2=45,lon_0=-97,lat_0=38)
geolocator = Here(APP_ID, APP_CODE)

# number of points, bins to plot.
bins = 12

# Read the data
data = np.genfromtxt(DF, dtype=str, delimiter=",", skip_header=1)

city =  data[:,1]
state = data[:,2]
country = data[:,3]
lookup = []
for f in data:
    if(not f[10]):
        lookup.append(f[1]+', '+f[2]+', '+f[3])

#####################
# Lookup lat / lon of places we don't have lat / lons for
lats = []
lons = []
for s in lookup:
    location = geolocator.geocode(s)
    lats.append(location.latitude)
    lons.append(location.longitude)
    print(s,location.latitude,location.longitude)

ii = 0
for f in data:
    if(not f[10]):
        f[10] = lats[ii]
        f[11] = lons[ii]
        ii = ii+1

# save if we had to lookup lat / lon data
if (lookup != []):
    np.savetxt(DF,data,fmt="%s",delimiter=",",header=HEADER)

lats=np.array(data[:,10], dtype=float)
lons=np.array(data[:,11], dtype=float)

#####################
fdg = 2
top = 52.3457868 + fdg # north lat
left = -126.7844079 - fdg # west long
right = -64.9513812 + fdg # east long
bottom =  17 - fdg # south lat

US_NS = np.linspace(bottom,top,np.abs(int(1*(top-bottom))))
US_EW = np.linspace(right,left,np.abs(int(1*(left-right))))
US_GRIDX, US_GRIDY = np.meshgrid(US_NS,US_EW)

#####################
# Get data over the US

tlats = US_GRIDX.reshape((np.size(US_GRIDX),1))
tlons = US_GRIDY.reshape((np.size(US_GRIDY),1))
dlats = []
dlons = []
bm = Basemap()
for a, b in zip(tlats,tlons):
    ii = a[0]
    jj = b[0]
    if bm.is_land(jj,ii):
        dlats.append(ii)
        dlons.append(jj)

print(len(dlats))
mm = np.zeros((len(lats),len(dlats)),dtype=float)
dd = np.zeros((len(lats),len(dlats)),dtype=float)
try:
    fh = open(MTX, 'r')
    fh = open(MDX, 'r')
    mm = np.genfromtxt(MTX, dtype=float)
    dd = np.genfromtxt(MDX, dtype=float)
except FileNotFoundError:
#    for ii in range(0,len(lats)):
#        for jj in range(0,len(dlats)):
    print(len(lats)*len(dlats))
    for ii in range(0,1):
        for jj in range(0,1):
            miles, time = calcCarTime(APP_ID,APP_CODE,dlats[jj],dlons[jj],lats[ii],lons[ii],tunit='h')
            print(ii, jj, miles, time)
            mm[ii][jj] = time
            dd[ii][jj] = miles
    np.savetxt(MTX, mm)
    np.savetxt(MDX, dd)

#####################
'''    
minm = []
maxm = []
mint = []
maxt = []
avem = []
avet = []
medm = []
medt = []
stdm = []
stdt = []
for ii in range(0,len(dlats)):
    minm.append(np.min(mm.T[ii]))
    mint.append(np.min(dd.T[ii]))
    maxm.append(np.max(mm.T[ii]))
    maxt.append(np.max(dd.T[ii]))
'''

x, y = m(lons, lats)
ux, uy = m(dlons, dlats)

plt.figure(figsize=(10,8))
m.drawcoastlines(linewidth=0.5)
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
plt.title('Driving Distance to Regionals')
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)
m.plot(ux,uy, 'o', markersize=5,zorder=6, markerfacecolor='#80b442',markeredgecolor="none", alpha=0.66)

'''
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,minm_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Miles")
plt.title('Drive Distance to Regionals - Minimum')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,mint_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Hours")
plt.title('Drive Time to Regionals - Closest')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,minm_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Miles")
plt.title('Drive Distance to Regionals - Maximum')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,mint_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Hours")
plt.title('Drive Time to Regionals - Maximum')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,minm_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Miles")
plt.title('Drive Distance to Regionals - Mean')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,mint_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Hours")
plt.title('Drive Time to Regionals - Mean')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,minm_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Miles")
plt.title('Drive Distance to Regionals - Median')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,mint_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Hours")
plt.title('Drive Time to Regionals - Median')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,minm_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Miles")
plt.title('Drive Distance to Regionals - SD')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
cmap = cm.get_cmap('jet')
CS1 = m.contourf(ux,uy,mint_mask,tri=True, extend='both',cmap=cmap)
# draw coastlines, lat/lon lines.
m.drawcoastlines(linewidth=0.5)
# draw parallels and meridians.
m.drawparallels(np.arange(-90.,91.,15.),labels=[True,True,False,False],dashes=[2,2])
m.drawmeridians(np.arange(-180.,181.,15.),labels=[False,False,False,True],dashes=[2,2])
m.drawcountries(linewidth=2, linestyle='solid', color='k' ) 
m.drawstates(linewidth=0.5, linestyle='solid', color='k')
cbar = m.colorbar(location="bottom",pad=0.4) # draw colorbar
cbar.set_label("Hours")
plt.title('Drive Time to Regionals - SD')
# translucent blue scatter plot of epicenters above histogram:    
m.plot(x, y, 'o', markersize=5,zorder=6, markerfacecolor='#80a442',markeredgecolor="none", alpha=0.66)

'''
plt.show()
