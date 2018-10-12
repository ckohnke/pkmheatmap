import requests
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import re
import numpy as np

def calcCarTime(APP_ID,APP_CODE,lat1,lon1,lat2,lon2,tunit='s'):
    url = "https://route.api.here.com/routing/7.2/calculateroute.xml"
    data = '?app_id=%s&app_code=%s&waypoint0=geo!%s,%s&waypoint1=geo!%s,%s&routeattributes=wp,sm&mode=fastest;car'%(APP_ID,APP_CODE,lat1,lon1,lat2,lon2)
    response = requests.post(url+data).text
    soup = BeautifulSoup(response, 'lxml')

    #print(soup.prettify())
    try:
        summary = soup.find_all('summary')[0]
        meter = soup.find_all('summary')[0].find_all('distance')[0].text
        secs  = soup.find_all('summary')[0].find_all('traveltime')[0].text
    except:
        return(np.nan, np.nan)

    re.sub("[^0-9]", "", meter)
    re.sub("[^0-9]", "", secs)

    if(tunit == 's'):
        time = float(secs)
    elif(tunit=='m'):
        time = float(secs)/60
    elif(tunit=='h'):
        time = float(secs)/60/60

    conv_fac = 0.621371 # km to miles
    miles = float(meter)/1000*conv_fac

    return(miles,time)

if __name__ == "__main__":
    config = np.loadtxt("../config",dtype=str)
    APP_ID = config[0]
    APP_CODE = config[1]    
    miles, time = calcCarTime(APP_ID,APP_CODE,39.755543,-105.221100,40.585258,-105.084419,'h') # Golden, CO to FOrt COllins, CO - approx 72 miles, 1.2 hours
    print(miles,time)
