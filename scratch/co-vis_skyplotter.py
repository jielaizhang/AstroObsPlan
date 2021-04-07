

'''
###################################################################################################
############        CO-VISIBILTY CALCULATOR AND PLOTTER FOR SEVERAL SKY POSITIONS       ###########
###################################################################################################

        co-vis_skyplotter.py--This is a script that plots the time(in minutes) 
        a target/positionon the celestial sphere can be observed by two [or more]
        telescopes/locations spatially seperated on the surface of the Earth.

!!!!! 

Third party dependencies: Astropy, Astroplan, cv2, pandas, colorama, matplotlib

!!!!!                                                                 


-Right Ascension(RA) is spaced from 0 to 360 *number of steps specified by user* 
-Declination(dec) is from 90 to -90 *number of steps specified by user*


-Start date and time of observation specified by user(format: YYYY,M,D, H,M)
i.e start_date = datetime(2021,1, 7, 18, 0)

-End date and time of observation specified by user(format: YYYY,M,D, H,M) 
i.e end_date = datetime(2021,1, 7, 18, 0)


-Time interval specified by user i.e time-interval_in_mins=15


-lower limit and upper limit of airmassess at both telescopes/locations speficied by user


-Geolocation information of FIRST Telescope/Observatory specified by user
i.e.

Obs1_longitude=-70.81500000000001       #lontitude of observatory 1 
Obs1_latitude=-30.165277777777778       #latitude of observatory 1 
obs1_elevation=2214.9999999993697       #elevation of observatory 1
obs1_name='CTIO'                        #name of observatory 1

-Specify if night only observation should be carried out at FIRST telescope/location
i.e night_only_obs1= 'y'      ('y' for yes or 'n'for no)  

-Geolocation information of SECOND Telescope/Observatory specified by user
i.e.

obs2_longitude=149.06119444444445   #lontitude of observatory 2
obs2_latitude=-31.273361111111104   #latitude of observatory 2 
obs2_elevation=1149.0000000015516   #elevation of observatory 2
obs2_name='KMTNET' 

-Specify if night only observation should be carried out at SECOND telescope/location
i.e night_only_obs2= 'y'      ( 'y' for yes or 'n'for no)  


Input location to save final plot i.e. save_location= '/home/andrew/src/Transient-Scheduler/scripts/' 

*****************************IMPORTANT******************************************************
*  In some cases, there may be 2 or more co visibility occurances.Hence user has to decide * 
*                                                                                          *
*  if time sum-total of co-vis occurances should be added and returned                     *
*        i.e "co-vis_type='sum'"                                                           *
*                                                                                          *
*  or the larger of the co_vis occurence be returned  i.e "co-vis_type='max'"              * 
*                                                                                          *
*  'sum' or 'max' is specified by user                                                     *
********************************************************************************************




Usage:
   co-vis_skyplotter.py][--co_vis_type=<'max'>] <yr1><mn1><dy1><hr1><min1><yr2><mn2><dy2><hr2><min2><time_interval=20><t1_lon> <t1_lat> <t1_el> <t1_name><t2_lon> <t2_lat> <t2_el> <t2_name>
    co-vis_skyplotter.py<yr1> <mn1> <dy1> <hr1> <min1> <yr2> <mn2> <dy2> <hr2> <min2> <time_interval=20> <t1_lon> <t1_lat> <t1_el> <t1_name><t2_lon> <t2_lat> <t2_el> <t2_name>
    co-vis_skyplotter.py [nt1='no'][nt2='no'][--co_vis_type=<'max'>] <yr1> <mn1> <dy1> <hr1> <min1> <yr2> <mn2> <dy2> <hr2><min2> <time_interval=20><t1_lon> <t1_lat> <t1_el> <t1_name><t2_lon> <t2_lat> <t2_el> <t2_name>





       co-vis_skyplotter.py  -h | --help
       co-vis_skyplotter.py  --version



Arguments:
    time_interval=<int>                      Time interval step between start and end of observations
    
    start_obs=<datetime(2021,1, 7, 18, 0)>   Start datetime of observation i.e (2021,1, 7, 18, 0)

    
    end_obs = <datetime(2021,1,8, 18, 0)>       End datetime of observation i.e (2021,1,8, 18, 0)


    yr1(int)                                 Start year of observation i.e (2021)
    mn1(int)                                 Start Month of observation i.e (02)
    dy1(int)                                 Start day of observation i.e(15)
    hr1(int)                                 Start hour of observation (18)
    min1(int)                                Start minute of observation(22)
    
    yr2(int)                                 End year of observation i.e (2021)        
    mn2(int)                                 End Month of observation i.e (02)
    dy2(int)                                 End day of observation i.e(15)
    hr2(int)                                 End hour of observation (18)
    min2(int)                                End minute of observation(22)

    t1_lon (float)                           Lontitude(deg) of first telescope/observatory location i.e -70.81
    t1_lat (float)                           Latitude(deg) of first telescope/observatory location -30.16
    t1_el (float)                            Elevation(m) of first telescope/observatory location 2214.99
    t1_name(string)                          Name of first telescope/observatory location i.e 'CTIO'

    t2_lon (float)                           Lontitude(deg) of second telescope/observatory location i.e 149.06
    t2_lat (float)                           Latitude(deg) of second telescope/observatory location -31.27
    t2_el (float)                            Elevation(m) of second telescope/observatory location 1149.00
    t2_name(string)                          Name of second telescope/observatory location i.e 'KMTNET'



Options:
     -h,--help                               Show this screen
     --nt1=<string>                          Night_only_obs1= 'y' [default: 'y']
     --nt2=<string>                          Night_only_obs2= 'y' [default: 'y']
     --nra=<int>                              Number of steps of RA [default: 1]
     --ndec=<int>                             Number of steps of Dec [default: 1]
     --co_vis_type=<string>                  Type of co-vis returned sum or max[default:'sum']
     --airmass_low=<int>                     Lower limit of airmasses [default: 1]
     --airmass_hi=<int>                      Upper limit of airmasses [default: 3]
     --version                               Show version.   
    -o SAVELOC, --out SAVELOC                Saved output as [default: ./]






Examples:
python co-vis_skyplotter.py -h --help
python co-vis_skyplotter.py ['no']['no']['max'][nra=30][ndec=20](2021,1, 7, 18, 0)(2021,1,8, 18, 0)<time_interval=20><t1_lon> <t1_lat> <t1_el> <t1_name><t2_lon> <t2_lat> <t2_el> <t2_name>

python co-vis_skyplotter.py (2021,1, 7, 18, 0)(2021,1,8, 18, 0)20 t1_lon t1_lat t1_el t1_name t2_lon t2_lat t2_el t2_name

python co-vis_skyplotter.py [co_vis_type='max'][--airmass_low=2][--airmass_hi=4]start_obs=<<datetime(2021,1, 7, 18, 0)> <end_obs = <datetime(2021,1,8, 18, 0)><time_interval=20><t1_lon> <t1_lat> <t1_el> <t1_name><t2_lon> <t2_lat> <t2_el> <t2_name>

python co-vis_skyplotter.py [nt1='no'][nt2='no][co_vis_type='max'][--airmass_low=2][--airmass_hi=4](2021,1, 7, 18, 0)(2021,1,8, 18, 0)20 -70.81 -30.16 2214.99 CTIO 149.06 -31.27 1149.00 KMTNET

python co-vis_skyplotter.py ['no']['no']['max'] (2021,1, 7, 18, 0) (2021,1,8, 18, 0) 20 -70.81 -30.16 2214.99 CTIO 149.06 -31.27 1149.00 KMTNET

python co-vis_skyplotter.py -o './home/user/plots' ['no']['no']['max'] (2021,1, 7, 18, 0) (2021,1,8, 18, 0) 20 -70.81 -30.16 2214.99 CTIO 149.06 -31.27 1149.00 KMTNET
'''




import docopt



##############################################################
####################### Main Function ########################
##############################################################





#load third party dependencies  [requirements- Astropy, pandas, numpy, colorama,cv2]

from astropy.time import Time
import pandas as pd

from astropy.coordinates import EarthLocation
from astroplan import Observer,FixedTarget
from astroplan.plots import plot_airmass,plot_altitude
import astropy.units as u
from astropy.coordinates import SkyCoord

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import colorama
from colorama import Fore, Style
import cv2


pd.set_option("display.max_rows", None, "display.max_columns", None)





#############################################################################################################

#Specify paramenters
#Co-vis type input 'max' for maximum co-vis or 'sum' for sum of all co-vis 

save_location= '/home/andrew/src/Transient-Scheduler/scripts/'  #Input location to save the plot


#'''
#Input sum or 'max'. When there are more than one occurence of co-visibility time,
#max will return the bigger co-vis time of all co-vis sections, sum will return the sum of all the co-vis times
#from all sections
#'''



#co_vis_type='sum' 


#RA_intervals=30
#DEC_intervals=20

#time_interval=20  #Specify time intervals

#low_airmasslim=1  #Lower limit of airmass
#high_airmasslim=3 #Upper limit of airmass


#start_date = datetime(2021,1, 7,    18, 0)  #Input START date and time of observation (format: YYYY,M,D,  H,M)
#end_date = datetime(2021, 1, 8,     18, 0)      #Input END date and time of observation (format: YYYY,M,D,  H,M)


#Observatory 1 parameters

#Obs1_longitude=-70.81500000000001       #lontitude of observatory 1 
#Obs1_latitude=-30.165277777777778       #latitude of observatory 1 
#obs1_elevation=2214.9999999993697       #elevation of observatory 1
#obs1_name='CTIO'                        #name of observatory 1


#Night only observatory or not for observatory 1 
#night_only_obs1= 'y'# 'y' for yes or 'n'   


#Observatory 2 parameters

#obs2_longitude=149.06119444444445   #lontitude of observatory 1
#obs2_latitude=-31.273361111111104   #latitude of observatory 1 
#obs2_elevation=1149.0000000015516   #elevation of observatory 1
#obs2_name='KMTNET'                  #name of observatory 1

#Night only observatory or not for observatory 2 
#night_only_obs2= 'n' #'y' for yes or 'n' for no     




##############################################################################################################

# Set up Observatories


obs1_coordinates = EarthLocation.from_geodetic(obs1_longitude*u.deg,obs1_latitude*u.deg,obs1_elevation*u.m)
obs1= Observer(location=obs1_coordinates, name=obs1_name, timezone='America/Santiago')
print(obs1)


obs2_coordinates = EarthLocation.from_geodetic(obs2_longitude*u.deg,obs2_latitude*u.deg,obs2_elevation*u.m)
obs2 = Observer(location=obs2_coordinates, name=obs2_name, timezone='Australia/Sydney')
print(obs2)






###########################################################################################################
#RA&DEC 
ra=np.array(range(0, 361, RA_intervals))

dec=np.array(range(-90, 91, DEC_intervals))
x,y=np.meshgrid(ra,dec)

bb = np.zeros(np.shape(x))
print(ra)
print(dec)

###########################################################################################################
#Time intervals
from datetime import datetime, timedelta

def daterange(start_date, end_date):
    delta = timedelta(minutes=time_interval)
    while start_date < end_date:
        yield start_date
        start_date += delta
        

times=[]
for single_date in daterange(start_date, end_date):
    times.append(single_date.strftime("%Y-%m-%d %H:%M:%S"))
times1=np.array(times)

times1=times1.astype(("datetime64[ns]"))
delta=end_date-start_date
total_mins = delta.total_seconds() / 60
print(total_mins)                     #Total minutes for observation 


############################################################################################################
#Loop through all ra and dec

for ii,r in enumerate(ra):
    for jj,d in enumerate(dec):
                
        target_coord = SkyCoord(ra=r*u.deg, dec=d*u.deg)
        target = FixedTarget(coord=target_coord, name="source")
        
        
        #Airmasses at observatory 1
        
        airmass_obs1=obs1.altaz(times, target).secz          #calculate airmass for observatory 1
        masked_airmass_obs1 = np.ma.array(airmass_obs1, mask=airmass_obs1 < 1)#mask airmasses for observatory 1

        
        #Airmasses at observatory2
        
        airmass_obs2=obs2.altaz(times,target).secz   #calculate airmass for observatory 2
        masked_airmass_obs2 = np.ma.array(airmass_obs2, mask=airmass_obs2 < 1)#mask airmasses for observatory 2
        
        xc=obs1.is_night(times,horizon= -12*u.deg)  #Times that it is night time at observatory 1 from time list 
        cc=obs2.is_night(times,horizon= -12*u.deg)  #Times that it is night time at observatory 2 from time list 
        
        
        
        
        # For the case night only observation for observatory 1 and 2 
        if night_only_obs1=='y'and night_only_obs2=='y':
            print(f'Night only observation for BOTH Observatory {obs1.name.upper()} and {obs2.name.upper()}')

            dk={'datetimes':times,'night_obs1':xc,'night_obs2':cc,
                'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2} #Create dictionary of values 

            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 
            
            # Dropping NaN values but maintaining indexes in case there are NANs

            df=df.dropna()
            df.reset_index(drop=True, inplace=True)
            #print(df)

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([(df['night_obs1'] == True) & (df['night_obs2'] == True)&df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {0} mins{Style.RESET_ALL}')
                    bb[jj,ii]=0
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                    
                    if co_vis_type=='sum':
                        
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        
                        bb[jj,ii] = bgr_covis 
                        
        #########################################################################################


        # For the case night only observation for observatory 1 but not required for observatory 2
        elif night_only_obs1=='y'and night_only_obs2=='n':
            print(f'Night only observation for {obs1.name.upper()} observatory only')

            dk={'datetimes':times,'night_obs1':xc,'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2}
            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 

            df=df.dropna()
            df.reset_index(drop=True, inplace=True)
            

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([(df['night_obs1'] == True) &df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                com_vals
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    fullmins=0
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')  
                    if co_vis_type=='sum':
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        bb[jj,ii] = bgr_covis 
        ####################################################################################################
        
        
        #For the case night only observation for observatory 2 but not required for observatory 1
        elif night_only_obs1=='n'and night_only_obs2=='y':

            print(f'Night only observation for {obs2.name.upper()} observatory only')

            dk={'datetimes':times,'night_obs2':cc,'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2}
            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 
             # Dropping NaN values but maintaining indexes in case there are NANs

            df=df.dropna()#Checking the dataframe
            df.reset_index(drop=True, inplace=True)
            #print(df)

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([(df['night_obs2'] == True)&df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                com_vals
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    fullmins=0
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    
                    if co_vis_type=='sum':
                        print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {bgr_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = bgr_covis 
                        
         #########################################################################################################
        
        # For the case night only observation for observatory 1 and 2 is not required 
        
        elif night_only_obs1=='n'and night_only_obs2=='n':
            print(f'Night only observation not required')
            dk={'datetimes':times,'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2}
            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 
             # Dropping NaN values but maintaining indexes in case there are NANs

            df=df.dropna()#Checking the dataframe
            df.reset_index(drop=True, inplace=True)
            #print(df)

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                com_vals
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    fullmins=0
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                     
                    if co_vis_type=='sum':
                        print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        bb[jj,ii] = bgr_covis 
        
        
        
        
        
        
        
        
      



print((bb)) #print the array of co-vis times
covis=pd.DataFrame(bb)
covis.to_csv('covistimes.csv')

# Plot

fig, axs = plt.subplots(figsize=(20,10))
c=plt.pcolor(x,y,bb,shading='auto', edgecolors='k', linewidths=0.4)
fig.colorbar(c, ax=axs,label='Co-visibilty time [Mins]')
plt.title(f"Co-observability times of {obs1.name.upper()} and {obs2.name} from  {start_date}  to "\
          
          f" {end_date}  for total of {total_mins} mins",fontsize=15)
          
plt.ylim(-90,90)
plt.xlim(0,360)
plt.xlabel('RA [degrees]')
plt.ylabel('DEC [degrees]')
plt.savefig(save_location+f"Co-vis times of {obs1.name.upper()} and {obs2.name} from  {start_date}  to "\
          
          f" {end_date}  for total of {total_mins} mins.png")
plt.show()







if __name__ == "__main__":

    
    arguments       = docopt.docopt(__doc__)
    
    time_interval   =arguments[<int(time_interval)>]
    
    yr1=arguments[<yr1>]
    mn1=arguments[<mn1>]
    dy1=arguments[dy1]
    hr1=arguments[hr1]
    min1=arguments[min1]
    
    
    yr2=arguments[<yr2>]
    mn2=arguments[<mn2>]
    dy2=arguments[<dy2>]
    hr2=arguments[<hr2>]
    min2=arguments[min2]
    
    co_vis_type=arguments[str('--co_vis_type=<string>')] #co-vis returned sum or max[default:'sum']
    
    
    
    
    
    
    start_obs=<datetime(yr1,mn1, dy1,hr1,min1)>   #Start datetime of observation i.e (2021,1, 7, 18, 0)

    
    end_obs = <datetime(yr2,mn2,dy2, hr2,min2)>       #End datetime of observation i.e (2021,1,8, 18, 0)


    obs1_longitude =arguments[float(t1_lon)]                        
    obs1_latitude =arguments[float(t1_lat)]                         
    obs1_elevation=arguments[float(t1_el)]                            
    obs1_name =arguments[str(t1_name)]
    
    
    obs2_longitude =arguments[float(t2_lon)]                        
    obs2_latitude =arguments[float(t2_lat)]                         
    obs2_elevation =arguments[float(t2_el)]                            
    obs2_name =arguments[str(t2_name)]                           

       
    verbose = arguments['--verbose']
    night_only_obs1=arguments[' --nt1=<string>']                         
    night_only_obs1=arguments[' --nt2=<string>']                        
    RA_intervals=arguments['--nra=<int>']                            
    DEC_intervals=arguments['--ndec=<int>']                       
     
    
    low_airmasslim=['--airmass_low=<int>']                    
    
    high_airmasslim=['--airmass_hi=<int>']                     
        
    -o SAVELOC, arguments['--out']           
    
##############################################################################################################

# Set up Observatories


obs1_coordinates = EarthLocation.from_geodetic(obs1_longitude*u.deg,obs1_latitude*u.deg,obs1_elevation*u.m)
obs1= Observer(location=obs1_coordinates, name=obs1_name, timezone='America/Santiago')
print(obs1)


obs2_coordinates = EarthLocation.from_geodetic(obs2_longitude*u.deg,obs2_latitude*u.deg,obs2_elevation*u.m)
obs2 = Observer(location=obs2_coordinates, name=obs2_name, timezone='Australia/Sydney')
print(obs2)






###########################################################################################################
#RA&DEC 
ra=np.array(range(0, 361, RA_intervals))

dec=np.array(range(-90, 91, DEC_intervals))
x,y=np.meshgrid(ra,dec)

bb = np.zeros(np.shape(x))
print(ra)
print(dec)

###########################################################################################################
#Time intervals
from datetime import datetime, timedelta

def daterange(start_date, end_date):
    delta = timedelta(minutes=time_interval)
    while start_date < end_date:
        yield start_date
        start_date += delta
        

times=[]
for single_date in daterange(start_date, end_date):
    times.append(single_date.strftime("%Y-%m-%d %H:%M:%S"))
times1=np.array(times)

times1=times1.astype(("datetime64[ns]"))
delta=end_date-start_date
total_mins = delta.total_seconds() / 60
print(total_mins)                     #Total minutes for observation 


############################################################################################################
#Loop through all ra and dec

for ii,r in enumerate(ra):
    for jj,d in enumerate(dec):
                
        target_coord = SkyCoord(ra=r*u.deg, dec=d*u.deg)
        target = FixedTarget(coord=target_coord, name="source")
        
        
        #Airmasses at observatory 1
        
        airmass_obs1=obs1.altaz(times, target).secz          #calculate airmass for observatory 1
        masked_airmass_obs1 = np.ma.array(airmass_obs1, mask=airmass_obs1 < 1)#mask airmasses for observatory 1

        
        #Airmasses at observatory2
        
        airmass_obs2=obs2.altaz(times,target).secz   #calculate airmass for observatory 2
        masked_airmass_obs2 = np.ma.array(airmass_obs2, mask=airmass_obs2 < 1)#mask airmasses for observatory 2
        
        
        xc=obs1.is_night(times)  #Times that it is night time at observatory 1 from time list 
        cc=obs2.is_night(times)  #Times that it is night time at observatory 2 from time list 
        
        
        
        
        
        # For the case night only observation for observatory 1 and 2 
        if night_only_obs1=='y'and night_only_obs2=='y':
            print(f'Night only observation for BOTH Observatory {obs1.name.upper()} and {obs2.name.upper()}')

            dk={'datetimes':times,'night_obs1':xc,'night_obs2':cc,
                'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2} #Create dictionary of values 

            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 
            
            # Dropping NaN values but maintaining indexes in case there are NANs

            df=df.dropna()
            df.reset_index(drop=True, inplace=True)
            #print(df)

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([(df['night_obs1'] == True) & (df['night_obs2'] == True)&df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {0} mins{Style.RESET_ALL}')
                    bb[jj,ii]=0
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                    
                    if co_vis_type=='sum':
                        
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        
                        bb[jj,ii] = bgr_covis 
                        
        #########################################################################################


        # For the case night only observation for observatory 1 but not required for observatory 2
        elif night_only_obs1=='y'and night_only_obs2=='n':
            print(f'Night only observation for {obs1.name.upper()} observatory only')

            dk={'datetimes':times,'night_obs1':xc,'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2}
            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 

            df=df.dropna()
            df.reset_index(drop=True, inplace=True)
            

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([(df['night_obs1'] == True) &df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                com_vals
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    fullmins=0
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')  
                    if co_vis_type=='sum':
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        bb[jj,ii] = bgr_covis 
        ####################################################################################################
        
        
        #For the case night only observation for observatory 2 but not required for observatory 1
        elif night_only_obs1=='n'and night_only_obs2=='y':

            print(f'Night only observation for {obs2.name.upper()} observatory only')

            dk={'datetimes':times,'night_obs2':cc,'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2}
            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 
             # Dropping NaN values but maintaining indexes in case there are NANs

            df=df.dropna()#Checking the dataframe
            df.reset_index(drop=True, inplace=True)
            #print(df)

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([(df['night_obs2'] == True)&df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                com_vals
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    fullmins=0
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    
                    if co_vis_type=='sum':
                        print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {bgr_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = bgr_covis 
                        
         #########################################################################################################
        
        # For the case night only observation for observatory 1 and 2 is not required 
        
        elif night_only_obs1=='n'and night_only_obs2=='n':
            print(f'Night only observation not required')
            dk={'datetimes':times,'obs1_airmass':masked_airmass_obs1,'obs2_airmass':masked_airmass_obs2}
            df1=pd.DataFrame(data=dk)

            df=df1.round(3 )# Rounding up values 
             # Dropping NaN values but maintaining indexes in case there are NANs

            df=df.dropna()#Checking the dataframe
            df.reset_index(drop=True, inplace=True)
            #print(df)

            covis_list=(df[df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])


            if len(covis_list)==0 or len(df)==0:
                mins=0
                print(f'{Fore.RED}No Co-Vis:Total time co-observable for {target_coord.ra.deg} and {target_coord.dec.deg} in minutes: {mins} mins{Style.RESET_ALL}')
                bb[jj,ii] = mins 
            else:   
                com_vals=np.array([df['obs1_airmass'].between(low_airmasslim,high_airmasslim) & df['obs2_airmass'].between(low_airmasslim,high_airmasslim)])
                com_vals
                fb=com_vals[0][:]
                p=int((len(fb))*0.2)
                nit=np.zeros(p,dtype=bool)

                yy=np.append(nit,fb) #Adding false to the beginning of the list 
                tt=np.append(yy,nit)
                gg=1*tt #converting to binary
                ff=np.zeros(len(gg)) #Generating zeros
                vls=np.array([ff,tt,ff])

                vls=vls.astype(np.uint8)
                (thresh, gray) = cv2.threshold(vls, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                ret, labels = cv2.connectedComponents(gray)
                #plt.figure(figsize=(5,5))
                #plt.imshow(vls)
                k=np.unique(labels, return_counts=True)

                cb=k[1][1:]
                if len(cb)==0:
                    fullmins=0
                    print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                     
                    if co_vis_type=='sum':
                        print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        bb[jj,ii] = bgr_covis 
        
        
        
        
        
        
        
        
      



print((bb)) #print the array of co-vis times
covis=pd.DataFrame(bb)
covis.to_csv('covistimes.csv')

# Plot

fig, axs = plt.subplots(figsize=(20,10))
c=plt.pcolor(x,y,bb,shading='auto', edgecolors='k', linewidths=0.4)
fig.colorbar(c, ax=axs,label='Co-visibilty time [Mins]')
plt.title(f"Co-observability times of {obs1.name.upper()} and {obs2.name} from  {start_date}  to "\
          
          f" {end_date}  for total of {total_mins} mins",fontsize=15)
          
plt.ylim(-90,90)
plt.xlim(0,360)
plt.xlabel('RA [degrees]')
plt.ylabel('DEC [degrees]')
plt.savefig(save_location+f"Co-vis times of {obs1.name.upper()} and {obs2.name} from  {start_date}  to "\
          
          f" {end_date}  for total of {total_mins} mins.png")
plt.show()

