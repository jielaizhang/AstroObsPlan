'''
CO-VISIBILTY CALCULATOR AND PLOTTER FOR SEVERAL SKY POSITIONS

RA is spaced from 0 to 360 , Dec is from 90 to -90
Takes in number of divisions which determin the intervals of RA and DEC.
Time interval is specified
lower limit and upper limit

Input start and end date and time of observation

Input properties(Longitude, latitiude, elevation, observatory name, and if observatio should be carried out night only)
of observatory 1 and 2 ,
Input location to save final plot

'''





#load third party modules [requirements- Astropy, pandas, numpy, colorama,cv2]

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

save_location= '/Users/jielaizhang/Desktop/'  #Input location to save the plot


#'''
#Input sum or 'max'. When there are more than one occurence of co-visibility time,
#max will return the bigger co-vis time of all co-vis sections, sum will return the sum of all the co-vis times
#from all sections
#'''


debug=False
co_vis_type='max' 


RA_intervals=10
DEC_intervals=10

time_interval=20  #Specify time intervals

low_airmasslim=1  #Lower limit of airmass
high_airmasslim=2.5 #Upper limit of airmass


start_date = datetime(2022,4,10,    00, 1)       #Input START date and time of observation (format: YYYY,M,D,  H,M)
end_date = datetime(2022, 4, 10,    23, 59)      #Input END date and time of observation (format: YYYY,M,D,  H,M)

#Observatory 1 parameters






#obs1_longitude=-0.30510      #lontitude of observatory 1 
#obs1_latitude=5.75044     #latitude of observatory 1 
#obs1_elevation=71      #elevation of observatory 1
#obs1_name='GRAO'                        #name of observatory 1
#night_only_obs1= 'n' #'y'
#timezone_obs1= 'Africa/Accra'

#obs1_longitude=-79.83983 
#obs1_latitude=-38.433121
#obs1_elevation=805
#obs1_name='GBT'

#obs2_longitude=-119.624
#obs2_latitude=49.321
#obs2_elevation=545
#obs2_name='CHIME'
#night_only_obs2= 'n' #'n'
#timezone_obs2= 'Canada/Pacific'


#obs1_longitude=-17.88508
#obs1_latitude= 28.75728
#obs1_elevation=2382
#obs1_name='NordicOT'

#Night only observatory or not for observatory 1 
#night_only_obs1= 'y'# 'y' for yes or 'n' 

#Observatory 2 parameters

#obs1_longitude=116.637   #lontitude of observatory 2
#obs1_latitude=-26.696  #latitude of observatory 2 
#obs1_elevation=377.83   #elevation of observatory 2
#obs1_name='ASKAP'      #name of observatory 2
#timezone_obs1= 'Australia/Perth'
#night_only_obs1= 'n' #'y' 

obs1_longitude=-70.81500000000001       #lontitude of observatory 1 
obs1_latitude=-30.165277777777778       #latitude of observatory 1 
obs1_elevation=2214.9999999993697       #elevation of observatory 1
obs1_name='CTIO'
night_only_obs1= 'y' #'y'
timezone_obs1= 'America/Santiago'

obs2_longitude=-70.81500000000001       #lontitude of observatory 1 
obs2_latitude=-30.165277777777778       #latitude of observatory 1 
obs2_elevation=2214.9999999993697       #elevation of observatory 1
obs2_name='CTIO'
night_only_obs2= 'y' #'y'
timezone_obs2= 'America/Santiago'

#obs2_longitude=-17.88508
#obs2_latitude= 28.75728
#obs2_elevation=2382
#obs2_name='Nordic-OT'
#timezone_obs2= 'GMT'

#obs2_longitude=-155.47, 
#obs2_latitude=19.8253
#obs2_elevation=4204
#obs2_name='CFHT'
#timezone_obs2= 'US/Hawaii'
#night_only_obs2= 'n' #'y' 

#obs2_longitude=27.68539
#obs2_latitude=-31.273361111111104
#obs2_elevation=1382
#obs2_name='HartRaO'
#night_only_obs2= 'n' #'y' for yes or 'n' for no  
#timezone_obs2= 'Africa/Johannesburg'

#obs2_longitude=149.06119444444445
#obs2_latitude=-31.273361111111104
#obs2_elevation=1149.0000000015516
#obs2_name='KMTNET'
#Night only observatory or not for observatory 2 
#night_only_obs2= 'y' #'y' for yes or 'n' for no     
#timezone_obs2= 'Australia/Perth'


  

##############################################################################################################

# Set up Observatories


obs1_coordinates = EarthLocation.from_geodetic(obs1_longitude*u.deg,obs1_latitude*u.deg,obs1_elevation*u.m)
obs1= Observer(location=obs1_coordinates, name=obs1_name, timezone=timezone_obs1)
print(obs1)


obs2_coordinates = EarthLocation.from_geodetic(obs2_longitude*u.deg,obs2_latitude*u.deg,obs2_elevation*u.m)
obs2 = Observer(location=obs2_coordinates, name=obs2_name, timezone=timezone_obs2)
print(obs2)

#ctio = Observer.at_site("ctio", timezone = 'America/Santiago')  

#kmtnet= Observer.at_site("Siding Spring Observatory", name='KMTNET',timezone = 'Australia/Sydney')






###########################################################################################################
#RA&DEC 
ra=np.array(range(0, 361, RA_intervals))

dec=np.array(range(-90, 90, DEC_intervals))
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
                
        target_coord = SkyCoord(ra=r*u.deg, dec=d*u.deg,frame='icrs')
        target = FixedTarget(coord=target_coord, name="Target")
        
        
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
            if debug:
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
                if debug:
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
                    if debug:
                        print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {0} mins{Style.RESET_ALL}')
                    bb[jj,ii]=0
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)
                        

                    
                    
                    if co_vis_type=='sum':
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        #total_covis=bgr_covis
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {bgr_covis} mins{Style.RESET_ALL}')
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
                if debug:
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
                    if debug:
                        print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    if co_vis_type=='sum':
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        #total_covis=bgr_covis
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {bgr_covis} mins{Style.RESET_ALL}')
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
                if debug:
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
                    if debug:
                        print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                    if co_vis_type=='sum':
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        #total_covis=bgr_covis
                        if debug:
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
                if debug:
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
                    if debug:
                        print(f'{Fore.RED}No Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} Co-vis time: {fullmins} mins{Style.RESET_ALL}')
                    bb[jj,ii]=fullmins
                else: 

                    cvs=[]
                    for i in (cb):
                        cvs.append((time_interval*i))
                        bgr_covis=np.max(cvs)
                        total_covis=sum(cvs)

                     
                    if co_vis_type=='sum':
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {total_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = total_covis 
                    elif co_vis_type=='max':
                        #total_covis=bgr_covis
                        if debug:
                            print(f'{Fore.GREEN}Mulitiple ranges of Co-Vis time for ra={target_coord.ra.deg} and dec={target_coord.dec.deg} in minutes: {bgr_covis} mins{Style.RESET_ALL}')
                        bb[jj,ii] = bgr_covis 
        
        
        
        
        
        
        
      




# Plot

fig, axs = plt.subplots(figsize=(20,10))
c=plt.pcolor(x,y,bb,shading='auto', edgecolors='k', linewidths=0.4)
fig.colorbar(c, ax=axs,label='Co-visibilty time [Mins]')
plt.title(f"Co-observability times ({co_vis_type}) of {obs1.name.upper()} and {obs2.name.upper()} for {time_interval}min(s) intervals from  {start_date}  to "\
          
          f" {end_date}",fontsize=15)
          
plt.ylim(-90,90)
plt.xlim(0,360)
plt.xlabel('RA [degrees]')
plt.ylabel('DEC [degrees]')
plt.savefig(save_location+f"Co-vis times{co_vis_type} of {obs1.name.upper()} and {obs2.name.upper()} from  {start_date}  to "\
          
          f" {end_date}.png")
plt.show()

