
#!/usr/bin/env python3

'''Plots the phases of the moon from  given date range. Moon phases are calculted based on Astral package.
See https://astral.readthedocs.io/en/latest/index.html for further info.

moon.phase() Calculates the phase of the moon on the specified date.

    Args:
        date: The date to calculate the phase for. Dates are always in the UTC timezone.
              If not specified then today's date is used.

    Returns:
        A number designating the phase.

        ============  ==============
        0 .. 6.99     New moon
        7 .. 13.99    First quarter
        14 .. 20.99   Full moon
        21 .. 27.99   Last quarter
        ============  ==============

'''

#importing modules Time,matplotlib and datetime 

from astropy.time import Time
import matplotlib.pyplot as plt
import datetime
from astral import moon



#Setting date range 

#Start date
start = input('Enter START date in YYYY-MM-DD format: ')
year, month, day = map(int, start.split('-'))
start_date = datetime.date(year, month, day)

#End date
end = input('Enter END date in YYYY-MM-DD format: ')
year2, month2, day2 = map(int, end.split('-'))
end_date = datetime.date(year2, month2, day2)


#Manual entry
#start_date = datetime.date(2020, 11, 20)
#end_date   = datetime.date(2021, 1, 30)


dates = [ start_date + datetime.timedelta(n) for n in range(int ((end_date - start_date).days))]
#dates


#Get moonphases 
moonphases=[]
for i in dates:
    moonphases.append(moon.phase(i))

#Plot moonphases 


save_option=input('Do you want to save the plot? input: y or n :')
if save_option=='y':

    plt.figure(figsize=(15,8))
    ax = plt.gca()
    plt.plot(dates,moonphases,linewidth=2.5,color='b',label='Moonphases')
    ax.axhline(y=6.99,color='g',linestyle='--',linewidth=2,alpha=0.9)
    ax.axhline(y = 14, color = 'r', linestyle = '--',linewidth=2,alpha=0.9) 
    ax.axhline(y= 20.99, color = 'r', linestyle = '--',linewidth=2,alpha=0.9) 
    ax.axhline(y = 27.99, color = 'b', linestyle = '--',alpha=0.9)

#fills
    ax.axhspan(0, 6.99, facecolor='cyan', alpha=0.25,label='Newmoon')
    ax.axhspan(7, 13.99, facecolor='violet', alpha=0.25,label='First quarter')
    ax.axhspan(14, 20.99, facecolor='gold', alpha=0.25,label='Fullmoon')
    ax.axhspan(21, 27.99, facecolor='tomato', alpha=0.25,label='Last quarter')
    ax.grid()
    plt.title('Moon phases from '+str(dates[0])+' to '+str(dates[-1])+'')
    plt.legend(loc=3,facecolor="w")
    plt.savefig('Moon phases from '+str(dates[0])+' to '+str(dates[-1])+'.png')
    plt.show()
    

elif save_option=='n':

    plt.figure(figsize=(15,8))
    ax = plt.gca()
    plt.plot(dates,moonphases,linewidth=2.5,color='b',label='Moonphases')
    ax.axhline(y=6.99,color='g',linestyle='--',linewidth=2,alpha=0.9)
    ax.axhline(y = 14, color = 'r', linestyle = '--',linewidth=2,alpha=0.9) 
    ax.axhline(y= 20.99, color = 'r', linestyle = '--',linewidth=2,alpha=0.9) 
    ax.axhline(y = 27.99, color = 'b', linestyle = '--',alpha=0.9)

#fills
    ax.axhspan(0, 6.99, facecolor='cyan', alpha=0.25,label='Newmoon')
    ax.axhspan(7, 13.99, facecolor='violet', alpha=0.25,label='First quarter')
    ax.axhspan(14, 20.99, facecolor='gold', alpha=0.25,label='Fullmoon')
    ax.axhspan(21, 27.99, facecolor='tomato', alpha=0.25,label='Last quarter')
    ax.grid()
    plt.title('Moon phases from '+str(dates[0])+' to '+str(dates[-1])+'')
    plt.legend(loc=3,facecolor="w")
    plt.show()

    print('Plot not saved')



