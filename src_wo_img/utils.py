'''
Author: Aman
Date: 2021-11-15 10:56:51
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-11-15 11:10:27
'''


import datetime


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))