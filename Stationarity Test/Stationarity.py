import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ctypes as ct
from typing import List, Union

MKTBUF = 2048   # Alloc for market info in chunks of this many records
                # This is not critical and can be any reasonable value

NGAPS = 11      # Number of gaps in analysis

def qsort(int first, int last, double* data)
{
    #fill in
}

def find_slope(int lookback, double* currPrice)
{
    #fill in    
}

def avgTrueRange(int lookback, double* high, double* low, double* close)
{
    #fill in   
}

def gap_analyse(int n,  double* x, double thresh, ngaps, int ngaps, int* gap_size, int* gap_count)
{
    #fill in
}

def main(int argc, char* argv[])
{
    #ints
    i, k, nprices, nind, lookback, bufcnt, *date, itemp, full_date, prior_date, year, month, ngaps, version, full_lookback, day = 0 #ints
    
    #arrays
    gap_size = np.empty(NGAPS-1)
    gap_count = np.empty(NGAPS)
    line = np.empty(256)
    filename = np.empty(4096)

    #ptrs
    charptr = ct.POINTER(ct.c_char)
    mktopen = ct.POINTER(ct.c_double)
    mkthigh = ct.POINTER(ct.c_double)
    mktlow = ct.POINTER(ct.c_double)
    mktclose = ct.POINTER(ct.c_double)
    trend = ct.POINTER(ct.c_double)
    trend_sorted = ct.POINTER(ct.c_double)
    volatility = ct.POINTER(ct.c_double)
    volatility_sorted = ct.POINTER(ct.c_double)

    #doubles
    trend_min, trend_max, trend_quantile, volatility_min, volatility_max, volatility_quantile, fractile = 0.0

    FILE *fp #handle file with python tools not ptrs

    #parse arguments here
    #-------

    bufcnt = MKTBUF
    print("Reading Market File")

    for(1)
    {
        #check got eof, or error reading file
        #parse dates from file
        #parse open from file
        #parse close from file
        #parse high from file
        #parse low from file
        #close file

        #Market data is readm initialize gap analysis 
        ngaps = NGAPS
        k = 1
        for (i=0 ; i<ngaps-1 ; i++) 
        {
            gap_size[i] = k
            k *= 2
        }

        #compute trend and find min, max and quantile
        nind = nprices - full_lookback + 1    // This many indicators

        trend.contents.value = 2 * nind
        if (bool(trend)) {
            print( "\n\nInsufficient memory.  Press any key..." )
            break
            }
        trend_sorted.contents.value = trend.contents.values + nind 

        trend_min = 1.e60 
        trend_max = -1.e60 
        for (i=0  i<nind  i++) {
            k = full_lookback - 1 + i 
            if (version == 0)
                trend[i] = find_slope ( lookback , close + k ) 
            else if (version == 1)
                trend[i] = find_slope ( lookback , close + k ) - find_slope ( lookback , close + k - lookback ) 
            else
                trend[i] = find_slope ( lookback , close + k ) - find_slope ( full_lookback , close + k ) 
            trend_sorted[i] = trend[i] 
            if (trend[i] < trend_min)
                trend_min = trend[i] 
            if (trend[i] > trend_max)
                trend_max = trend[i] 
            }

        qsortd ( 0 , nind-1 , trend_sorted ) 
        k = (int) (fractile * (nind+1)) - 1 
        if (k < 0)
            k = 0 
        trend_quantile = trend_sorted[k] 

        prin ( "\n\nTrend  min=%.4lf  max=%.4lf  %.3lf quantile=%.4lf",
                trend_min, trend_max, fractile, trend_quantile ) 
        }

}

if __name__ == '__main__':

    main()