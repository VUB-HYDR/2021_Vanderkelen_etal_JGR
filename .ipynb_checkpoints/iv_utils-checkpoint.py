"""
Python module with all utils and functions for analysing CESM output on cheyenne
"""

# --------------------------------------------------------------------
# Import modules 
# -------------------------------------------------------------------
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import regionmask
import cartopy.crs as ccrs
import warnings
import numpy as np

# --------------------------------------------------------------------
# Settings - define variables necessary for functions
# --------------------------------------------------------------------


# set directories
outdir = '/glade/scratch/ivanderk/'

# Define directory where processing is done -- subject to change
procdir =  '/glade/work/ivanderk/postprocessing/' 

# go to processing directory 
os.chdir(procdir)

# ignore all runtime warnings
warnings.filterwarnings('ignore')

# default settings -- can be changed within the functions  
# set individual case names for reference

case_res_ind   = 'f.FHIST.f09_f09_mg17.CTL.001'
case_nores_ind   = 'f.FHIST.f09_f09_mg17.NORES.001'
case   = 'f.FHIST.f09_f09_mg17.CTL.001'
block  = 'atm' 
stream = 'h0' 
n_ens  = 5 

# define start and end year
spstartyear = '1979'   # spin up start year 
startyear   = '1984'   # start year, spin up excluded
endyear     = '2014'   # last year of the simulation


# ---------------------------------------------------------------------
# Functions to open datasets
# ---------------------------------------------------------------------

def open_ds(var, case, stream=stream, block=block, lunit = 0):
   #""" open nc variable as dataset and interpolate lnd variables to atm grid """     
    tfreqs = {'h0' : 'month_1'                     , 'h1' : 'day_1'                            , 'h2' : 'month_1'}
    tspans = {'h0' : spstartyear+'01-'+endyear+'12', 'h1' : spstartyear+'0101-'+ endyear+'1231', 'h2' : spstartyear+'01-'+endyear+'12'} 

    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}

    # Define directory where timeseries data is stored
    tseriesdir = outdir + 'archive/' + case + '/' + block + '/proc/tseries/' + tfreqs[stream] + '/'
    
    # define filename
    fn = case + '.'+ model[block] + '.' + stream + '.' + var + '.' + tspans[stream] +'.nc'

    # check if variable timeseries exists and open variable as data array
    if not os.path.isfile(tseriesdir + fn):
        print(fn + ' does not exists in ')
        print(tseriesdir)
        return
    else: 
        
        # open the dataset
        ds = xr.open_dataset(tseriesdir+fn)
        
        # if h2 -> select specific land unit, defined by lunit argument
        if lunit > 0: 
            ds = ds.isel(lunit=lunit)
            
            # Overview of different lunits:
            # 1: soil (vegetated or bare soil landunit)
            # 2: crop (only for crop configuration)
            # 3: UNUSED
            # 4: land ice
            # 5: deep lake
            # 6: wetland
            # 7: urban tbd
            # 8: urban hd
            # 9: urban md
        
        # the lats of the atm and lnd grid differ with about E-7. 
        # therefore, interpolate land to atm grid to be able to work with exactly the same grids. 
        if block == 'lnd':
            ds_atm = open_ds('TREFHT',case=case, block='atm', stream = 'h0', lunit=0)
            ds = ds.interp_like(ds_atm)
     
            
    return ds


def extract_anaperiod(da, stream): 
    """Extract analysis period out of data-array (1900-2015)"""
    # number of spin up years
    nspinupyears = int(startyear) - int(spstartyear)
    
    if nspinupyears == 0 :
        # no spin up 
        da = da[:-1,:,:]
        
    elif stream == 'h1' : # this option still to test 
        # daily timesteps
        # last day of previous year is also saved in variable therefore add one
        nspinupdays = (nspinupyears * 365) + 1
    
        # exclude spin up year and last timestep ()
        da = da[nspinupdays:-1,:,:]

    elif stream =='xtrm': 
        # annual timesteps
        da = da[nspinupyears:-1,:,:]
        
    else: 
        # spin up with monthly timestep
        # first month of first year is not saved in variable therefore substract one
        nspinupmonths = (nspinupyears * 12) - 1
    
        # exclude spin up year and last timestep ()
        da = da[nspinupmonths:-1,:,:]

    return da

# open variable as data-array
def open_da(var, case=case, stream=stream, block=block, lunit=0):

    ds = open_ds(var, case, stream=stream, block=block, lunit=lunit)
    da = ds[var]
    
    # extract analysis period - not necessary
    da = extract_anaperiod(da, stream=stream)
    
    # convert hydrological variables (only if they are in the list)
    da = conv_hydrol_var(var, da)
    
    return da

# open average data array of all ensemble members
# possible to open all ensemble members, mean or stdev
def open_da_ens(var, case=case, n_ens = n_ens, stream=stream, block = block, lunit=0, mode='mean'):

    # loop over ensemble members
    for i in range(1,n_ens+1): 

        case_name = case+'.00'+str(i) 

        da = open_da(var, case=case_name, stream=stream, block=block, lunit=lunit)

        if i==1: 
            da_concat = da 
        else: 
            da_concat = xr.concat((da_concat, da), dim='ens_member')

    # different output options
    
    # return ensemble mean
    if mode == 'mean': 
        return da_concat.mean(dim='ens_member', keep_attrs='True')
    
    # standard deviation
    if mode == 'std': 
        return da_concat.std(dim='ens_member', keep_attrs='True')
    
    # the full ensemble with dim (ens_member)
    if mode == 'all':
        return da_concat


def open_da_delta(var, case, case_ref, stream=stream, block=block, lunit=0, ens=True, mode='mean'): 
    """ open and caluclate the difference between the ensemble means of two members """
    
    # Load the two datasets
    if ens == True:
        da_res = open_da_ens(var,case=case, stream=stream, block=block, lunit=lunit, mode = mode)
        da_ctl = open_da_ens(var,case=case_ref, stream=stream, block=block, lunit=lunit, mode = mode)     
    else: # open single simulation
        da_res = open_da(var,case=case, stream=stream, block=block, lunit=lunit)
        da_ctl = open_da(var,case=case_ref, stream=stream, block=block, lunit=lunit)

    # calculate difference and update attributes
    da_delta = da_res - da_ctl

    da_delta.attrs['long_name'] = '$\Delta$ '+ da_ctl.long_name
    da_delta.attrs['units'] = da_ctl.units
    da_delta.name = '$\Delta$ '+ da_ctl.name

    return da_delta


# open dataset of extremes
def open_da_xtrm(var,case=case, block=block):

    tspan = spstartyear+'01-'+endyear+'12'
    model = {'lnd' : 'clm2', 'atm' : 'cam', 'rof' : 'mosart'}

    savedir = outdir + 'postprocessing/extremes/f09_f09/'

    # define filename
    fn = case + '.'+ model[block] + '.' + var + '.' + tspan +'.nc'

    # check if variable timeseries exists and open variable as data array
    if not os.path.isfile(savedir + fn):
        print(fn + ' does not exist')
        return
    else: 
        ds = xr.open_dataset(savedir+fn)
        
    da = ds[var]
    
    # not for da wihtout time (eg. TX90)
    if len(da.dims) >2:
    # extract analysis period
        da = extract_anaperiod(da, 'xtrm')

    # convert hydrological variables (only if they are in the list)
    da = conv_hydrol_var(var, da)
    
    return da

def open_da_xtrm_ens(var, case=case, n_ens = n_ens, block = block, mode='mean'):

    # loop over ensemble members
    for i in range(1,n_ens+1): 

        case_name = case+'.00'+str(i) 

        da = open_da_xtrm(var,case=case_name, block=block)

        if i==1: 
            da_concat = da 
        else: 
            da_concat = xr.concat((da_concat, da), dim='ens_member')

    # different output options
    
    # return ensemble mean
    if mode == 'mean': 
        return da_concat.mean(dim='ens_member', keep_attrs='True')
    
    # standard deviation
    if mode == 'std': 
        return da_concat.std(dim='ens_member', keep_attrs='True')
    
    # the full ensemble with dim (ens_member)
    if mode == 'all':
        return da_concat

# fucntion to open (and calculate) delta of extreme 
def open_da_delta_xtrm(var, case, case_ref, block=block, ens=True, mode='mean'): 

    # Load the two datasets
    if ens == True:
        da_res = open_da_xtrm_ens(var,case=case    , block=block, mode = mode)
        da_ctl = open_da_xtrm_ens(var,case=case_ref, block=block, mode = mode)
   
    else: # open single simulation
        # Load the two datasets
        da_res = open_da_xtrm(var,case=case, block=block)
        da_ctl = open_da_xtrm(var,case=case_ref, block=block)
  
    # calculate difference and update attributes
    da_delta = da_res - da_ctl

    da_delta.attrs['long_name'] = '$\Delta$ '+ da_ctl.long_name
    da_delta.attrs['units'] = da_ctl.units
    da_delta.name = '$\Delta$ '+ da_ctl.name

    return da_delta

def open_da_delta_pctl(var, case_res, case_nores):
    """Open percentile da (e.g. TX10, R05 etc) """
    # calculate delta to plot
    da_res = open_da_xtrm(var, case_res, block='atm')
    da_nores = open_da_xtrm(var, case_nores, block='atm')
    da_delta = da_res - da_nores

    da_delta.attrs['long_name'] = '$\Delta$ '+ da_res.long_name
    da_delta.attrs['units'] = da_res.units
    da_delta.name = '$\Delta$ '+ da_res.name
    
    return da_delta


# ---------------------------------------------------------------------
# 2. Functions to perform calculations
# ---------------------------------------------------------------------

def calc_srex_mean(da):
    """calculate mean of every srex region"""
    mask = regionmask.defined_regions.srex.mask(da)

    da_srex = da.groupby(mask).mean('stacked_lat_lon')

    # extract the abbreviations and the names of the regions from regionmask and add as attributes to data-array
    abbrevs = regionmask.defined_regions.srex[da_srex.region.values].abbrevs
    names = regionmask.defined_regions.srex[da_srex.region.values].names
    da_srex.coords['abbrevs'] = ('region', abbrevs)
    da_srex.coords['names'] = ('region', names)
    
    # set attributes
    da_srex.attrs['long_name'] = da.long_name
    da_srex.attrs['units'] = da.units
    
    return da_srex

def calc_seasmean(da, season):
    """calculate seasonal mean of data array, (season= 'JJA', 'SON', 'DJF' or 'JFM')"""
    da_season = da.groupby('time.season').mean().sel({'season':season})
    da_season.attrs['long_name']= season +' '+ da.long_name
    da_season.attrs['units']= da.units  

    return da_season

def get_resmask(threshold = 0):
    """ open reservoir mask (threshold in %)"""
    
    file = 'resmask_f09_f09_threshold_'+str(threshold)+'pct.nc'
    if not os.path.isfile(procdir + file):
        da_delta = open_da_delta('PCT_LANDUNIT',case_res_ind,case_nores_ind, block='lnd', ens = False)
        dpctlake = da_delta.sel(ltype = 4)[1,:,:]
        resmask = dpctlake > threshold
        resmask.to_dataset(name='resmask').to_netcdf(procdir+file)
    else: 
                                        
        resmask = xr.open_dataset(procdir+file)['resmask']
    
    return resmask

# open land mask (threshold in %)
def get_landmask():
    
    file = 'landmask_f09_f09.nc'
    if not os.path.isfile(procdir + file):
        
        ds = open_ds('PCT_LANDUNIT', case_res_ind, block='lnd')
        landmask = ds.landmask == 1
        landmask.to_dataset().to_netcdf(procdir+file)
    else: 
                                        
        landmask = xr.open_dataset(procdir+file)['landmask']
                                        
    return landmask

# get res mask with one neigbouring cell
def set_neigbourcell_true(resmask):

    [x_inds,y_inds] = np.where(resmask)
    resmask_neigb1 = np.zeros(np.shape(resmask), dtype=bool)

    for x,y in zip(x_inds,y_inds):
        if y+1 == np.shape(resmask)[1]:
            yp1 = 0
        else: 
            yp1= y+1

        resmask_neigb1[x+1,y] = True
        resmask_neigb1[x+1,y-1] = True
        resmask_neigb1[x+1,yp1] = True
        resmask_neigb1[x,y] = True
        resmask_neigb1[x,y-1] = True
        resmask_neigb1[x,yp1] = True
        resmask_neigb1[x-1,y] = True
        resmask_neigb1[x-1,yp1] = True
        resmask_neigb1[x-1,y-1] = True

    return resmask_neigb1

def get_resmask_1step():
    
    resmask = get_resmask()
    return set_neigbourcell_true(resmask)    

def get_resmask_2step():
    
    resmask = get_resmask()
    resmask_neigb1 = set_neigbourcell_true(resmask)
    return set_neigbourcell_true(resmask_neigb1)

def get_resmask_1step_nores():
    """1 step neigbouring cells without resmask """
    resmask = get_resmask()
    noresmask = np.logical_not(resmask)

    return set_neigbourcell_true(resmask)*noresmask

def get_resmask_2step_nores():
    """2 step neigbouring cells without resmask """

    resmask = get_resmask()
    resmask_neigb1 = set_neigbourcell_true(resmask)
    noresmask = np.logical_not(resmask)
    return set_neigbourcell_true(resmask_neigb1)*noresmask

# calculate mean (of all) 
def calc_mean(da):
    # check if input da is already ymean, otherwise do calculation 
    if len(da) < 500: 
        da_mean = da.mean(dim=('lon','lat')).mean('year')
    else: 
        da_mean = da.mean(dim=('lon','lat')).mean('time')
    return da_mean



# check if in hydrol variable list and if so, convert units
def conv_hydrol_var(var, da_in, hydrol_vars=False):
    
    # hydrological variables not defined when function is used - use predefined list
    if not hydrol_vars: 
        hydrol_vars = ['PRECT','PRECMC', 'PRECC','PRECL','Rx1day', 'QRUNOFF']
        
    if var in hydrol_vars: 
        da_out = conv_m_s_to_mm_day(da_in)
        
    else: 
        da_out = da_in
        
    return da_out

# conversion function
def conv_m_s_to_mm_day(da_in):

    if not da_in.attrs['units'] == 'mm/day':
        da_out = da_in * 86400000  
        # update attributes and change units
        da_out.attrs= da_in.attrs
        da_out.attrs['units'] = 'mm/day' 
    else: 
        da_out = da_in
        
    return da_out


# ---------------------------------------------------------------------
# 3. Functions to plot
# ---------------------------------------------------------------------

def set_plot_param():
    """Set my own customized plotting parameters"""
    
    import matplotlib as mpl
    mpl.rc('axes',edgecolor='grey')
    mpl.rc('axes',labelcolor='dimgrey')
    mpl.rc('xtick',color='dimgrey')
    mpl.rc('xtick',labelsize=12)
    mpl.rc('ytick',color='dimgrey')
    mpl.rc('ytick',labelsize=12)
    mpl.rc('axes',titlesize=14)
    mpl.rc('axes',labelsize=12)
    mpl.rc('legend',fontsize='large')
    mpl.rc('text',color='dimgrey')


def plot_ts(da):
    """ plot timeseries (of spatial mean) """
    da_ts = da.mean(dim=('lon','lat'))
    da_ts.plot()
    plt.title(da.long_name + '('+da.units+')')
    plt.xlim(da_ts.time[0].values,da_ts.time[-1].values)


def plot_ymean(da):
    """calculate and plot annual mean timeseries """
    # check if input da is already ymean, otherwise do calculation 
    if len(da) < 500: 
        da_ymean = da.mean(dim=('lon','lat'))
    else: 
        da_ymean = da.mean(dim=('lon','lat')).groupby('time.year').mean('time')
    
    xlims = (da_ymean.year[0].values,da_ymean.year[-1].values)
    da_tseries = da_ymean.plot(xlim=xlims)
    plt.title(da.long_name, pad=5)
    plt.ylabel(da.name+' [' + da.units + ']')
    #plt.plot([da_ymean.year[0],da_ymean.year[-1]], [0,0], linewidth=1, color='gray')

# calculate and plot annual sum timeseries 
def plot_ysum(da):
    
    da_ymean = da.sum(dim=('lon','lat')).groupby('time.year').mean('time')
    xlims = (da_ymean.year[0].values,da_ymean.year[-1].values)
    da_tseries = da_ymean.plot(xlim=xlims)
    plt.title(da.long_name+' [' + da.units + ']' )


# plot timmean per selected region
def plot_yts_sel_regions(da_to_mask, selected_regions):
    mask = regionmask.defined_regions.srex.mask(da_to_mask)
    
    # annual means are already calculated
    if len(da_to_mask) < 50:
        da_mask_ts = da_to_mask.groupby(mask).mean('stacked_lat_lon').sel(region=selected_regions)
    else: 
        da_mask_ts = da_to_mask.groupby(mask).mean('stacked_lat_lon').groupby('time.year').mean().sel(region=selected_regions)

    # add abbreviations and names
    abbrevs = regionmask.defined_regions.srex[da_mask_ts.region.values].abbrevs
    names = regionmask.defined_regions.srex[da_mask_ts.region.values].names
    da_mask_ts.coords['abbrevs'] = ('region', abbrevs)
    da_mask_ts.coords['names'] = ('region', names)
    
    f, axes = plt.subplots(3, 2, figsize=(8,5))
    f.suptitle(da_to_mask.name, fontsize=14)

    low = da_mask_ts.min()
    high = da_mask_ts.max()
    for i in range(len(selected_regions)):
        (nx,ny) = axes.shape
        if i < nx : ax = axes[i,0]
        else      : ax = axes[i-nx,1]
            
        ts_region = da_mask_ts.isel(region=i)
        ts_region.plot(ax=ax)
        ax.set_title(da_mask_ts.isel(region=i).names.values)
        ax.set_ylim(low,high)
        ax.set_ylabel('('+da_to_mask.units+')')
        ax.set_xlim(da_mask_ts.year[0],da_mask_ts.year[-1])
        ax.plot([da_mask_ts.year[0],da_mask_ts.year[-1]], [0,0], linewidth=1, color='gray')
    
    plt.setp(axes, xlabel="")

    f.tight_layout()


# plot global map of difference 
def plot_delta_map(da_delta, plot_regions=False, vlims=False, calcsum=False, cmap='BrBG'):
    
    # calculate annual sum instead of mean (precip)
    if calcsum: 
        da_delta_ysum = da_delta.groupby('time.year').sum()
        da_delta_mean = da_delta_ysum.mean('year')
        da_delta_mean.attrs['units'] = 'mm/year'
    # only one value
    elif len(da_delta.dims) < 3: 
        da_delta_mean = da_delta
    # annual means already taken
    elif len(da_delta) < 50:
        da_delta_mean = da_delta.mean('year')
    else:
        da_delta_mean = da_delta.mean('time')
    
    plt.figure(figsize=(12,5))
    proj=ccrs.PlateCarree()
    ax = plt.subplot(111, projection=proj)
        
    # limiting values for plotting are given    
    if vlims==False: 
        da_delta_mean.plot(ax=ax, cmap=cmap, cbar_kwargs={'label': da_delta.name+' ('+da_delta.units+')', 'fraction': 0.02, 'pad': 0.04})
    else: 
        da_delta_mean.plot(ax=ax, cmap=cmap, vmin=vlims[0], vmax=vlims[1], extend='both',  cbar_kwargs={'label': da_delta.name+' ('+da_delta.units+')', 'fraction': 0.02, 'pad': 0.04}, add_labels=False)
        
    ax.set_title(da_delta.long_name, loc='right')
    ax.coastlines(color='dimgray', linewidth=0.5)
    # exclude Antactica from plot
    ax.set_extent((-180,180,-63,90), crs=proj) 

    if plot_regions: regionmask.defined_regions.srex.plot(ax=ax,add_ocean=False, coastlines=False, add_label=False) #label='abbrev'
    return ax



f, ax = plt.subplots()

# plot global map of difference 
def plot_delta_map_noax(ax, da_delta, plot_regions=False, vlims=False, calcsum=False, cmap='BrBG'):
    """plot difference maps without creating a figure within function"""
    # calculate annual sum instead of mean (precip)
    if calcsum: 
        da_delta_ysum = da_delta.groupby('time.year').sum()
        da_delta_mean = da_delta_ysum.mean('year')
        da_delta_mean.attrs['units'] = 'mm/year'
    # only one value
    elif len(da_delta.dims) < 3: 
        da_delta_mean = da_delta
    # annual means already taken
    elif len(da_delta) < 50:
        da_delta_mean = da_delta.mean('year')
    else:
        da_delta_mean = da_delta.mean('time')
    
    # limiting values for plotting are given    
    if vlims==False: 
        da_delta_mean.plot(ax=ax, cmap=cmap, cbar_kwargs={'label': da_delta.name+' ('+da_delta.units+')'})
    else: 
        da_delta_mean.plot(ax=ax, cmap=cmap, vmin=vlims[0], vmax=vlims[1], extend='both',  cbar_kwargs={'label': da_delta.name+' ('+da_delta.units+')'}, add_labels=False)
        
    ax.set_title(da_delta.long_name, loc='right')
    ax.coastlines(color='dimgray', linewidth=0.5)
    # exclude Antactica from plot
    ax.set_extent((-180,180,-63,90)) 

    if plot_regions: regionmask.defined_regions.srex.plot(ax=ax,add_ocean=False, coastlines=False, add_label=False) #label='abbrev'
    return ax




# plot boxplot for different regions for one variable.  Possible to use with ax keyword as part of subplot. 
def plot_boxplot_regions(da_in, selected_regions, is_subplot=False, ax=ax):
    # monthly or annual means will depend on input. 

    # boxplots
    mask = regionmask.defined_regions.srex.mask(da_in)

    # annual means are already calculate
    da_regions = da_in.groupby(mask).mean('stacked_lat_lon', keep_attrs=True)
    
    # add abbreviations and names
    abbrevs = regionmask.defined_regions.srex[da_regions.region.values].abbrevs
    names = regionmask.defined_regions.srex[da_regions.region.values].names
    da_regions.coords['abbrevs'] = ('region', abbrevs)
    da_regions.coords['names'] = ('region', names)    
    
    
    # select the desired regions
    da_mask_ts = da_regions.sel(region=selected_regions)

    # initialise lists
    data = []
    xlabels = []

    # save values of different regions in boxplots 
    for i in range(len(selected_regions)):
        data_values = da_mask_ts.isel(region=i).values
        data = data + [data_values]
        xlabels = xlabels + [str(da_mask_ts.isel(region=i).abbrevs.values)]


    # create new figure
    if is_subplot==False:
        f, ax = plt.subplots()
        
    ax.yaxis.grid(True, linestyle='-', which='major', color='silver', alpha=0.5)
    ax.boxplot(data)
    ax.set_xticklabels(xlabels)
    ax.set_title(da_mask_ts.name)
    ax.set_ylabel(' ('+da_in.units+')')
    
    return ax

plt.close()


# plot one boxplot (in one column) for the selected regions
def plot_boxplot_regs(var, selected_regions, use_resmask=True):

    f, ax = plt.subplots()
   
    # open data for variable
    da_delta = open_da_delta(var,case_res,case_nores)

    # process data to plot
    if use_resmask: # if to be plotted with reservoir mask 
        resmask = get_resmask()
        da_toplot = da_delta.groupby('time.year').mean(keep_attrs=True).where(resmask)
        da_toplot.name = da_toplot.name +' (reservoir grid cells)'
        
    else: 
        da_toplot = da_delta.groupby('time.year').mean(keep_attrs=True)
            
    
    plot_boxplot_regions(da_toplot, selected_regions, is_subplot=True, ax=ax )

    
# plot multiple boxplots, one per variable (in one column) for the selected regions
def plot_boxplots_vars(variables, selected_regions, use_resmask=True):

    f, axes = plt.subplots(len(variables), figsize=(5,10))
    
    for i in range(len(variables)):
        
        # select specific variable
        var = variables[i]
        
        # open data for variable
        da_delta = open_da_delta(var,case_res,case_nores)

        # process data to plot
        if use_resmask: # if to be plotted with reservoir mask 
            resmask = get_resmask()
            da_toplot = da_delta.groupby('time.year').mean(keep_attrs=True).where(resmask)
        else: 
            da_toplot = da_delta.groupby('time.year').mean(keep_attrs=True)
            
        
        plot_boxplot_regions(da_toplot, selected_regions, is_subplot=True, ax=axes[i])

    f.tight_layout()

    
    
def plot_boxplot_cells(da_in, is_subplot=False, ax=ax):
    # monthly or annual means will depend on input. 

    resmask = get_resmask()
    landmask = get_landmask()
    resmask_1step = get_resmask_1step()
    resmask_2step = get_resmask_2step()

    # annual means are already calculate
    da_res = da_in.where(resmask).mean(dim=('lat','lon'), keep_attrs=True)
    da_res_1step = da_in.where(resmask_1step).mean(dim=('lat','lon'), keep_attrs=True)
    da_res_2step = da_in.where(resmask_2step).mean(dim=('lat','lon'), keep_attrs=True)
    da_lnd = da_in.where(landmask).mean(dim=('lat','lon'), keep_attrs=True)
    da_all = da_in.mean(dim=('lat','lon'), keep_attrs=True)

    # initialise lists
    data = [da_res, da_res_1step,da_res_2step, da_lnd, da_all]
    xlabels = ['res', 'res+1', 'res+2', 'land', 'all']

    # create new figure
    if is_subplot==False:
        f, ax = plt.subplots()

    ax.yaxis.grid(True, linestyle='-', which='major', color='silver', alpha=0.5)
    ax.boxplot(data)
    ax.set_xticklabels(xlabels)
    ax.set_title(da_res.name)
    ax.set_ylabel(' ('+da_in.units+')')

    return ax

    plt.close()
    
def plot_bxplts_all_res(var, ylims=False, isxtrm = False, block=block,stream = stream, da_delta=np.empty(1)):
    """plot row of 2 boxplot of different SREX regions, one with all pixels, one with only reservoir pixels"""
    # check if variable is extreme (because different store location)
    if not isxtrm:
        da_delta = open_da_delta(var,case_res,case_nores,block = block, stream=stream)
    else:          
        # da_delta is given (for CDD)
        if len(da_delta) > 1: 
            da_delta = da_delta
        else:
            da_delta = open_da_delta_xtrm(var,case_res,case_nores,block = block)
       
        
    # make 2 panel boxplot - all land and only reservoir
    f, axes = plt.subplots(1,2, figsize=(10,4))

    # set ylims if hardcoded 
    if not ylims == False: 
        for axs in axes:
            axs.set_ylim(ylims)

    resmask = get_resmask()
    
    # adjust title (if var from lnd, all land, if from atm: all gridcells)
    if block == 'lnd': mask_name = ' (all land)'
    elif block == 'atm': mask_name = ' (all grid cells)'
    
    da_bp_all = da_delta.groupby('time.year').mean(keep_attrs=True).rename(da_delta.name + mask_name)
    plot_boxplot_regions(da_bp_all, selected_regions, is_subplot=True, ax=axes[0])

    da_bp_respix = da_delta.groupby('time.year').mean(keep_attrs=True).where(resmask).rename(da_delta.name  +' (reservoir grid cells)')
    plot_boxplot_regions(da_bp_respix, selected_regions, is_subplot=True, ax=axes[1])

    f.tight_layout()

    # 
def plot_bxplts_all_lnd_res(var, ylims=False, isxtrm = False, block=block, stream=stream, da_delta=np.empty(1), calcsum = False):
    """plot row of 2 boxplot of different SREX regions, one with all pixels, one with all land and one with only reservoir pixels"""
    
    # check if variable is extreme (because different store location)
    if not isxtrm: 
        
        # calculate annual mean (already done for extremes)
        da_delta = open_da_delta(var,case_res,case_nores,block = block, stream=stream)
        
        if calcsum: 
            da_delta = da_delta.groupby('time.year').sum(keep_attrs=True)
            da_delta.attrs['units'] = 'mm/year'
        else: 
            da_delta = da_delta.groupby('time.year').mean(keep_attrs=True)
        
    else:
        # da_delta is given (for CDD)
        if len(da_delta) > 1: 
            da_delta = da_delta
        else:
            da_delta = open_da_delta_xtrm(var,case_res,case_nores,block = block)
       
    
    # make 2 panel boxplot - all land and only reservoir
    f, axes = plt.subplots(1,3, figsize=(15,4))

    # set ylims if hardcoded 
    if not ylims == False: 
        for axs in axes:
            axs.set_ylim(ylims)

    resmask = get_resmask()
    landmask = get_landmask()
        
    da_bp_all = da_delta.rename(da_delta.name +' (all grid cells)')
    plot_boxplot_regions(da_bp_all, selected_regions, is_subplot=True, ax=axes[0])

    da_bp_land = da_delta.where(landmask).rename(da_delta.name +' (all land)')
    plot_boxplot_regions(da_bp_land, selected_regions, is_subplot=True, ax=axes[1])

    da_bp_respix = da_delta.where(resmask).rename(da_delta.name  +' (reservoir grid cells)')
    plot_boxplot_regions(da_bp_respix, selected_regions, is_subplot=True, ax=axes[2])

    f.tight_layout()
 
def plot_seascycle_respct(da_delta, ax=False, title=False, legend=True, vlims = False, panel_label = False, xlabel=True):
    """plot the seasonal cycle averaged over reservoir grid cells for different thresholds"""
    import matplotlib as mpl

    # choosing sequential colors for lines

    cmap = mpl.cm.get_cmap('YlGnBu')
    color_0 = cmap(0.2)
    color_1 = cmap(0.3)
    color_2 = cmap(0.45)
    color_5 = cmap(0.6)
    color_10 = cmap(0.75)
    color_15 = cmap(0.9)
    
    color_gray = 'powderblue'

    if ax == False: 
        f, ax = plt.subplots(figsize=(7,5))

    ax.axhline(color='gray', linewidth = 0.8, label='_nolegend_')

    da_delta.where(np.logical_not(get_resmask(threshold=1))).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_gray)
    da_delta.where(get_resmask(threshold=0)).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_0)
    da_delta.where(get_resmask(threshold=1)).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_1)
    da_delta.where(get_resmask(threshold=2)).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_2)
    da_delta.where(get_resmask(threshold=5)).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_5)
    da_delta.where(get_resmask(threshold=10)).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_10)
    da_delta.where(get_resmask(threshold=15)).groupby('time.month').mean().mean(dim=('lat','lon')).plot(ax=ax, color=color_15)

    legend_text = ['no reservoirs',
                    '> 0% reservoir ',   '> 1% reservoir ', \
                   '> 2% reservoir ',   '> 5% reservoir ', \
                   '> 10% reservoir',   '> 15% reservoir']
    
    ax.set_xlim([1,12]);
    ax.set_ylabel(da_delta.units)
    ax.set_xticks(np.arange(1,13)); 
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D']); 
    
    if not vlims == False: 
        ax.set_ylim(vlims)
    
    if legend == True:
        ax.legend(legend_text, loc='center left',  bbox_to_anchor=(1, 0.5), frameon=False);
        
    if title == False: 
        ax.set_title(da_delta.long_name, loc='right');
    else: 
        ax.set_title(title, loc='right');     
    
    if xlabel == False: 
        ax.set_xlabel(' ')
        
    if panel_label != False:
        ax.text(0, 1.02, panel_label, color='dimgrey', fontsize=12, transform=ax.transAxes, weight = 'bold')
    return ax

def plot_seasmeans(da_delta, cmap='RdBu_r', vlims=False):
    seasons = ['MAM', 'JJA','SON','DJF']
    # create figure
    fig, axes = plt.subplots(nrows=2,ncols=2, figsize=(18,8), subplot_kw={'projection': ccrs.PlateCarree()})
    axes=axes.flatten()

    # loop over seasons
    for i in range(0,4):
        # calculate seasonal mean
        da_delta_mean = calc_seasmean(da_delta,seasons[i])
        if vlims==False:
            da_delta_mean.plot(ax=axes[i], cmap=cmap,  extend='both', cbar_kwargs={'label': da_delta.name+' ('+da_delta.units+')'}, add_labels=False)

        else:    
            da_delta_mean.plot(ax=axes[i], cmap=cmap, vmin=vlims[0], vmax=vlims[1], extend='both',  cbar_kwargs={'label': da_delta.name+' ('+da_delta.units+')'}, add_labels=False)

        axes[i].set_title(seasons[i], loc='left', fontsize=14)
        axes[i].coastlines(color='dimgray', linewidth=0.5)
        axes[i].set_extent((-180,180,-63,90)) 
    fig.suptitle(da_delta.long_name, fontsize = 16, y=0.95); 

    
def get_bottom(data): 
    """ helper function to calculate bottom to plot negative and positive stacked barplots """
    
    # Take negative and positive data apart and cumulate
    def get_cumulated_array(data, **kwargs):
        cum = data.clip(**kwargs)
        cum = np.cumsum(cum, axis=0)
        d = np.zeros(np.shape(data))
        d[1:] = cum[:-1]
        return d  

    cumulated_data = get_cumulated_array(data, min=0)
    cumulated_data_neg = get_cumulated_array(data, max=0)

    # Re-merge negative and positive data.
    row_mask = (data<0)
    cumulated_data[row_mask] = cumulated_data_neg[row_mask]
    data_stack = cumulated_data
    
    return data_stack
    
# ---------------------------------------------------------------------
# 4. Functions to calculate extremes
# ---------------------------------------------------------------------

# Extreme functions

# get da of CDD: annual number of consecutive dry days 
# because not possible to save as netcdf. 
def get_CDD(var_or, case=case, block='atm'):
    
    # define new variable name
    var = 'CDD'
    
    # open da with daily data and convert to mm/day
    da = open_da(var_or,case=case, stream='h1', block=block)
    da = conv_m_s_to_mm_day(da)

    # initialise np array to fill per year
    da_year_mean = da.groupby('time.year').mean()
    cdd_max_peryear = np.empty_like(da_year_mean.values)

    # Loop over grouped data per year 
    for i, da_year in enumerate(list(da.groupby('time.year'))): 
        # create empty np array to save boolean variables indication day is dry
        drydays = np.empty_like(da_year[1].values)
        drydays = da_year[1].values < 1

        # empty cdd matrix per year 
        cdd = np.empty_like(da_year[1].values)

        # loop over all days of the year
        for d in range(len(drydays)):
            if d > 0: 
                ddind = np.where(drydays[d,:,:])
                notddind = np.where(drydays[d,:,:]==False)

                cdd_curd = cdd[d,:,:] 
                cdd_prevd = cdd[d-1,:,:]

                cdd_curd[ddind] = cdd_prevd[ddind] +1
                cdd_curd[notddind] = 0

                cdd[d,:,:] = cdd_curd


        # define maximum of cdd of the year
        cdd_max_peryear[i,:,:] =  cdd.max(axis=0)


    # save into data array and update attributes
    da_xtrm = xr.DataArray(cdd_max_peryear, coords= da_year_mean.coords,
         dims=da_year_mean.dims)

    da_xtrm.name = 'CDD'
    da_xtrm.attrs['units'] = 'days'
    da_xtrm.attrs['long_name'] = 'Annual maximal number of consecutive dry days (PR>1 mm/day)'

    return da_xtrm


# get CDD for every ensemble member and calculate ensemble mean
def get_CDD_ens(case_nores, case_res, n_ens = n_ens): 
    
    for i in range(1,n_ens+1):
        
        print('Calculating CDDs for ensemble member '+ str(i))

        case_res_name   = case_res+'.00'  +str(i)
        case_nores_name = case_nores+'.00'+str(i)
    
        cdd_nores = get_CDD('PRECT',case_nores_name)
        cdd_res   = get_CDD('PRECT',case_res_name)
        
        if i==1: 
            cdd_nores_concat = cdd_nores 
            cdd_res_concat   = cdd_res 

        else: 
            cdd_nores_concat = xr.concat((cdd_nores_concat, cdd_nores), dim='ens_member')
            cdd_res_concat = xr.concat((cdd_res_concat, cdd_res), dim='ens_member')

    cdd_nores_ensmean = cdd_nores_concat.mean(dim='ens_member', keep_attrs='True')
    cdd_res_ensmean   = cdd_res_concat.mean(  dim='ens_member', keep_attrs='True')   

    # calculate delta and update variables
    cdd_delta = cdd_res_ensmean - cdd_nores_ensmean
    cdd_delta.attrs['long_name'] = '$\Delta$ '+ cdd_res.long_name
    cdd_delta.attrs['units'] = cdd_res.units
    cdd_delta.name = '$\Delta$ '+ cdd_res.name
    
    return cdd_delta

# ---------------------------------------------------------------------
# 5. Functions to calculate and plot statistical significance
# ---------------------------------------------------------------------

def calc_pval(da_delta_ens_all, isxtrm = False):
    """ calculate pvalue ing the two-sided, paired, non-parametric Wilcoxon signed rank test """
    
    from scipy.stats import wilcoxon

    # extreme values don't have dimension 'time', but 'year' instead
    if isxtrm : lumped = da_delta_ens_all.stack(time_ensmember=("ens_member", "year"))
    else : lumped = da_delta_ens_all.stack(time_ensmember=("ens_member", "time"))

    # wilcoxon signed rank test for every grid point
    ncells = len(lumped.stack(gridcell=("lat","lon")).transpose())    
    p_values = np.empty(ncells)  

    for ind,gridcell in enumerate(lumped.stack(gridcell=("lat","lon")).transpose().values):
        
        # for all zero values, wilcoxon will not work. Assign manually a p-value of 1 as result will by default be non significant. 
        # this is the case for the ocean grid cells for the land model variables. 
        if np.count_nonzero(gridcell) == 0: 
            p = np.nan
        else: 
            w, p = wilcoxon(gridcell)
        
        p_values[ind] = p

    p_values_2D = p_values.reshape(len(da_delta_ens_all['lat']), len(da_delta_ens_all['lon']))

    da_p_values = xr.DataArray(data=p_values_2D, coords=(da_delta_ens_all.coords['lat'],da_delta_ens_all.coords['lon']), dims=('lat','lon'))

    return da_p_values

def add_statsign( da_delta_ens_all, ax=ax, isxtrm=False, alpha = 0.05): 
    """Add hatching to plot, indicating statistical significance"""

    # calculate pvalues
    da_p_values = calc_pval(da_delta_ens_all, isxtrm)
    
    #levels = [0, 0.05, 1]
    ax.contourf(da_p_values.lon, da_p_values.lat, da_p_values, levels = [0, alpha, 1], hatches=['....', ''], colors='none')
    
    return ax
    
def get_statsign_mask(da_delta_ens_all, isxtrm=False, alpha = 0.05): 
    """get a statistical significance mask at alpha from data array with all ensemble members """
    
    # calculate pvalues
    da_p_values = calc_pval(da_delta_ens_all, isxtrm)
     
    return da_p_values< alpha

def get_statsign_fieldsign_mask(da_delta_ens_all, isxtrm=False, alpha = 0.05): 
    """get a statistical significance mask at alpha from data array with all ensemble members
    and mask for grid cells with field significance (based on False Discovery Rate)"""
    
    # calculate pvalues for two sided wilcoxon test 
    da_p_values = calc_pval(da_delta_ens_all, isxtrm)
    
    # mask p-values for field significance with False Discovery Rate
    sig_FDR = calc_fdr(da_p_values, alpha, print_values=False)
    
    h_values = da_p_values< alpha
    
    # get 1 and 0 mask 
    h_values_fs = h_values.where(sig_FDR==1)
     
    # return boolean
    return h_values_fs > 0

def calc_fdr(p_values, alpha = 0.05, print_values=True):
    """ function to check field significance with False Discovery Rate
        Wilks 2006, J. of Applied Met. and Clim.
        p_val = P-values at every grid point,
        h_val = 0 or 1 at every grid point, depending on significance level
        code translated from R to python from Lorenz et al. (2016)
        https://github.com/ruthlorenz/stat_tests_correlated_climdata/blob/master/FStests/false_discovery_rate_package.R """

    h_values = p_values< alpha
    
    # where pvalues nan, set h_values also to nan
    h_values.where(p_values != np.nan, np.nan)

    # K sum of local p values
    K = (h_values != np.nan).sum().values.item()

    # put all p-values in 1D vector
    prob_1D = p_values.stack(z=('lat','lon'))

    # sort vector increasing
    p_sort = prob_1D.sortby(prob_1D)


    # create empty data arrays
    p = prob_1D * np.nan
    fdr = p_values*0+1
    sig_FDR = p_values*0+1

    # reject those local tests for which max[p(k)<=(siglev^(k/K)]
    for k in range(0,K):
        if (p_sort[k] <= alpha*(k/K)):
            p[k] = p_sort[k].values.item()
        else: 
            p[k] = 0

    p_fdr = p.max() 

    fdr = fdr.where(p_values <= p_fdr)
    sig_FDR = sig_FDR.where(np.logical_and(fdr==1, h_values==1))

    sig_pts = sig_FDR.sum(skipna=True).values

    if print_values: 
        print('False Discovery Rate for Field Significance')
        print("Field significance level: "+ str(alpha))
        print('Number of significant points: '+str(sig_pts))
        print('Total Number of Tests: '+str(K))

    return sig_FDR

def calc_bin_statistics(da_tobin, da_forbinning, nbins, resmask, da_type, var_tobin):
    """ function to calculate bin statistics for the whole ensemble """

    # initialise empty arrays for statistics
    da_bin_median_all = np.array([])
    da_bin_25pct_all = np.array([])
    da_bin_75pct_all = np.array([])

    # loop over ensemble members and save per ensemble member if is not existing yet
    for ensmem in range(0,n_ens): #n_ens

        print('binning ens member: ' + str(ensmem+1))

        # selec ensemble members and take reservoir mask
        da_forbinning_mem = da_forbinning[ensmem,:]
        da_tobin_mem = da_tobin[ensmem,:]

        # do binning
        da_binned_mem = da_tobin_mem.groupby_bins(da_forbinning_mem, nbins)

        # calculate bin statistics per member
        da_bin_median_mem = da_binned_mem.median().values
        da_bin_25pct_mem = da_binned_mem.quantile(0.25).values
        da_bin_75pct_mem = da_binned_mem.quantile(0.75).values

        bin_dict_mem = {'median': da_bin_median_mem, 'Q25' : da_bin_25pct_mem, 'Q75' : da_bin_75pct_mem}
            


        if ensmem == 0:
            da_bin_median_all = da_bin_median_mem
            da_bin_25pct_all = da_bin_25pct_mem
            da_bin_75pct_all = da_bin_75pct_mem
        else: 
            da_bin_median_all = np.vstack((da_bin_median_all,da_bin_median_mem))
            da_bin_25pct_all  = np.vstack((da_bin_25pct_all,da_bin_25pct_mem))
            da_bin_75pct_all  = np.vstack((da_bin_75pct_all, da_bin_75pct_mem))

        
    # calculate bin statistics for whole ensemble
    da_bin_median = da_bin_median_all.mean(axis=0)
    da_bin_25pct = da_bin_25pct_all.mean(axis=0) 
    da_bin_75pct = da_bin_75pct_all.mean(axis=0)
    return (da_bin_median,da_bin_25pct, da_bin_75pct)


def save_bins(var_to_bin, var_for_binning):
    
    nbins = 20
    # load all ens members
    # used to determine bins
    da_forbinning_res   = open_da_ens(var_for_binning, case_res,   stream = 'h1', mode='all').where(resmask).mean(dim=('lat','lon'))

    if var_to_bin == 'WBGT': 
        da_forbinning_res = da_forbinning_res[:,:-1,:,:]    
                
    # used for binning
    da_tobin_res   = open_da_ens(var_to_bin, case_res,   stream = 'h1', mode='all').where(resmask).mean(dim=('lat','lon'))
    
    # calculate bin statistics
    (da_bin_median_res  , da_bin_25pct_res  , da_bin_75pct_res)    =  calc_bin_statistics(da_tobin_res,   da_forbinning_res,   nbins, resmask, 'res', var_to_bin)  
   
    # do the same for no res
    da_forbinning_nores = open_da_ens(var_for_binning, case_nores, stream = 'h1', mode='all')
    da_tobin_nores = open_da_ens(var_to_bin, case_nores, stream = 'h1', mode='all')

    if var_to_bin == 'WBGT': 
        da_forbinning_nores = da_forbinning_nores[:,:-1,:,:]
                
    (da_bin_median_nores, da_bin_25pct_nores, da_bin_75pct_nores)  =  calc_bin_statistics(da_tobin_nores, da_forbinning_nores, nbins, resmask, 'nores', var_to_bin)    
    
    bindiff       = da_bin_median_res - da_bin_median_nores
    bindiff_25pct = da_bin_25pct_res - da_bin_25pct_nores
    bindiff_75pct = da_bin_75pct_res - da_bin_75pct_nores
    
    bindiff_dict = {'median' : bindiff, 'Q25' : bindiff_25pct, 'Q75' : bindiff_75pct}
    np.save(var_to_bin+'_binT.npy', bindiff_dict, allow_pickle = True)