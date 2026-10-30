import numpy as np
import operator
from netCDF4 import Dataset
import time, os, glob, h5py, csv, sys
from math import ceil, sqrt
from scipy.spatial import QhullError
import pandas as pd
from global_land_mask import globe
from pyproj import Proj, transform
import matplotlib.path
from scipy import signal
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

start_time = time.time()

"""
Author: Kyle Duncan
Date: 10/28/2026


CODE SUMMARY:
The following code takes a single ICESat-2 ATL03 file (Neumann et al., 2025) as input and tracks
sea ice elevation using the University of Maryland-Ridge Detection Algorithm (UMD-RDA) describe in
Duncan & Farrell (2022). In the Arctic, ATL03 segments 3, 4, and 5 (Neumann et al., 2025) are considered
for processing. Atmospheric and tidal corrections contained within the ATL03 file are applied to the sea
ice elevations including the dynamic atmosphere correction (DAC), ocean tide, and long period tide. In the
study by Duncan & Farrell (2022) the sea ice elevations were referenced to the Danish Technical University
Mean Sea Surface Model 2021 (DTU21 MSS, Andersen et al., 2023). Here, we indicate where the user may
define/import their own MSS model or other desired reference surface.

The output is a NetCDF file containing longitude, latitude, time, and sea ice elevation tracked using the
UMD-RDA. Further information about the workflow of this algorithm can be found in Duncan & Farrell (2022; 2026a).

The example provided here uses "ATL03_20190422132750_03710304_007_01.h5". The example data output were
obtained from the outside strong beam (gt1l) on RGT 371, segment 4, acquired on April 22, 2019. The
reference surface used was the DTU21 MSS (Andersen et al., 2023). See the figures titled
"ATL03_20190422132750_03710304_007_01_gt1l_UMDRDA_SurfElevation_ALL.png" and
"ATL03_20190422132750_03710304_007_01_gt1l_UMDRDA_SurfElevation_20km.png" within the "example_output"
folder.

Monthly gridded sea ice deformation data spanning 7 years (Oct 2018 - Sep 2025), at 3.125 km, 10 km, and 25 km
resolution, produced using the UMD-RDA, is available at the PANGAEA data repository See "REFERENCES" below for
dataset citations (i.e. Duncan & Farrell, 2026b, 2026c, and 2026d).

USAGE NOTES: Cloud filtering is done at the user's own discretion. The ATL07 Sea Ice Height product
(Kwok et al., 2025) provides cloud flags derived from the ICESat-2 ATL09 L3A Calibrated Backscatter Profiles
and Atmospheric Layer Characteristics product (Palm et al., 2025). For example, a flag called "layer_flag",
located within the ATL07 group "/sea_ice_segments/stats/layer_flag" may be used.

REFERENCES:

Andersen, O. B., Rose, S. K., Abulaitijiang, A., Zhang, S., and Fleury, S. (2023). The DTU21
global mean sea surface and first evaluation, Earth Syst. Sci. Data, 15, 4065–4075,
https://doi.org/10.5194/essd-15-4065-2023

Duncan, K., & Farrell, S. L. (2022). Determining variability in Arctic sea ice pressure ridge
topography with ICESat-2. Geophysical Research Letters, 49, e2022GL100272.
https://doi.org/10.1029/2022GL100272

Duncan, K., & Farrell, S. L. (2026a). Satellite-derived Arctic sea ice surface roughness, ridging and deformation
data product. Nature Scientific Data, under review.

Duncan, K., & Farrell, S. L. (2026b) Gridded 3.125 km Arctic sea ice deformation data derived from the
ICESat-2 University of Maryland-Ridge Detection Algorithm [dataset]. PANGAEA.
https://doi.org/10.1594/PANGAEA.990275

Duncan, K., & Farrell, S. L. (2026c) Gridded 10 km Arctic sea ice deformation data derived from the ICESat-2
University of Maryland-Ridge Detection Algorithm [dataset]. PANGAEA.
https://doi.org/10.1594/PANGAEA.990265

Duncan, K., & Farrell, S. L. (2026d) Gridded 25 km Arctic sea ice deformation data derived from the ICESat-2
University of Maryland-Ridge Detection Algorithm [dataset]. PANGAEA.
http://doi.org/10.1594/PANGAEA.990274

Kwok, R. et al. (2025) ATLAS/ICESat-2 L3A Sea Ice Height. (ATL07, Version 7). [Data Set]. Boulder, Colorado
USA. NASA National Snow and Ice Data Center Distributed Active Archive Center.
https://doi.org/10.5067/ATLAS/ATL07.007

Palm, S. P. et al. (2025) ATLAS/ICESat-2 L3A Calibrated Backscatter Profiles and Atmospheric Layer Characteristics.
(ATL09, Version 7). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active
Archive Center. https://doi.org/10.5067/ATLAS/ATL09.007


"""

##########################################################################################################################

# Please reference the following publications when using this data:

# Duncan, K., & Farrell, S. L. (2026). Satellite-derived Arctic sea ice surface roughness, ridging and deformation
# data product. Nature Scientific Data, under review.

# Duncan, K., & Farrell, S. L. (2022). Determining variability in Arctic sea ice pressure ridge
# topography with ICESat-2. Geophysical Research Letters, 49, e2022GL100272.
# https://doi.org/10.1029/2022GL100272

##########################################################################################################################

def find_median(x):
    return np.nanmedian(x)

def reproject_to_PSG(x, y):
    inProj,outProj = Proj(init='epsg:4326'), Proj(init='epsg:3413')
    x2,y2 = transform(inProj,outProj,x,y)
    return x2,y2

def reproject_to_WGS84(x, y):
    inProj,outProj = Proj(init='epsg:3413'), Proj(init='epsg:4326')
    x2,y2 = transform(inProj,outProj,x,y)
    return x2,y2

def get_closest(arr, vals):
    idxs = np.searchsorted(arr, vals, side='left')
    prev_idx = ((idxs == len(arr))|(np.fabs(vals - arr[np.maximum(idxs-1,0)]) < np.fabs(vals - arr[np.minimum(idxs,len(arr)-1)])))
    idxs[prev_idx] -= 1
    return arr[idxs]

def filter2(arr):
    num_ph = len(arr)
    arr = arr[(arr >= -49.157) & (arr <= 68.144)]
    arr.sort()
    per15 = int(ceil(len(arr)*0.15))
    per85 = int(len(arr)*0.85)
    interp = 0.99 * (per85 - 1 - per15)
    interp = interp - int(interp)
    arr = arr[per85-2]*(1.0 - interp) + arr[per85-1] * interp
    return tuple((arr,num_ph))

def smooth_surface(z,idx,c):
    bins = np.arange(-49.5,68.5,0.5)
    hph_shot_list = []

    for x in range(0,len(idx)-1):
        hph_shot = z[idx[x]:idx[x+1]]
        if (hph_shot.size >= 3):
            hph_shot_list.append(hph_shot)

    mode_ph_low = np.zeros(len(hph_shot_list)-(c+1))
    mode_ph_high = np.zeros(len(hph_shot_list)-(c+1))
    hph_arr = np.zeros(len(hph_shot_list)-(c+1))
    num_ph = np.zeros(len(hph_shot_list)-(c+1))

    for x in range(c,len(hph_shot_list)-(c+1)):
        arr1,arr2,arr3,arr4,arr5 = hph_shot_list[x-c],hph_shot_list[x-(c-1)],hph_shot_list[x],hph_shot_list[x+(c-1)],hph_shot_list[x+c]
        window = np.hstack((arr1,arr2,arr3,arr4,arr5))
        if (len(window[((window >= -100.0) & (window <= 100.0))]) != 0):
            hist = np.histogram(window, bins=bins)[0]
            mode = np.where(hist == hist.max())[0]
            good_bins = bins[mode]
            diff = np.diff(good_bins)
            diff_idx = np.where(diff == 0.5)[0]

            if (len(mode) == 1):
                mode_ph_low[x-c] = good_bins[0]-2.0
                mode_ph_high[x-c] = good_bins[0]+10.0
                if ((c+1 <= x <= len(z)-c) & (abs(mode_ph_low[x-c] - mode_ph_low[x-c-1]) < 2.0)):
                    if (window[((window >= good_bins[0]-2.0) & (window <= good_bins[0]+10.0))].size >= 3):
                        hph_arr[x-c], num_ph[x-c] = filter2(window[((window >= good_bins[0]-2.0) & (window <= good_bins[0]+10.0))])
                
            elif ((len(mode) >= 2) & (len(good_bins[diff_idx]) == 0) & (good_bins[0] >= 0)):
                mode_ph_low[x-c] = good_bins[0]-2.0
                mode_ph_high[x-c] = good_bins[0]+10.0
                if ((c+1 <= x <= len(z)-c) & (abs(mode_ph_low[x-c] - mode_ph_low[x-c-1]) < 2.0)):
                    if (window[((window >= good_bins[0]-2.0) & (window <= good_bins[0]+10.0))].size >= 3):
                        hph_arr[x-c], num_ph[x-c] = filter2(window[((window >= good_bins[0]-2.0) & (window <= good_bins[0]+10.0))])
                    
            elif ((len(mode) >= 2) & (len(good_bins[diff_idx]) == 0) & (good_bins[0] < 0)):
                mode_ph_low[x-c] = good_bins[0]+5.0
                mode_ph_high[x-c] = good_bins[0]+15.0
                if ((c+1 <= x <= len(z)-c) & (abs(mode_ph_low[x-c] - mode_ph_low[x-c-1]) < 2.0)):
                    if (window[((window >= good_bins[0]+5.0) & (window <= good_bins[0]+15.0))].size >= 3):
                        hph_arr[x-c], num_ph[x-c] = filter2(window[((window >= good_bins[0]+5.0) & (window <= good_bins[0]+15.0))])
                
            elif ((len(mode) >= 2) & (len(good_bins[diff_idx]) != 0)):
                mode_ph_low[x-c] = good_bins[diff_idx][0]-2.0
                mode_ph_high[x-c] = good_bins[diff_idx][0]+10.0
                if ((c+1 <= x <= len(z)-c) & (abs(mode_ph_low[x-c] - mode_ph_low[x-c-1]) < 2.0)):
                    if (window[((window >= good_bins[diff_idx][0]-2.0) & (window <= good_bins[diff_idx][0]+10.0))].size >= 3):
                        hph_arr[x-c], num_ph[x-c] = filter2(window[((window >= good_bins[diff_idx][0]-2.0) & (window <= good_bins[diff_idx][0]+10.0))])

        else:
            hph_arr[x-c] = 9999.0
                
    return hph_arr,num_ph

def interpolate_refsurf():
    if (lon_arr_final.size != 0):
        lon1 = lon_arr_final.min()
        lon2 = lon_arr_final.max()
        lat_min = np.asarray(tuple((lat_arr_final[0], lat_arr_final[len(lat_arr_final)-1]))).min()
        lat_max = np.asarray(tuple((lat_arr_final[0], lat_arr_final[len(lat_arr_final)-1]))).max()

        ptA = reproject_to_PSG(lon_arr_final[0],lat_arr_final[0])
        ptB = reproject_to_PSG(lon_arr_final[len(lon_arr_final)-1],lat_arr_final[len(lat_arr_final)-1])

        Dx = ptB[0] - ptA[0]
        Dy = ptB[1] - ptA[1]
        D = sqrt((Dx*Dx) + (Dy*Dy))
        W = 60000
        Dx = 0.5 * W * Dx / D
        Dy = 0.5 * W * Dy / D

        poly = [(ptA[0]-Dy,ptA[1]+Dx),(ptA[0]+Dy,ptA[1]-Dx),(ptB[0]+Dy,ptB[1]-Dx),(ptB[0]-Dy,ptB[1]+Dx)]

        if ((lon1 < 0) & (lon2 > 0)):

            lat_idx = np.where(refsurf_lat >= lat_min - 0.25)[0]

            lon_subset = refsurf_lon[lat_idx]
            lat_subset = refsurf_lat[lat_idx]
            refsurf_subset = refsurf_vals[lat_idx]

            subset_PSG = reproject_to_PSG(lon_subset,lat_subset)

            points = np.vstack((subset_PSG[0],subset_PSG[1])).T
            p = matplotlib.path.Path(poly)
            grid = p.contains_points(points)

            lon_masked = subset_PSG[0][np.where(grid == True)[0]]
            lat_masked = subset_PSG[1][np.where(grid == True)[0]]
            refsurf_masked = refsurf_subset[np.where(grid == True)[0]]

            subset_WGS84 = reproject_to_WGS84(lon_masked, lat_masked)

            refsurf_coords = np.column_stack((subset_WGS84[0],subset_WGS84[1]))
            is2_coords = np.column_stack((lon_arr_final,lat_arr_final))
            
            if ((refsurf_coords.size != 0) & (refsurf_masked.size != 0)):
                try:
                    z = griddata(refsurf_coords, refsurf_masked, is2_coords, method='linear', fill_value=-9999)
                except QhullError:
                    print ('Not enough points for reference surface interpolation...')
                    z = []
                
            else:
                z = []

        if ((lon1 > 0) & (lon2 > 0)):

            lon_subset = np.where((refsurf_lon >= lon1 - 0.25) & (refsurf_lon <= lon2 + 0.25))[0]
            lat_subset = np.where((refsurf_lat >= lat_min - 0.25) & (refsurf_lat <= refsurf_lat.max() + 0.25))[0]

            same = np.intersect1d(lon_subset,lat_subset)

            lon_subset = refsurf_lon[same]
            lat_subset = refsurf_lat[same]
            refsurf_subset = refsurf_vals[same]

            subset_PSG = reproject_to_PSG(lon_subset,lat_subset)

            points = np.vstack((subset_PSG[0],subset_PSG[1])).T
            p = matplotlib.path.Path(poly)
            grid = p.contains_points(points)

            lon_masked = subset_PSG[0][np.where(grid == True)[0]]
            lat_masked = subset_PSG[1][np.where(grid == True)[0]]
            refsurf_masked = refsurf_subset[np.where(grid == True)[0]]

            subset_WGS84 = reproject_to_WGS84(lon_masked, lat_masked)

            refsurf_coords = np.column_stack((subset_WGS84[0],subset_WGS84[1]))
            is2_coords = np.column_stack((lon_arr_final,lat_arr_final))
            
            if ((refsurf_coords.size != 0) & (refsurf_masked.size != 0)):
                try:
                    z = griddata(refsurf_coords, refsurf_masked, is2_coords, method='linear', fill_value=-9999)
                except QhullError:
                    print ('Not enough points for reference surface interpolation...')
                    z = []
                
            else:
                z = []

        if ((lon1 < 0) & (lon2 < 0)):

            lon_subset = np.where((refsurf_lon >= lon1 - 0.25) & (refsurf_lon <= lon2 + 0.25))[0]
            lat_subset = np.where((refsurf_lat >= lat_min - 0.25) & (refsurf_lat <= refsurf_lat.max() + 0.25))[0]

            same = np.intersect1d(lon_subset,lat_subset)

            lon_subset = refsurf_lon[same]
            lat_subset = refsurf_lat[same]
            refsurf_subset = refsurf_vals[same]

            subset_PSG = reproject_to_PSG(lon_subset,lat_subset)

            points = np.vstack((subset_PSG[0],subset_PSG[1])).T
            p = matplotlib.path.Path(poly)
            grid = p.contains_points(points)

            lon_masked = subset_PSG[0][np.where(grid == True)[0]]
            lat_masked = subset_PSG[1][np.where(grid == True)[0]]
            refsurf_masked = refsurf_subset[np.where(grid == True)[0]]

            subset_WGS84 = reproject_to_WGS84(lon_masked, lat_masked)

            refsurf_coords = np.column_stack((subset_WGS84[0],subset_WGS84[1]))
            is2_coords = np.column_stack((lon_arr_final,lat_arr_final))

            if ((refsurf_coords.size != 0) & (refsurf_masked.size != 0)):
                try:
                    z = griddata(refsurf_coords, refsurf_masked, is2_coords, method='linear', fill_value=-9999)
                except QhullError:
                    print ('Not enough points for reference surface interpolation...')
                    z = []
                
            else:
                z = []

    return z


pathname = '/Path/to/ATL03 file/' ## Path to ICESat-2 ATL03 .h5 files
pathname2 = '/Path/to/reference surface file/' ## Path to user defined reference surface (i.e. MSS, geoid, etc.)

refsurf_nc = Dataset(pathname2 + 'file name', 'r') ## Read reference surface data

refsurf_lon = ## Longitude array of reference surface data
refsurf_lat = ## Latitude array of reference surface data
refsurf_vals = ## Array of reference surface data values

##### Gather ICESat-2 ATL03 file path into numpy array #####

ATL03_files = np.array(glob.glob(pathname + '/*.h5')) ## For example, "ATL03_20190422132750_03710304_007_01.h5". Only ATL03 segments 3, 4, and 5 (>= 60 deg N) are processed with the UMD-RDA in the Arctic

##### Loop through ATL03 files and process for each ICESat-2 strong beam #####

for w,i in enumerate(ATL03_files):

    f = h5py.File(i, 'r') ## Read in ATL03 .h5 file

    ##### Select specific ICESat-2 beams using "/orbit_info/sc_orient" variable. This example shows how to pull in data from the strong beams. #####

    sc_orient = f["orbit_info"]["sc_orient"][:]

    if (sc_orient[0] == 0):
        beam = "gt1l"
        beam2 = "gt2l"
        beam3 = "gt3l"
    elif (sc_orient[0] == 1):
        beam = "gt1r"
        beam2 = "gt2r"
        beam3 = "gt3r"
    else:
        continue

    beams = np.array(list((beam,beam2,beam3)))

    for beam in beams:
        print (beam)
        if (beam in f):
            ## Extract photon height, longitude, latitude, time, surface type, and POD-PPD flag from the ATL03 file
            gt2l_hph = f[beam]["heights"]["h_ph"][:]
            gt2l_hph_idx = np.where((gt2l_hph > -100.0) & (gt2l_hph < 100.0))[0]
            gt2l_hph = gt2l_hph[gt2l_hph_idx]
            gt2l_lon = f[beam]["heights"]["lon_ph"][:][gt2l_hph_idx]
            gt2l_lat = f[beam]["heights"]["lat_ph"][:][gt2l_hph_idx]
            gt2l_dtime = f[beam]["heights"]["delta_time"][:][gt2l_hph_idx]
            gt2l_surftype = f[beam]["geolocation"]["surf_type"][:]
            gt2l_podppd_flag = f[beam]["geolocation"]["podppd_flag"][:]
            
            surftype_seaice_idx = np.where((np.concatenate(gt2l_surftype)[2::5] == 1) & (gt2l_podppd_flag == 0))[0] ## Use ATL03 surface type to select only sea ice data

            if (surftype_seaice_idx.size != 0):
                
                ## Store atmosphere and tidal correction data in array at sea ice locations (surftype_seaice_idx) only
        
                gt2l_dac = f[beam]["geophys_corr"]["dac"][:][surftype_seaice_idx]
                gt2l_oceantide = f[beam]["geophys_corr"]["tide_ocean"][:][surftype_seaice_idx]
                gt2l_long_tide = f[beam]["geophys_corr"]["tide_equilibrium"][:][surftype_seaice_idx]
                gt2l_geophys_dtime = f[beam]["geophys_corr"]["delta_time"][:][surftype_seaice_idx]

                gt2l_dtime_idx = np.where((gt2l_dtime >= gt2l_geophys_dtime[0]) & (gt2l_dtime <= gt2l_geophys_dtime.max()))[0]

                shot_diff = np.diff(gt2l_dtime[gt2l_dtime_idx])
                shot_idx = np.insert(np.where(shot_diff > 0)[0] + 1,0,0)
                good_shots_size = np.where(np.diff(shot_idx) >= 3)[0].size

                hph_split = np.split(gt2l_hph[gt2l_dtime_idx],shot_idx)

                hph_split_len = np.array([len(x) for x in hph_split])

                arr_len = np.vectorize(len)

                if (len(hph_split) != 0):
                    hph_idx = np.where(hph_split_len >= 3)[0]

                    lon_split = np.array(np.split(gt2l_lon[gt2l_dtime_idx],shot_idx),dtype='object')[hph_idx]
                    lat_split = np.array(np.split(gt2l_lat[gt2l_dtime_idx],shot_idx),dtype='object')[hph_idx]
                    dtime_split = np.array(np.split(gt2l_dtime[gt2l_dtime_idx],shot_idx),dtype='object')[hph_idx]

                    ##### Start surface tracking process #####

                    c=2

                    if (good_shots_size >= 5):
                        hph_arr_final, num_ph_final = smooth_surface(gt2l_hph[gt2l_dtime_idx],shot_idx,2)
                    else:
                        continue
                    
                    dtime_arr_final = np.vectorize(find_median)(dtime_split)[c:-(c-1)]
                    lon_arr_final = np.vectorize(find_median)(lon_split)[c:-(c-1)]
                    lat_arr_final = np.vectorize(find_median)(lat_split)[c:-(c-1)]

                    hph_diff = np.where((np.diff(hph_arr_final) <= 0.75) & (np.diff(hph_arr_final) >= -0.75))[0] + 1

                    hph_medfilt = hph_arr_final[hph_diff]

                    ##### Interpolate user defined reference surface data #####

                    z = interpolate_refsurf()

                    if (len(z) == 0):
                        continue

                    ##### Outlier Removal #####
                    
                    hph_medfilt_diff = np.diff(hph_medfilt)
                    outlier1 = np.where((num_ph_final[hph_diff][:-1] > 0) & (num_ph_final[hph_diff][:-1] <= 15.0) & (hph_medfilt_diff >= 0.65))[0]
                    outlier2 = np.where((num_ph_final[hph_diff][:-1] > 0) & (num_ph_final[hph_diff][:-1] <= 15.0) & (hph_medfilt_diff <= -0.65))[0]
                    combined = np.sort(np.hstack((outlier1,outlier2)))
                    combined_diff = np.where(np.diff(combined) > 20)[0] + 1

                    split = np.split(combined,combined_diff)

                    ind_list = []

                    for x in range(0,len(split)-1):   
                        b = np.arange(split[x][0],split[x][len(split[x])-1]+1)
                        ind_list.append(b)

                    if (len(ind_list) != 0):
                        ind_arr = list(np.concatenate(ind_list))

                        lon_final = np.delete(lon_arr_final[hph_diff],ind_arr)
                        lat_final = np.delete(lat_arr_final[hph_diff],ind_arr)
                        time_sec_final = np.delete(dtime_arr_final[hph_diff],ind_arr)
                        height_final = np.delete(hph_medfilt,ind_arr)
                        num_ph_final = np.delete(num_ph_final[hph_diff],ind_arr)
                        mss_z = np.delete(z[hph_diff],ind_arr)
                        
                        height_good = np.where((height_final >= -80.0) & (height_final <= -0.0001))[0]
                        height_good2 = np.where((height_final >= 0.0001) & (height_final <= 100.0))[0]
                        joined = np.sort(np.concatenate((height_good,height_good2)))
                        
                    else:
                        lon_final = lon_arr_final[hph_diff]
                        lat_final = lat_arr_final[hph_diff]
                        time_sec_final = dtime_arr_final[hph_diff]
                        height_final = hph_medfilt
                        num_ph_final = num_ph_final[hph_diff]
                        mss_z = z[hph_diff]

                        height_good = np.where((height_final >= -80.0) & (height_final <= -0.0001))[0]
                        height_good2 = np.where((height_final >= 0.0001) & (height_final <= 100.0))[0]
                        joined = np.sort(np.concatenate((height_good,height_good2)))


                    if (len(height_final[joined]) >= 2):

                        height_grad = np.gradient(height_final[joined])

                        outlier1 = np.where(height_grad >= 0.6)[0]
                        outlier2 = np.where(height_grad <= -0.6)[0]
                        combined = np.sort(np.hstack((outlier1,outlier2)))
                        combined_diff = np.where(np.diff(combined) > 20)[0] + 1

                        split = np.split(combined,combined_diff)

                        ind_list = [np.arange(split[x][0],split[x][len(split[x])-1]+1) for x in range(0,len(split)-1)]

                        if (len(ind_list) != 0):
                            ind_arr = list(np.concatenate(ind_list))

                            lon_final = np.delete(lon_final[joined],ind_arr)
                            lat_final = np.delete(lat_final[joined],ind_arr)
                            time_sec_final = np.delete(time_sec_final[joined],ind_arr)
                            height_final = np.delete(height_final[joined],ind_arr)
                            num_ph_final = np.delete(num_ph_final[joined],ind_arr)
                            mss_z = np.delete(mss_z[joined],ind_arr)

                            height_good = np.where((height_final >= -80.0) & (height_final <= -0.0001))[0]
                            height_good2 = np.where((height_final >= 0.0001) & (height_final <= 100.0))[0]
                            joined2 = np.sort(np.concatenate((height_good,height_good2)))

                        else:
                            lon_final = lon_final[joined]
                            lat_final = lat_final[joined]
                            time_sec_final = time_sec_final[joined]
                            height_final = height_final[joined]
                            num_ph_final = num_ph_final[joined]
                            mss_z = mss_z[joined]

                            height_good = np.where((height_final >= -80.0) & (height_final <= -0.0001))[0]
                            height_good2 = np.where((height_final >= 0.0001) & (height_final <= 100.0))[0]
                            joined2 = np.sort(np.concatenate((height_good,height_good2)))


                        ##### Apply elevation corrections using DAC, ocean tide, long period tide, and MSS #####

                        dac_interp = np.interp(time_sec_final[joined2], gt2l_geophys_dtime, gt2l_dac)
                        oceantide_interp = np.interp(time_sec_final[joined2], gt2l_geophys_dtime, gt2l_oceantide)
                        longtide_interp = np.interp(time_sec_final[joined2], gt2l_geophys_dtime, gt2l_long_tide)
                        mss_final = mss_z[joined2]

                        corrections = mss_final + dac_interp + oceantide_interp + longtide_interp

                        height_corrected_final = height_final[joined2] - corrections
                        
                        lon_final = lon_final
                        lat_final = lat_final
                        time_sec_final = time_sec_final
                        height_final = height_corrected_final
                        num_ph_final = num_ph_final

                        ##### Prepare data for writing to file #####

                        if (lon_final.size != 0):
                            data_filter = np.where(height_final < 7.0)[0]

                            if (data_filter.size != 0):
                                dtime_range = np.arange(time_sec_final[data_filter].min(),time_sec_final[data_filter].max()+0.14285714286,0.14285714286) ## Create array for 1 km time steps

                                time_idx_arr = np.round(np.unique(get_closest(time_sec_final[data_filter],dtime_range[1:])),4)
                                xy, x_ind, y_ind = np.intersect1d(time_idx_arr,np.round(time_sec_final[data_filter],4),return_indices=True)
                                y_ind = np.delete(y_ind,np.where(np.diff(y_ind) == 1)[0])
                                time_split = np.array(np.array_split(time_sec_final[data_filter],y_ind),dtype='object')
                                height_split = np.array(np.array_split(height_final[data_filter],y_ind),dtype='object')
                                time_split_concatenated = np.concatenate(time_split)

                                if ((len(time_split) > 2) & (len(time_split) != len(time_split_concatenated))):
                                    low_dens = np.where((arr_len(time_split) / 1428.5714) < 0.8)[0]

                                    low_dens_filt_list = [signal.medfilt(height_split[x],5) for x in low_dens]

                                    if ((len(low_dens_filt_list) == 1) & (low_dens_filt_list[0][0] == 0.0)):
                                        low_dens_filt_list = np.delete(low_dens_filt_list,0)
                                        height_insert = np.insert(np.array(height_split),low_dens, low_dens_filt_list)
                                    else:
                                        height_insert = np.insert(np.array(height_split),low_dens, low_dens_filt_list)

                                    low_dens2 = np.where((arr_len(height_insert) / 1428.5714) < 0.8)[0]
                                    
                                    height_new = np.delete(np.array(height_insert), low_dens2[1::2])
                                                        
                                    height_final = np.concatenate(height_new)
                                    lon_final = lon_final[data_filter]
                                    lat_final = lat_final[data_filter]
                                    time_sec_final = time_sec_final[data_filter]
                                    num_ph_final = num_ph_final[data_filter]

                                    ##### Write data to NetCDF #####
                                    
                                    nc_file = Dataset('/path/to/output/netcdf/file', mode='w', format='NETCDF4')

                                    ## Write netcdf dimensions and variables for longitude, latitude, time, and surface elevation ##

                                    lon_dim = nc_file.createDimension('lon',)
                                    lat_dim = nc_file.createDimension('lat',)
                                    dtime_dim = nc_file.createDimension('dtime',)
                                    surfelev_dim = nc_file.createDimension('surf_elev',)
                                    
                                    nc_file_lon = nc_file.createVariable('longitude', np.float64, ('lon',), compression='zlib',least_significant_digit=8)
                                    nc_file_lon.long_name = 'Longitude'
                                    nc_file_lon.units = 'degrees east'

                                    nc_file_lat = nc_file.createVariable('latitude', np.float64, ('lat',), compression='zlib',least_significant_digit=8)
                                    nc_file_lat.long_name = 'Latitude'
                                    nc_file_lat.units = 'degrees north'

                                    nc_file_dtime = nc_file.createVariable('dtime', np.float64, ('dtime',), compression='zlib',least_significant_digit=4)
                                    nc_file_dtime.long_name = 'Delta Time'
                                    nc_file_dtime.units = 'seconds since 2018-01-01'

                                    nc_surfelev = nc_file.createVariable('surface_elevation', np.float64, ('surf_elev',), compression='zlib',least_significant_digit=8)
                                    nc_surfelev.long_name = 'UMD-RDA sea ice surface elevation'
                                    nc_surfelev.units = 'meters, relative to DTU21 Mean Sea Surface.'
                                    nc_surfelev.description = 'Sea ice elevation derived using the UMD-RDA, relative to the DTU21 mean sea surface, and including the following additional geophysical corrections: Dynamic Atmosphere Correction (DAC), ocean tide, and long period tide as well as corrections applied to the ATL03 photon heights'

                                    ## Create land mask ##
                    
                                    ocean_idx = np.where(globe.is_ocean(lat_final,lon_final) == True)[0]

                                    ## Write masked longitude, latitude, time, and surface elevation to netCDF file ##

                                    nc_file_lon[:] = lon_final[ocean_idx]
                                    nc_file_lat[:] = lat_final[ocean_idx]
                                    nc_file_dtime[:] = time_sec_final[ocean_idx]
                                    nc_surfelev[:] = height_final[ocean_idx]

                                    ## See figures within the "example_output" folder for profiles of surface elevation, tracked using this code, for the strong beam "gt1l",
                                    ## using the ATL03 file "ATL03_20190422132750_03710304_007_01.h5"

                                    nc_file.close()
                                    

print(("%s seconds" % (time.time() - start_time)))
