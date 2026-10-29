# UMDRDA_ICESat2
Provided here is code used to process ICESat-2 ATL03 files using the University of Maryland-Ridge Detection Algorithm (UMD-RDA) and turn the ICESat-2 UMD-RDA along-track data into monthly grids, calculate monthly climatologies, monthly anomalies, and the removal of outliers. Gridded data files are calculated at three spatial resolutions of 25 km, 10 km, and 3.125 km.

### Step 1: UMD-RDA
"Process_ATL03_Surface_Finding.py" contains code for the along-track processing of ICESat-2 ATL03 files with the University of Maryland-Ridge Detection Algorithm (UMD-RDA). 

Please read the "Code Summary" and additional information that is provided at the top of the file before using. 

Please see the "example_output" folder within this repository for figures showing two different profiles of surface elevation, one of the entire ~2200 km segment and another showing a 20 km zoom-in, obtained using the output from the "Process_ATL03_Surface_Finding.py" code when using the ATL03 file "ATL03_20190422132750_03710304_007_01.h5", acquired on April 22, 2019.

### Step 2: Grid Data
"Arctic_Grid_netCDF_Monthly_PyGMT_FINAL.py" contains code for gridding the ICESat-2 UMD-RDA along-track data using PyGMT

### Step 3: Calculate Climatology
"UMDRDA_Grid_Monthly_Climatology.py" calculates monthly climatologies for January-December for the ICESat-2 period of October 2018 to September 2025

### Step 4: Calculate Monthly Anomaly
"UMDRDA_Grid_Monthly_Anomaly.py" calculates monthly anomalies based on the monthly climatologies from Step 2

### Step 5: Identify Outliers
"UMDRDA_Grid_Fix_Outliers.py" uses the monthly anomalies and sea ice concentration data to identify and remove outliers, producing a final monthly gridded netcdf file

The gridded deformation data product is available at PANGAEA (https://www.pangaea.de). The DOIs for the different resolution products are:

25 km: https://doi.org/10.1594/PANGAEA.990274 \
10 km: https://doi.org/10.1594/PANGAEA.990265 \
3.125 km: https://doi.org/10.1594/PANGAEA.990275

The product "User Guide" can be found at: https://download.pangaea.de/reference/137687/attachments/UMD_Arctic_SeaIceDeformation_UserGuide.pdf

Please email kduncan@umd.edu for any questions related to this repository

### References
Please reference the following publication, in addition to the dataset, when using this data:

Duncan, K., & Farrell, S. L. (2022). Determining variability in Arctic sea ice pressure ridge topography with ICESat-2.Geophysical Research Letters, 49, e2022GL100272. https://doi.org/10.1029/2022GL100272

### Acknowledgements & Funding:
This work was supported by NASA Cryosphere Program Grant 80NSSC24K1450 and by NOAA Grant NA24NESX432C0001 (Cooperative Institute for Satellite Earth System Studies – CISESS) at the University of Maryland/ESSIC.

