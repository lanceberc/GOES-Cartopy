#!/usr/local/bin/python3
"""
Reproject a Geosynchronous image to Mercator using Cartopy

Copyright 2019 Lance Berc
Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:
The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING 

This has only been tested on GOES-17 PACUS images. I suspect
it won't work for Full Disk or other sectors where off-earth pixels
need to be masked off or redirected earthwards.

WARNING WARNING WARNING WARNING WARNING WARNING WARNING WARNING 

Getting the needed environment working on MacOS was tricky.

(a) Uninstall Proj, GEOS, and GDAL if present
sudo port uninstall Proj4 Proj6 Geos GDAL

(b) Install the version from KyngChaos
https://www.kyngchaos.com/software/frameworks/

(c) Use pip3 to install shapely and cartopy, but avoid pre-built binaries
pip3 uninstall shapely
pip3 uninstall cartopy
pip3 install shapely cartopy --no-binary shapely --no-binary cartopy

---

Satellite imager and projection parameters from NASA
https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf

Computing the x & y projection coordinates was hinted at by NetCDF
https://cf-trac.llnl.gov/trac/ticket/72

Source code for Geos projection:
https://svn.osgeo.org/metacrs/proj/trunk/proj/src/PJ_geos.c

Cartopy inspiration from Brian Blaylock's examples
https://github.com/blaylockbk/pyBKB_v3/blob/master/BB_GOES/mapping_GOES16_TrueColor.ipynb

"""

import numpy as np
import pyproj
import matplotlib
import matplotlib.pyplot
import matplotlib.font_manager
import cartopy
import cartopy.crs

import argparse
import logging
import os
import os.path
import re

# Constants taken from NASA Product Users Guide, Volume 3, Section 5.1.2
# https://www.goes-r.gov/users/docs/PUG-L1b-vol3.pdf

goes16 = { # GOES-16, aka GOES-R, GOES-EAST
    "perspective_height": 35786023.0, # from the ellipsoid
    "height": 42164160.0,             # from center of the earth
    "longitude": -75.0,
    "sweep_axis": 'x',
    "semi_major": 6378137.0,          # GRS-80 ellipsoid
    "semi_minor": 6356752.31414,      # GRS-80 ellipsoid
    "flattening": 298.257222096,
    "eccentricity": 0.0818191910435,
    # The other resolution (.5k, 2k, 4k, etc) can be added here
    "1k": {
        "resolution": 0.000028,       # radians per pixel
        "FD": {
            "x_offset": -0.151858,    # radians from nadir
            "y_offset":  0.151858,
            "shape": (10848, 10848)   # pixels in image
        },
        "CONUS": {
            "x_offset": -0.101346,
            "y_offset":  0.128226,
            "shape": (5000, 3000)
        }
    }
}

goes17 = { # GOES-17, aka GOES-S, GOES-WEST
    "perspective_height": 35786023.0,
    "height": 42164160.0,
    "longitude": -137.0,
    "sweep_axis": 'x',
    "semi_major": 6378137.0,
    "semi_minor": 6356752.31414,
    "flattening": 298.257222096,
    "eccentricity": 0.0818191910435,
    "1k": {
        "resolution": 0.000028,
        "FD": {
            "x_offset": -0.151858,
            "y_offset":  0.151858,
            "shape": (10848, 10848)
        },
        "CONUS": {
            "x_offset": -0.069986,
            "y_offset":  0.128226,
            "shape": (5000, 3000)
        }
    }
}

# Several interesting (to me) areas
extents = {
    "california": [-134.0, -113.5, 32.0, 41.0],
    "socal"     : [-125.0, -115.0, 32.0, 36.5],
    "snowcal"   : [-134.0, -115.0, 33.5, 44.0],
    "storm"     : [-150.5, -111.1, 33.0, 50.1]
}

regions = {}
regions['storm'] = {
    "sdir": "M:/NASA/NESDIS_CONUS-West",
    "ddir": "M:/NASA/2019-11_NorCal",
    "sat": goes17,
    "goes": "17",
    "sector": "CONUS",
    "res": "1k",
    "starttime": "201911260000",
    "extent": extents['storm']
}

# Cartopy projections
geostationary = None
mercator = None
platecarree = None

# Create a dummy array w/ data in the proper shape - pcolormesh will use colorTuple for values
dummy_array = None

test = False
dpi = 100
lons = []
lats = []

# Variables that need to be set up only once
def init(region):
    global geostationary, mercator, platecarree, dummy_array, lons, lats
    r = regions[region]
    s = r['sat']

    # We need Plate-Carree to define extent because Cartopy isn't complete and can't do inverse mapping w/ Mercator
    geostationary = cartopy.crs.Geostationary(central_longitude=s["longitude"],
                                              satellite_height=s["perspective_height"],
                                              sweep_axis=s["sweep_axis"])
    mercator = cartopy.crs.Mercator()
    platecarree = cartopy.crs.PlateCarree()

    res = r['res']
    sector = r['sector']

    dummy_array = np.empty([s[res][sector]["shape"][1], s[res][sector]["shape"][0]])
    
    matplotlib.rcParams['savefig.pad_inches'] = 0
    matplotlib.rcParams['savefig.bbox'] = "tight"
    matplotlib.rcParams['savefig.dpi'] = dpi
    matplotlib.rcParams['savefig.jpeg_quality'] = 75
    matplotlib.rcParams['font.family'] = 'monospace'
    matplotlib.rcParams['font.size'] = 12
    
    # Projection coords are scan angle * satellite height over ellipsoid (not center of earth)
    logging.debug("Calculating projection data")
    w, h = s[res][sector]["shape"]
    resolution = s[res]['resolution']
    x_off = s[res][sector]["x_offset"]
    y_off = s[res][sector]["y_offset"]
    p_height = s["perspective_height"]

    x = np.ndarray(shape=(w))
    for i in range(w):
        x[i] = (x_off + (i*resolution)) * p_height
    y = np.ndarray(shape=(h))
    for i in range(h):
        y[i] = (y_off - (i*resolution)) * p_height
    X, Y = np.meshgrid(x, y)

    logging.debug("Converting to projection coordinates to lon/lat w/Proj")
    # Convert from projection coords to lon/lat
    p = pyproj.Proj(proj='geos', h=s["perspective_height"], lon_0=s["longitude"], sweep=s["sweep_axis"])
    lons, lats = p(X, Y, inverse=True)     # lons, lats are globals
    del x
    del y
    del X
    del Y

def reproject(region, infile, outfile, tsstring=None):
    r = regions[region]
    logging.debug("Reproject %s -> %s" % (infile, outfile))

    # The goal is a 16:9 jpg file with no matplotlib decoration that is sized w/dpi
    # This gets close
    w, h = (1920/dpi, 1280/dpi)
    fig = matplotlib.pyplot.figure(frameon=False) # Setting size here doesn't work right
    fig.set_size_inches(w, h)
    ax = matplotlib.pyplot.axes(projection=mercator, frameon=False)
    ax.axis('off')
    ax.set(xlim=[0, w], ylim=[h, 0], aspect=1) # Square pixels (and they are at nadir)
    
    # Create a color tuple for pcolormesh
    # Don't use the last column of the RGB array or else the image is not reprojected
    # Don't know why
    mpimg = matplotlib.pyplot.imread(infile)
    rgb = mpimg[:,:-1,:] * (1.0/255.0) # Normalize from RGB(0-255) to (0-1)
    colorTuple = rgb.reshape((rgb.shape[0] * rgb.shape[1]), 3) # Flatten the array for pcolormesh
    # Stack Overflow suggests adding an alpha channel speed up pcolormesh (32-bit aligned?)
    # Didn't go faster for me but used more memory
    #colorTuple = np.insert(colorTuple, 3, 1.0, axis=1)

    # Cartopy is incomplete - not all transformations are available
    # For some reason we have to use PlateCarree here instead of Mercator
    ax.set_extent(r['extent'], crs=cartopy.crs.PlateCarree())
    newimg = ax.pcolormesh(lons, lats, dummy_array, color=colorTuple, linewidth=0, transform=platecarree)
    newimg.set_array(None) # Without this line the RGB colorTuple is ignored and only dummy is plotted.

    if tsstring != None:
        matplotlib.pyplot.text(0.02, 0.02, tsstring, transform=ax.transAxes,
                               size=16, color="white", ha="left", va="bottom",
                               bbox=dict(boxstyle="round", ec=None, fc=(0.2, 0.2, 0.2), alpha=0.75))

    logging.info("Creating %s" % (outfile))
    matplotlib.pyplot.savefig(outfile)
    matplotlib.pyplot.close(fig) # release memory!
    del mpimg
    del rgb
    del colorTuple
    del newimg
    pimg = None
    rgb = None
    colorTuple = None
    newimg = None

def find_source_files(region):
    logging.debug("Searching for source files")
    r = regions[region]
    s = r['sat']
    path = r["sdir"]
    d = os.listdir(path)
    pat = re.compile("GOES-%s_%s_(\d{12}).jpg$" % (r['goes'], r['sector']))
    files = []
    for date in d:
        datedir = "%s/%s" % (path, date)
        if not os.path.isdir(datedir):
            continue
        l = os.listdir(datedir)
        for e in l:
            m =  pat.match(e)
            if m:
                f = m.group(1)
                if (r["starttime"] <= f):
                    files.append("%s/%s/%s" % (path, date, e))
    files.sort()
    logging.info("Found %d images" % (len(files)))
    return(files)

def reproject_region(region):
    init(region)
    r = regions[region]

    sources = find_source_files(region)

    if test:
        sfn = sources[-1]
        fn = sfn.split('/')[-1]
        goes = fn[0:7]
        year = fn[14:18]
        month = fn[18:20]
        day = fn[20:22]
        hour = fn[22:24]
        minute = fn[24:26]
        s = "%s %s-%s-%s %s:%sZ" % (goes, year, month, day, hour, minute)
        reproject(region, sfn, "tst.jpg", s)
        return
    
    ddir = r["ddir"]
    if not os.path.isdir(ddir):
        os.makedirs(ddir)

    for i, sfn in enumerate(sources):
        logging.info("Source #%d: %s" % (i, sfn))
        # Assumes source is .jpg or .png (or .XYZ)
        date = sfn[-16:-8]
        ddatedir = r["ddir"] + "/" + date
        if not os.path.isdir(ddatedir):
            os.makedirs(ddatedir)
        dfn = ddatedir + "/" + sfn[-16:-4] + ".jpg"
        
        if os.path.isfile(dfn):
            logging.info("Skipping #%d" % (i))
            continue

        fn = sfn.split('/')[-1]
        goes = fn[0:7]
        year = fn[14:18]
        month = fn[18:20]
        day = fn[20:22]
        hour = fn[22:24]
        minute = fn[24:26]
        s = "%s %s-%s-%s %s:%sZ" % (goes, year, month, day, hour, minute)
        reproject(region, sfn, dfn, s)

if __name__ == '__main__':
    #reproject("GOES-17_CONUS_201911181906.jpg", "tst.jpg")
    loglevel = logging.DEBUG
    parser = argparse.ArgumentParser()

    parser.add_argument("-storm", default=False, action='store_true', help="GOES-17 West Coast Storm")
    parser.add_argument("-log", choices=["debug", "info", "warning", "error", "critical"], default="info", help="Log level")
    parser.add_argument("-test", default=False, action='store_true', help="Run only the last image and save to tst.jpg")
    args = parser.parse_args()

    if args.log == "debug":
        loglevel = logging.DEBUG
    if args.log == "info":
        loglevel = logging.INFO
    if args.log == "warning":
        loglevel = logging.WARNING
    if args.log == "error":
        loglevel = logging.ERROR
    if args.log == "critical":
        loglevel = logging.CRITICAL

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=loglevel)

    logging.debug("Numpy      version: %s" % (np.__version__))
    logging.debug("Matplotlib version: %s" % (matplotlib.__version__))
    logging.debug("Cartopy    version: %s" % (cartopy.__version__))

    test = args.test
    if args.storm:
        reproject_region("storm")
