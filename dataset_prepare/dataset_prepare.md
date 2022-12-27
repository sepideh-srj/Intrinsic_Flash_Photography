# Dataset Preparation
Our dataset contains flash/no-flash pairs from [Flash and Ambient Illuminations Dataset
](http://yaksoy.github.io/faid/), [Dataset of Multi-Illumination Images in the Wild
](https://projects.csail.mit.edu/illumination/) and [DeepFlash](http://graphics.unibas.it/www/flash_no_flash/index.md.html).
## Flash and Ambient Illuminations Dataset
Download all the illuminations from the main dataset in the link and extract them into a single folder called 'Illuminations'. 

Download and extract the exif files.

Use the (IluminationsToXYZ.m) script to map the illuminations to XYZ color space with the color matrix available in exif data of the PNG files. 

Use the (FAID.py) script to convert the illuminations to linear RGB and white balance them. 

## Multi-Illumination Dataset

Download the multi illumination dataset through the link. 
 

Use the (MID.py) script to convert the illuminations to linear RGB and white balance them and put the different ambient illuminations for each scene in different sub-folders.