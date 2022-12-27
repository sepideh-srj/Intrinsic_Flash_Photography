# Dataset Preparation
Our dataset contains flash/no-flash pairs from [Flash and Ambient Illuminations Dataset
](http://yaksoy.github.io/faid/), [A Dataset of Multi-Illumination Images in the Wild
](https://projects.csail.mit.edu/illumination/) and [DeepFlash](http://graphics.unibas.it/www/flash_no_flash/index.md.html).
## Flash and Ambient Illuminations Dataset
Download all the illuminations from the main dataset and extract them into a single folder called 'Illuminations'. 
Download and extract the exif files.
Map the illuminations to XYZ color space by getXYZ.m function with the color matrix available in exif data of the PNG files. 
 
