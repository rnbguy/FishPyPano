# FishPyPano
Stitch two fish eye images to get 360° panorama

FishPyPano works with ***Mi Sphere dual*** fisheye photos. I intend to make it work for any dual fisheye images.

It takes two >180° HFOV/VFOV fisheye images and stitch them.

### Setup
Before running, set up requirements, executing `pip install -r requirements.txt`

### Running
The code offers two functions.
1. Calibration - You have to do this one time. You give a list of matched points in the image from JSON file and it calibrates the lens parameters. After calibration it will generate a json file with the calibrated parameters.
   - I have cooked up a [html](https://codepen.io/ranadeep/full/XVaPwy/) page which you can use to generate the json file.
2. Stitch - Pass the image and wait for the magic.
   - For now, it dumps all the mid way images for debugging.
