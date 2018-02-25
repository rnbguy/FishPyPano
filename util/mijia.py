from PIL import Image
import struct
import PIL.ExifTags


def get_mijia360_gyro(img_path):
    im = Image.open(img_path)
    exif_data = im._getexif()
    USERCOMMENT = None
    for intval, strval in PIL.ExifTags.TAGS.items():
        if strval == 'UserComment':
            USERCOMMENT = intval
    raw_data = exif_data[USERCOMMENT]
    return [e[0] for e in struct.iter_unpack('f', raw_data)]
