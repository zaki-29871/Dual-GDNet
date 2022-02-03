import numpy as np
import cv2
import os
import struct
import imagesize

original_data_folder = r'F:\Dataset\KITTI 2015 Data Augmentation\training'

class UnknownImageFormat(Exception):
    pass

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

first = True
min_width, min_height = 0, 0
max_width, max_height = 0, 0

image_2_path = os.path.join(original_data_folder, 'image_2')
files = os.listdir(image_2_path)
total_data_size = len(files)

for i in range(total_data_size):
    file_path = os.path.join(image_2_path, files[i])
    print(f'[{i + 1}/{total_data_size} {i/total_data_size:.0%}] {files[i]}')
    width, height = imagesize.get(file_path)

    if first:
        min_width = width
        max_width = width
        min_height = height
        max_height = height
        first = False
    else:
        if width < min_width:
            min_width = width
        if width > max_width:
            max_width = width
        if height < min_height:
            min_height = height
        if height > max_height:
            max_height = height

print(f'width = [{min_width}, {max_width}]')
print(f'height = [{min_height}, {max_height}]')