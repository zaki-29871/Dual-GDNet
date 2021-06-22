# height, width = utils.dataset.FlyingThings3D.image_size
height, width = 1242, 375
downsample = 32

print(f'height = {int(height / downsample) * downsample}')
print(f'width = {int(width / downsample) * downsample}')