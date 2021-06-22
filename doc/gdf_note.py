# cost: b f d h w
# cost_agg: b f 4 d h w
# g0: b f dir h w
# filter: b f dir k k h w

batch = 1
channel = 32
direction = 4
depth = 192
height = 240
width = 576
weight_size = 3**2+1

down_sampling = 4

height = height // down_sampling
width = width // down_sampling
depth = depth // down_sampling

# Calculating
cost = batch*channel*depth*height*width
cost_aggregation = batch*channel*direction*depth*height*width
weight = batch*channel*direction*weight_size*height*width

# For data type
cost *= 4
cost_aggregation *= 4
weight *= 4

# For MB
cost /= 1024*1024
cost_aggregation /= 1024*1024
weight /= 1024*1024

print('MB size')
print('cost = {:.2f} MB'.format(cost))
print('cost_aggregation = {:.2f} MB'.format(cost_aggregation))
print('weight = {:.2f} MB'.format(weight))

sum_value = sum([cost, cost_aggregation, weight])
print('Sum {:.2f} MB'.format(sum_value))