# cost: b f d h w
# cost_agg: b f dir d h w
# max index: b f dir h w
# weight: b f dir 5 h w

# Disable ->
# x = SgaFunction.apply(x, g)
# x = x.max(axis=2)[0]  # max in direction axis
#
# 6297 -> 4315: 1982
# 1986 / 2 = 991 MB

# save cost agg to cpu
# 6297 MB ->  5235MB (- 531 MB for each SGA)
# Elapsed: 0:00:03.392968 (just gpu: 02.606813, +0.79s)

# load cost_aggregation
# Elapsed: 0:00:00.647753
# sga forward
# Elapsed: 0:00:00.416161
# sga backward
# Elapsed: 0:00:01.025726

# 640 -> 32 * 4 * 5
# 960 -> 48 * 4 * 5

batch = 1
channel = 32
direction = 4
depth = 192
height = 240
width = 576
weight_size = 5

down_sampling = 4

height = height // down_sampling
width = width // down_sampling
depth = depth // down_sampling

# Calculating
cost = batch*channel*depth*height*width
output_cost = batch*channel*depth*height*width
max_index = batch*channel*height*width
cost_aggregation = batch*channel*direction*depth*height*width
weight = batch*channel*direction*weight_size*height*width

# For data type
cost *= 4
output_cost *= 4
cost_aggregation *= 4
weight *= 4

# For MB
cost /= 1024*1024
output_cost /= 1024*1024
max_index /= 1024*1024
cost_aggregation /= 1024*1024
weight /= 1024*1024

print('MB size')
print('cost = {:.2f} MB'.format(cost))
print('output_cost = {:.2f} MB'.format(output_cost))
print('max_index = {:.2f} MB'.format(max_index))
print('cost_aggregation = {:.2f} MB'.format(cost_aggregation))
print('weight = {:.2f} MB'.format(weight))

sum_value = sum([cost, output_cost, max_index, cost_aggregation, weight])
print('Sum {:.2f} MB'.format(sum_value))


