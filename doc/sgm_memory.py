
depth = 192
height = 8708
width = 11608
direction = 1

cost_agg = direction*depth*height*width
cost_agg *= 4  # float
cost_agg /= 1024*1024  # MB

print('cost_agg = {:.2f} MB'.format(cost_agg))
print('cost_agg = {:.2f} GB'.format(cost_agg/1024))