from matplotlib.path import Path
import numpy as np

points = [(1,1), (1,2), (2,3), (3,3), (4,3), (4,2), (4,1), (3,1), (2,1)]
poly_path = Path(points)
y, x = np.mgrid[:10,:10]
print ("y is")
print (y)
print ("x is")
print (x)
print ("after reshape")
print (x.reshape(-1, 1))
coors = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
print (coors)
mask = poly_path.contains_points(coors).reshape(10, 10)
print (mask)
for x,y in points:
    mask[y,x] = 1
print (mask)

