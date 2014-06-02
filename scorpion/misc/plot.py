import pdb
import numpy as np
import matplotlib.pyplot as plt
from pylab import get_cmap


from gensinglecluster import gen_points
from scorpion.db import connect


db = connect("postgresql://localhost/intel")
res = db.execute("select light, voltage, temp from readings where light > 0 and humidity > 0 and temp >= 0 and temp <= 300 and sensor >= 0 and sensor < 100 and voltage >= 0.1 and voltage < 10;")
pts = [list(r) for r in res]
pts = np.array(pts)
mask = np.equal(pts[:,0], None) | np.equal(pts[:,1], None) | np.equal(pts[:,2], None)
mask = np.invert(mask)
pts = pts[mask]
pts[:-1] += min(pts[:,-1])
print min(pts[:,0]), max(pts[:,0])
print min(pts[:,1]), max(pts[:,1])

uh, sh, uo, so = 10, 10, 80, 10
#outlier_box, super_box, schema,  pts = gen_points(500, 2, 2, .25, uh, sh, uo, so)

#pts = np.array(list(pts) )

from pylab import *
cdict = {'red': (
  (0.0, 0.92, 0.92),
  (0.4, 0.88, 0.88),
  (0.6, 1., 1.),
  (0.8, 1.0, 1.0),
  (1.0, 1.0, 1.0)),
'green': (
  (0.0, 0.92, 0.92),
  (0.4, 0.88, 0.88),
  (0.6, .5, .5),
  (0.8, 0.0, 0.0),
  (1.0, 0.0, 0.0)),
'blue': (
  (0.0, 0.92, 0.92),
  (0.4, 0.88, 0.88),
  (0.6, .0, .6),
  (0.8, 0.0, 0.0),
  (1.0, 0.0, 0.0))}

cdict = {'red': (
  (0.0, 0.6, .6),
  (1.0, 1.0, 1.0)),
'green': (
  (0.0, 0.6, .6),
  (1.0, 0.0, 0.0)),
'blue': (
  (0.0, 0.6, .6),
  (1.0, 0.0, 0.0))}


my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

#pts[(pts[:,0] >= 50) & (pts[:,0] <= 80) & (pts[:,1] >= 50) & (pts[:,1] <= 80) & (pts[:,-1] < 70),-1] =  60

fig = plt.figure(figsize=(10, 10))
ax = fig.add_axes([0., 0., 1., 1.])
ax.axis('off')
ax.set_axis_off()
#ax.scatter(pts[:,0], pts[:,1], alpha=0.7, lw=0, s=pts[:,-1]**1.2+30, c=pts[:,-1], cmap=my_cmap)#get_cmap('YlOrRd'))
ax.scatter(pts[:,0], pts[:,1], alpha=0.7, lw=0, c=pts[:,-1], cmap=my_cmap)#get_cmap('YlOrRd'))
#ax.set_xlim(-0, 100)
#ax.set_ylim(-0, 100)
extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('plot.png', format='png', bbox_inches=extent, dpi=50, pad_inches=0)
exit()

print pts
#plt.gray()
plt.axes().set_ylim([-30, 130])
plt.gcf().set_size_inches(20,10)
plt.savefig('plot.png', size=(20, 20))