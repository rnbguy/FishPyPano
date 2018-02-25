import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import argparse
import mijia
import angle

ap = argparse.ArgumentParser()
ap.add_argument("-i", help="dual fisheye image path", required=True)
args = ap.parse_args()

v = np.array(mijia.get_mijia360_gyro(args.i)).reshape(3, 3).T
print(v)

fig = plt.figure()
ax = fig.add_subplot(121, aspect='equal', projection='3d')

o = np.zeros_like(v)
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color=['r', 'g', 'b'])

print(angle.rotationMatToEuler(v))

ax = fig.add_subplot(122, aspect='equal', projection='3d')
v = angle.rotateX(-np.pi / 2).dot(v)
# v = v.dot(angle.rotateX(np.pi/2))
o = np.zeros_like(v)
ax.axis('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color=['r', 'g', 'b'])

print(angle.rotationMatToEuler(v))

plt.show()
