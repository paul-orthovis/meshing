import numpy as np
import zmesh

labels = np.zeros( (10, 10, 10), dtype=np.uint32)
labels[1:-1, 2:-2, 2:-2] = 1

mesher = zmesh.Mesher( (1,1,2) )
mesher.mesh(labels)
mesh = mesher.get(1, normals=False, reduction_factor=0)

print(mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
