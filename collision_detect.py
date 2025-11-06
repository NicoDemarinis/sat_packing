import numpy as np
import trimesh
from vedo import Box, Mesh, show, Plotter, Point, Arrow
import trimesh.collision as tc
from vedo.utils import vedo2trimesh 

# List of components [x,y,z] - Mass (kg)
# Green is moving / Red is Stationary / Fuel Tanks are Yellow
# rotation : [x_rot, y_rot, z_rot] in degrees}

components = [
    {'name': 'battery1', 'file': 'STL_Parts\Battery_8S6P_V1.STL', 'mass': 5.9, 'color': 'green', 'offset': [0, 150, -250],'rotation': [90, 0, 0]},
    {'name': 'battery2', 'file': 'STL_Parts\Battery_8S6P_V1.STL', 'mass': 5.9, 'color': 'green', 'offset': [0, 200, -250],'rotation': [90, 0, 0]},
]

# Envelope size - Sat Bus
envelope_size = np.array([711.2, 965.2, 609.6]) # mm - (28"x38"x24")
envelope = Box(pos=[0, 0, 0], size=envelope_size, c='blue5', alpha=0.2).wireframe(False) # From vedo

# For CG calculation (Initialization)
total_mass = 0.0
weighted_centroids = []
meshes = [envelope]
#cm = CollisionManager()
cm = tc.CollisionManager()

# Mesh each component (trimesh and vedo)
for comp in components:
    mesh_tm = trimesh.load_mesh(comp['file'])
    rotation_center = mesh_tm.centroid.copy() # Make a copy of the centriod location - may not need

    # Rotate each componet
    rx, ry, rz = comp.get('rotation', [0, 0, 0])
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0], point=rotation_center))
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0], point=rotation_center))
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1], point=rotation_center))

    # Translate 
    mesh_tm.apply_translation(comp['offset'])
    centroid = mesh_tm.centroid

    # Add to the mesh
    mesh_vd = Mesh(mesh_tm, c=comp['color'], alpha=0.4).lighting('plastic')
    meshes.append(mesh_vd)  

    # Collision detection 
    cm.add_object(comp['name'], mesh_tm)

 
#Old methods - Just mentions collision
# if cm.in_collision_internal():
#     print("Collision detected")
#     print(cm.in_collision_internal(return_names=True)) # shows which ones collide
# else:
#     print("No collisions")

# Check for collisions
cm.add_object("envelope", vedo2trimesh(envelope)) # Add the envelope
is_col, contact_list = cm.in_collision_internal(return_data=True)
if not is_col or not contact_list:
    print("No collisions")
else:
    # pick the deepest contact overall
    cd = max(contact_list, key=lambda c: c.depth)
    n1, n2 = list(cd.names)  # names may be a set
    print("Collisions detected:")
    print(f"{n1},{n2}")
    print(f"  Depth:  {cd.depth:.3f}")
    print(f"  Point:  {cd.point}")
    print(f"  Normal: {cd.normal}")

# Display
plotter = Plotter(title='Satellite Packing', axes=1, bg='white')
plotter.show(*meshes, viewup='z', interactive=True)

#note for later (minimum translation vector)
#MTV = cd.normal * cd.depth   # move the first body by -MTV or the second by +MTV

# For Debugging
# if __name__ == "__main__":
#     import code
#     code.interact(local=locals())