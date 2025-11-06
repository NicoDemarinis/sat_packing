import numpy as np
import trimesh
from vedo import Box, Mesh, show, Plotter, Point, Arrow
import trimesh.collision as tc
from vedo.utils import vedo2trimesh 

# Helper function to make a trimesh box at a center
def make_trimesh_box(size_xyz, center=(0.0, 0.0, 0.0)):
    tm = trimesh.creation.box(extents=np.asarray(size_xyz, float))
    tm.apply_translation(np.asarray(center, float))
    return tm

# List of components [x,y,z] - Mass (kg)
# Green is moving / Red is Stationary / Fuel Tanks are Yellow
# rotation : [x_rot, y_rot, z_rot] in degrees}

components = [
    {'name': 'battery1', 'file': 'STL_Parts\\Battery_8S6P_V1.STL', 'mass': 5.9, 'color': 'green', 'offset': [0, 150, -250],'rotation': [90, 0, 0]},
    # {'name': 'battery2', 'file': 'STL_Parts\\Battery_8S6P_V1.STL', 'mass': 5.9, 'color': 'green', 'offset': [0, 200, -250],'rotation': [90, 0, 0]},
]

# Envelope size - Sat Bus
envelope_size = np.array([711.2, 965.2, 609.6]) # mm - (28"x38"x24")
envelope = Box(pos=[0, 0, 0], size=envelope_size, c='blue5', alpha=0.2).wireframe(False) # From vedo

# For CG calculation (Initialization)
total_mass = 0.0
weighted_centroids = []
meshes = [envelope]
cm = tc.CollisionManager()

# Mesh each component (trimesh and vedo)
for comp in components:
    mesh_tm = trimesh.load_mesh(comp['file'])
    rotation_center = mesh_tm.centroid.copy() # Make a copy of the centroid location - may not need

    # Rotate each component
    rx, ry, rz = comp.get('rotation', [0, 0, 0])
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0], point=rotation_center))
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0], point=rotation_center))
    mesh_tm.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1], point=rotation_center))

    # Translate 
    mesh_tm.apply_translation(comp['offset'])
    centroid = mesh_tm.centroid

    # Add to the mesh (visualization)
    mesh_vd = Mesh(mesh_tm, c=comp['color'], alpha=0.4).lighting('plastic')
    meshes.append(mesh_vd)  

    # Collision detection (add to manager)
    cm.add_object(comp['name'], mesh_tm)

# =====================================
# Expanding inner box (per-axis growth)

center = (0.0, 0.0, 0.0)                 # inner box center
limits = envelope_size.copy()            # cap growth by outer bus size
size    = np.array([1.0, 1.0, 1.0])      # [x,y,z] in mm
active  = [True, True, True]             # grow X, Y, Z until each collides
step    = 20.0                           # initial step (mm)
tol     = 0.5                            # stop when step < tol
max_iters = 5000

def collides(size_xyz):
    # test collisions against components already in cm
    return cm.in_collision_single(make_trimesh_box(size_xyz, center=center))

iters = 0
while any(active) and iters < max_iters:
    iters += 1
    progressed = False

    for axis in range(3):
        if not active[axis]:
            continue

        proposed = size.copy()
        proposed[axis] = min(size[axis] + step, limits[axis])

        if np.isclose(proposed[axis], size[axis]):
            active[axis] = False
            continue

        if collides(proposed):
            # lock this axis at current size for this step
            active[axis] = False
        else:
            size = proposed
            progressed = True

    if not progressed:
        if step > tol:
            step *= 0.5
        else:
            break

print(f"Final grown box size (mm): {size}")

# Visualize the grown box
grown_box_vd = Box(pos=center, size=size, c='green5', alpha=0.25).wireframe(False)
meshes.append(grown_box_vd)

# Add the grown box to cm so you can inspect final contacts
cm.add_object("grown_box", make_trimesh_box(size, center=center))

# Add envelope to collision manager if you want to include it in the final contact check
cm.add_object("envelope", vedo2trimesh(envelope))

# Check for collisions
is_col, contact_list = cm.in_collision_internal(return_data=True)
if not is_col or not contact_list:
    print("No collisions")
else:
    # pick the deepest contact overall
    cd = max(contact_list, key=lambda c: c.depth) #Contact Depth
    n1, n2 = list(cd.names)
    print("Collisions detected:")
    print(f"{n1},{n2}")
    print(f"  Depth:  {cd.depth:.3f}")
    print(f"  Point:  {cd.point}")
    print(f"  Normal: {cd.normal}")

# Display
plotter = Plotter(title='Satellite Packing', axes=1, bg='white')
plotter.show(*meshes, viewup='z', interactive=True)
