import numpy as np
import trimesh
import trimesh.collision as tc
from vedo import Box, Mesh, Plotter  # only used if visualize=True

def get_sizes_and_collisions(components=None, envelope_size=None, visualize=False):
    if envelope_size is None:
        envelope_size = [711.2, 965.2, 609.6]  # mm

    if components is None:
        components = [
            {'name': 'battery1', 'file': 'STL_Parts/Battery_8S6P_V1.STL',
             'mass': 5.9, 'color': 'green', 'offset': [15, 150, 250], 'rotation': [90, 0, 0]},
            {'name': 'battery2', 'file': 'STL_Parts/Battery_8S6P_V1.STL',
             'mass': 5.9, 'color': 'green', 'offset': [350, 300, 350], 'rotation': [90, 0, 0]},
            {'name': 'computer', 'file': 'STL_Parts/Image_Processing_Computer_IPC_7000.stl',
             'mass': 5.8, 'color': 'green', 'offset': [100, 300, 350], 'rotation': [0, 0, 0]},
            {'name': 'PCDU', 'file': 'STL_Parts/Colossus_PCDU.stl',
             'mass': 7.7, 'color': 'green', 'offset': [100, 500, 0], 'rotation': [90, 0, 0]},
        ]

    results = {
        "envelope_size_mm": np.asarray(envelope_size, float),
        "epsilon_mm": 5.0,  # wall clearance
        "components": [],
        "cm_obstacles": None,  # add fixed keep-outs here
    }

    # --- Build local meshes (rotation applied), keep offsets as initial guess only ---
    for comp in components:
        mesh_local = trimesh.load_mesh(comp['file'])

        # Apply fixed rotations about the mesh centroid (so translation stays clean)
        rc = mesh_local.centroid.copy()
        rx, ry, rz = comp.get('rotation', [0, 0, 0])
        if rx: mesh_local.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rx), [1, 0, 0], point=rc))
        if ry: mesh_local.apply_transform(trimesh.transformations.rotation_matrix(np.radians(ry), [0, 1, 0], point=rc))
        if rz: mesh_local.apply_transform(trimesh.transformations.rotation_matrix(np.radians(rz), [0, 0, 1], point=rc))

        # Keep in local frame.
        # Precompute locals for bounds and CG
        bmin_local, bmax_local = mesh_local.bounds
        cg_local = mesh_local.center_mass  # in the rotated local frame

        results["components"].append({
            "name": comp['name'],
            "mesh": mesh_local,                          # rotated, untranslated
            "mass": float(comp.get('mass', 0.0)),        # kg
            "xyz0_mm": list(map(float, comp.get('offset', [0, 0, 0]))),
            "cg_local_mm": cg_local.astype(float).tolist(),
            "bounds_local": (bmin_local.astype(float), bmax_local.astype(float)),
            # (optional) keep original color for viz
            "color": comp.get('color', 'silver'),
        })

    # # --- Visualization of initial placement ---
    # if visualize:
    #     Ex, Ey, Ez = results["envelope_size_mm"].tolist()
    #     env_center = [Ex/2.0, Ey/2.0, Ez/2.0]
    #     envelope_viz = Box(pos=env_center, size=[Ex, Ey, Ez], c='blue5', alpha=0.15).wireframe(False)
    #     meshes_viz = [envelope_viz]

    #     # show each mesh translated to its xyz0_mm so you can see the starting layout
    #     for comp in results["components"]:
    #         m = comp["mesh"].copy()
    #         m.apply_translation(comp["xyz0_mm"])
    #         meshes_viz.append(Mesh(m, c=comp["color"], alpha=0.5).lighting('plastic'))

    #     Plotter(title='Satellite Packing (initial guess)', axes=1, bg='white').show(*meshes_viz, viewup='z', interactive=True)

    return results
