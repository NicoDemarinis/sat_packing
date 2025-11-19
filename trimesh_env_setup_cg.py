import numpy as np
import trimesh
import trimesh.collision as tc
# from vedo import Box, Mesh, Plotter 

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
        "cm_obstacles": None,   # reserved if you add fixed keep-outs later
    }

    # --- Build local meshes (rotation applied), keep offsets as initial guess only ---
    for comp in components:
        mesh_local = trimesh.load_mesh(comp['file'])

        # Apply fixed rotations about the mesh centroid
        rc = mesh_local.centroid.copy()
        rx, ry, rz = comp.get('rotation', [0, 0, 0])
        if rx:
            mesh_local.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.radians(rx), [1, 0, 0], point=rc
                )
            )
        if ry:
            mesh_local.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.radians(ry), [0, 1, 0], point=rc
                )
            )
        if rz:
            mesh_local.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.radians(rz), [0, 0, 1], point=rc
                )
            )

        bmin_local, bmax_local = mesh_local.bounds
        cg_local = mesh_local.center_mass

        results["components"].append({
            "name": comp['name'],
            "mesh": mesh_local,                           # rotated, untranslated
            "mass": float(comp.get('mass', 0.0)),         # kg
            "xyz0_mm": list(map(float, comp.get('offset', [0, 0, 0]))),
            "cg_local_mm": cg_local.astype(float).tolist(),
            "bounds_local": (bmin_local.astype(float), bmax_local.astype(float)),
            "color": comp.get('color', 'silver'),
        })

    return results

def get_cg(visualize=False):
    """
    Wrapper for compatibility with your existing code name.
    Also builds a collision manager for components in their *local* frame.
    """
    scene = get_sizes_and_collisions(visualize=visualize)

    cm = tc.CollisionManager()
    for comp in scene["components"]:
        # NOTE: meshes are local (no translation)
        cm.add_object(comp["name"], comp["mesh"])
    scene["cm_components"] = cm

    return scene
