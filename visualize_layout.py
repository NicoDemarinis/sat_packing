import numpy as np
from vedo import Box, Mesh, Plotter
from trimesh_env_setup_cg import get_cg

def visualize_result(scene, p_star):
    components = scene["components"]
    Ex, Ey, Ez = scene["envelope_size_mm"]

    env_center = [Ex/2, Ey/2, Ez/2]
    envelope = Box(pos=env_center, size=[Ex, Ey, Ez], c="blue5", alpha=0.15)
    envelope.wireframe(False)

    meshes_viz = [envelope]

    for i, comp in enumerate(components):
        m = comp["mesh"].copy()
        m.apply_translation(p_star[i])
        mv = Mesh(m, c=comp["color"], alpha=0.9).lighting("plastic")
        meshes_viz.append(mv)

    Plotter(title="Optimized Satellite Layout", axes=1, bg="white").show(
        *meshes_viz, viewup="z", interactive=True
    )


if __name__ == "__main__":
    # Load scene and optimized p_star
    scene = get_cg(visualize=False)

    # Paste solution here
    p_star = np.array([
[112.99277311, 264.99411356 ,250.        ], 
[351.78753073 ,379.80031926, 350.        ], 
[219.54744698, 348.12231633, 350.        ], 
[195.60406092 ,507.13992069 ,  0.        ] 
    ])

    visualize_result(scene, p_star)
