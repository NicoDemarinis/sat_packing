import numpy as np
import openmdao.api as om
import trimesh
from trimesh_env_setup import get_sizes_and_collisions

# This one works 

# Design vector: xmin,xmax,ymin,ymax,zmin,zmax
# Goal: maximize the box volume inside the envelope while avoiding collisions with components 
# (and maintaining an ε-clearance margin)


# ----------------- Environment from trimesh -----------------
scene = get_sizes_and_collisions(visualize=False)

ENV = np.asarray(scene["envelope_size_mm"], float)  # [Ex, Ey, Ez]
Ex, Ey, Ez = ENV.tolist()
xmin_e, ymin_e, zmin_e = 0.0, 0.0, 0.0
xmax_e, ymax_e, zmax_e = Ex, Ey, Ez

# ----------------- Component ---------------------
class BoxCost(om.ExplicitComponent):
    def setup(self):
        # decision vector: [xmin, xmax, ymin, ymax, zmin, zmax]
        self.add_input('x', val=np.zeros(6))
        self.add_output('J', val=0.0)  # scalar cost to MINIMIZE
        self.declare_partials(of='J', wrt='x', method='fd', step=5.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        xmin, xmax, ymin, ymax, zmin, zmax = x
        Lx, Ly, Lz = xmax - xmin, ymax - ymin, zmax - zmin

        # Normalize volume so weights have bite
        V_env = Ex * Ey * Ez
        V = max(Lx, 0.0) * max(Ly, 0.0) * max(Lz, 0.0)
        v = V / max(V_env, 1e-12)

        # Weights (dimensionless)
        w_dim = 10.0
        w_col = 100.0

        # Encourage positive lengths (normalized)
        g_dim = np.array([Lx/Ex, Ly/Ey, Lz/Ez], float) 

        # Build candidate box at current x
        cx, cy, cz = (xmin + xmax)/2.0, (ymin + ymax)/2.0, (zmin + zmax)/2.0
        candidate = trimesh.creation.box(extents=[max(Lx,0.0), max(Ly,0.0), max(Lz,0.0)])
        candidate.apply_translation([cx, cy, cz])

        # Components contact per face (normalized by axis length)
        cm = scene.get("cm_components", None)
        d6 = np.zeros(6, float)  # [-x, +x, -y, +y, -z, +z]
        if cm is not None:
            is_col, contacts = cm.in_collision_single(candidate, return_data=True)
            # in_collision_single(..., return_data=True) returns: 
            # is_col: any overlap?
            # contacts: list of contact objects (each has depth, normal, etc.).
            if is_col and contacts:
                cd = max(contacts, key=lambda c: c.depth)   # pick deepest contact
                depth = float(cd.depth)                     # penetration amount (mm)
                nx, ny, nz = map(float, cd.normal)          # collision normal
                ax = int(np.argmax(np.abs([nx, ny, nz])))   # dominant axis of the normal (0=x,1=y,2=z)
                sgn = np.sign([nx, ny, nz][ax])             # which side along that axis (+ or -)
                if ax == 0:      d6[1 if sgn > 0 else 0] = -depth / Ex # Divide my Ex to make it dimensionless
                elif ax == 1:    d6[3 if sgn > 0 else 2] = -depth / Ey 
                else:            d6[5 if sgn > 0 else 4] = -depth / Ez

        # --- Clearance via collision depth: inflate box by epsilon and collide ---
        epsilon = 5.0  # mm desired clearance to components
        d6_margin = np.zeros(6, float)
        if cm is not None:
            candidate_eps = trimesh.creation.box(extents=[
                # trimesh.creation.box(extents=...) makes a box centered at the origin;
                # then translate to (cx,cy,cz) so it coincides with your current candidate box’s center.
                max(Lx, 0.0) + 2*epsilon,
                max(Ly, 0.0) + 2*epsilon,
                max(Lz, 0.0) + 2*epsilon,
            ])
            candidate_eps.apply_translation([cx, cy, cz])

            hit2, contacts2 = cm.in_collision_single(candidate_eps, return_data=True)
            if hit2 and contacts2:
                cd2 = max(contacts2, key=lambda c: c.depth)
                depth2 = float(cd2.depth)  # overlap of the ε-shell
                nx2, ny2, nz2 = map(float, cd2.normal)
                ax2 = int(np.argmax(np.abs([nx2, ny2, nz2])))
                sgn2 = np.sign([nx2, ny2, nz2][ax2])

                # map to loaded face and normalize depth by axis length
                if ax2 == 0:      d6_margin[1 if sgn2 > 0 else 0] = -depth2 / max(Ex, 1e-12)
                elif ax2 == 1:    d6_margin[3 if sgn2 > 0 else 2] = -depth2 / max(Ey, 1e-12)
                else:             d6_margin[5 if sgn2 > 0 else 4] = -depth2 / max(Ez, 1e-12)
        # ------------------------------------------------------------------------

        # Combine direct-contact and clearance penalties (more negative = worse)
        d6_total = np.minimum(d6, d6_margin)

        # Final objective: maximize volume, penalize negative lengths and contacts/clearance
            # penalty methods for inequality constraints (if constraint ≥ 0 → fine, else penalize how negative it is)
            # **2 to make the penalty smooth and differentiable
            # .sum makes sure everything gets added but from the different directions
        J = -v \
            + w_dim * float((np.maximum(0.0, -g_dim)**2).sum()) \
            + w_col * float((np.maximum(0.0, -d6_total)**2).sum())

        outputs['J'] = J

# ----------------- Problem wiring ----------------
# MDAO Stuff:
# Initial guess: 10 mm cube starting at (10,10,10) mm
x0 = np.array([10, 20, 10, 20, 10, 20], float)

prob = om.Problem()
ivc  = om.IndepVarComp()
ivc.add_output('x', val=x0)
prob.model.add_subsystem('ivc', ivc)
prob.model.add_subsystem('cost', BoxCost())
prob.model.connect('ivc.x', 'cost.x')

# Driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['tol'] = 1e-9
prob.driver.options['debug_print'] = []   # no per-iteration spam
prob.driver.opt_settings['maxiter'] = 200
prob.driver.opt_settings['disp'] = False
prob.driver.opt_settings['eps'] = 10.0    # Larger FD step for SLSQP

# Bounds (from trimesh envelope)
lower = np.array([xmin_e, xmin_e, ymin_e, ymin_e, zmin_e, zmin_e], float)
upper = np.array([xmax_e, xmax_e, ymax_e, ymax_e, zmax_e, zmax_e], float)

# Scale each coordinate to ~0..1 so optimizer steps are healthy
scaler = np.array([1/Ex, 1/Ex, 1/Ey, 1/Ey, 1/Ez, 1/Ez], float)
prob.model.add_design_var('ivc.x', lower=lower, upper=upper, scaler=scaler)

# Objective
prob.model.add_objective('cost.J')

# Force TOTAL-derivative FD so the driver sees a gradient
prob.model.approx_totals(method='fd', step=5.0)

# ----------------- Run --------------------------
prob.setup()

# Evaluate once at x0
prob.run_model()
print("\n--- Initial ---")
print("x0:", prob.get_val('ivc.x').ravel())
print("J0:", float(prob.get_val('cost.J')))

# Optimize
prob.run_driver()
x_star = prob.get_val('ivc.x').ravel()
J_star = float(prob.get_val('cost.J'))

Lx_star = x_star[1] - x_star[0]
Ly_star = x_star[3] - x_star[2]
Lz_star = x_star[5] - x_star[4]
V_star  = Lx_star * Ly_star * Lz_star
cx_star, cy_star, cz_star = (x_star[0] + x_star[1]) / 2.0, (x_star[2] + x_star[3]) / 2.0, (x_star[4] + x_star[5]) / 2.0

print("\n--- Optimized ---")
print("x*:", x_star)
print(f"Sizes*: Lx={Lx_star:.3f} mm, Ly={Ly_star:.3f} mm, Lz={Lz_star:.3f} mm")
print(f"Center*: ({cx_star:.3f}, {cy_star:.3f}, {cz_star:.3f}) mm")
print(f"Min corner*: ({x_star[0]:.3f}, {x_star[2]:.3f}, {x_star[4]:.3f}) mm")
print(f"Volume*: {V_star:.3f} mm^3")
print("J*:", J_star)
