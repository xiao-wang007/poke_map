Read [](file:///home/xiao/0_codes/poke_map/env/add_finger_robot.py#1-1), lines 1 to 300

Good — the file is clean and correct. Here's the regenerated tutorial reflecting exactly what's in the working code:

---

## Tutorial: Manual Link + Joint Articulation in Isaac Sim (USD Physics + Experimental API)

### The two-layer architecture

```
─── BEFORE play() ──────────────────────────────────────────────
USD Stage  (pxr)         read once at sim start
  PrismaticJoint
    DriveAPI stiffness ──▶ loaded into physics tensor gains
    DriveAPI damping   ──▶ loaded into physics tensor gains
    TargetPosition         NOT used after sim starts

─── AFTER play() ────────────────────────────────────────────────
Physics tensor  (Isaac Sim experimental API)
  articulation.set_dof_gains(...)           ← override gains
  articulation.set_dof_position_targets()   ← command joints
```

After `play()`, USD attribute changes are **frozen**. All control must go through the tensor API.

---

### Step 1 — Articulation root

```python
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf

stage = stage_utils.get_current_stage(backend="usd")

root_prim = stage.GetPrimAtPath("/World/MyRobot")
UsdGeom.Xform.Define(stage, "/World/MyRobot")
UsdPhysics.ArticulationRootAPI.Apply(root_prim)

# Disable sleep — prevents PhysX from freezing the articulation when idle
physx_api = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
physx_api.CreateSleepThresholdAttr(0.0)
```

---

### Step 2 — Links with explicit mass

Every link needs `RigidBodyAPI`. Since links start colocated (no geometry separation), always set mass **explicitly** via `MassAPI` — never rely on collision geometry for mass computation:

```python
prim = stage.GetPrimAtPath("/World/MyRobot/link_a")
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.05)  # kg
```

> ⚠️ **Do NOT add `CollisionAPI` to colocated links.** Links in a cartesian robot all start at the same world position. Overlapping collision shapes create contact forces that block joint motion (X and Y axes are especially vulnerable — gravity doesn't help break horizontal contacts).

---

### Step 3 — Fixed base joint

PhysX articulations **cannot contain kinematic links** — that's a rigid body simulation concept, not an articulation concept. Instead, anchor the base to the world with a `FixedJoint`. Leaving `body0` unset makes the world the implicit anchor:

```python
fixed_joint = UsdPhysics.FixedJoint.Define(stage, "/World/MyRobot/base_joint")
# body0 intentionally omitted — world is implicit
fixed_joint.CreateBody1Rel().SetTargets(["/World/MyRobot/base"])
fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
PhysxSchema.JointStateAPI.Apply(fixed_joint.GetPrim(), "linear")
```

---

### Step 4 — Prismatic joints with drives

```python
joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/MyRobot/x_joint")

# Register as linear DOF (fixes DOF type mismatch warnings)
PhysxSchema.JointStateAPI.Apply(joint.GetPrim(), "linear")

# Drive — gains loaded into physics tensor at sim start
drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")  # "angular" for revolute
drive.CreateTypeAttr("force")
drive.CreateStiffnessAttr(1000.0)    # N/m  (N·m/rad for revolute)
drive.CreateDampingAttr(100.0)       # N·s/m
drive.CreateMaxForceAttr(500.0)      # N
drive.CreateTargetPositionAttr(0.0)  # ⚠️ must be Created (not just Get) before use

# Body connections
joint.CreateBody0Rel().SetTargets(["/World/MyRobot/base"])
joint.CreateBody1Rel().SetTargets(["/World/MyRobot/x_link"])

# Axis — plain string, NOT a GfVec3f
joint.CreateAxisAttr("X")   # "X", "Y", or "Z" only

# Joint limits (meters for prismatic, degrees for revolute)
joint.CreateLowerLimitAttr(-0.25)
joint.CreateUpperLimitAttr(0.25)

# Local anchor frames (identity = joint at link origins)
joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
```

---

### Step 5 — Start sim, then push gains

```python
from isaacsim.core.experimental.prims import Articulation
import omni.timeline, omni.kit.app

articulation = Articulation("/World/MyRobot")

omni.timeline.get_timeline_interface().play()
omni.kit.app.get_app().update()

# ⚠️ Critical: USD DriveAPI gains are frozen at sim start.
# The physics tensor still has gains=0 unless you set them explicitly.
# Scalar broadcasts to all DOFs automatically.
articulation.set_dof_gains(
    stiffnesses=1000.0,  # N/m
    dampings=100.0,      # N·s/m
)
```

---

### Step 6 — Control and readback

```python
# Get DOF indices by name (robust to ordering)
dof_indices = articulation.get_dof_indices(articulation.dof_names)

# Command a target pose
articulation.set_dof_position_targets(
    positions=[0.1, 0.05, 0.12],   # one value per DOF, in meters
    dof_indices=dof_indices,
)

# Advance simulation
for _ in range(120):
    omni.kit.app.get_app().update()

# Read back
print(articulation.get_dof_positions())   # current positions
print(articulation.get_dof_position_targets())  # commanded targets
```

---

### Common mistakes table

| Mistake | Symptom | Fix |
|---|---|---|
| `CreateAxisAttr(Gf.Vec3f(...))` | `Type mismatch: expected TfToken` | `CreateAxisAttr("X")` — string only |
| Kinematic link inside articulation | `Articulations with kinematic bodies not supported` | Use `FixedJoint` with no body0 instead |
| `CollisionAPI` on colocated links | X/Y joints don't move; Z escapes via gravity | Remove `CollisionAPI`; use explicit `MassAPI` |
| Missing `CreateTargetPositionAttr` | `Set()` silently no-ops; joint doesn't move | Call `drive.CreateTargetPositionAttr(0.0)` at joint creation |
| Skipping `set_dof_gains` after `play()` | Joints don't respond to `set_dof_position_targets` | Call `articulation.set_dof_gains(stiffnesses=..., dampings=...)` after `play()` |
| Using `dof_indices` as property | `AttributeError: 'Articulation' has no attribute 'dof_indices'` | Use `get_dof_indices(names)` method |