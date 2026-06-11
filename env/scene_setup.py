import gc

from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
from isaacsim.core.experimental.objects import Cube, DistantLight, GroundPlane
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim

#! for using raw USD API
import omni.usd
from pxr import Gf, UsdGeom


#* this is useful
def clear_previous_handles():
    for handle_name in (
        "distant_light",
        "yellow_material",
        "cyan_material",
        "visual_cube",
        "test_cube",
    ):
        globals().pop(handle_name, None)
    gc.collect()


def main():
    clear_previous_handles()
    stage_utils.create_new_stage()

    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])

    distant_light = DistantLight("/World/DistantLight")
    distant_light.set_intensities(300)

    yellow_material = PreviewSurfaceMaterial("/Materials/yellow")
    yellow_material.set_input_values("diffuseColor", [1.0, 1.0, 0.0])

    cyan_material = PreviewSurfaceMaterial("/Materials/cyan")
    cyan_material.set_input_values("diffuseColor", [0.0, 1.0, 1.0])

    visual_cube = Cube(
        paths="/World/visual_cube",
        positions=[0, 0.5, 0.5],
        sizes=0.3,
    )
    visual_cube.apply_visual_materials(yellow_material)

    test_cube = Cube(
        paths="/World/test_cube",
        positions=[0, -0.5, 0.5],
        sizes=0.3,
    )
    test_cube.apply_visual_materials(cyan_material)

    #! using raw USD API to create a cube with more control
    stage = omni.usd.get_context().get_stage()
    path = "/visual_cube_usd"
    cube_geom = UsdGeom.Cube.Define(stage, path)
    cube_prim = stage.GetPrimAtPath(path)
    size = 0.5
    offset = Gf.Vec3f(1.5, -0.2, 1.0)
    cube_geom.CreateSizeAttr(size)
    if not cube_prim.HasAttribute("xformOp:translate"):
        UsdGeom.Xformable(cube_prim).AddTranslateOp().Set(offset)
    else:
        cube_prim.GetAttribute("xformOp:translate").Set(offset)

    #! turn on physics and collision
    RigidPrim(paths="/World/test_cube")
    GeomPrim(paths="/World/test_cube", apply_collision_apis=True)
    
    #! move, rotate and scale the cube
    translate_offset = [1.5, 1.2, 1.0]
    orientation_offset = [0.7, 0.7, 0, 1]
    scale = [1, 1.5, 0.2] #*Ok, this stretches the object
    cube_prim = XformPrim(paths="/World/test_cube")
    cube_prim.set_world_poses(translate_offset, orientation_offset)
    cube_prim.set_local_scales(scale)

    #! moving an object using raw USD API
    cube_prim = stage.GetPrimAtPath("/visual_cube_usd")
    translate_offset = Gf.Vec3f(1.5, -0.2, 1.0)
    rotate_offset = Gf.Vec3f(90, -90, 180)  # Note this is in degrees.
    scale = Gf.Vec3f(1, 1.5, 0.2)

    if not cube_prim.HasAttribute("xformOp:translate"):
        UsdGeom.Xformable(cube_prim).AddTranslateOp().Set(translate_offset)
    else:
        cube_prim.GetAttribute("xformOp:translate").Set(translate_offset)

    if not cube_prim.HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(cube_prim).AddRotateXYZOp().Set(rotate_offset)
    else:
        cube_prim.GetAttribute("xformOp:rotateXYZ").Set(rotate_offset)

    if not cube_prim.HasAttribute("xformOp:scale"):
        UsdGeom.Xformable(cube_prim).AddScaleOp().Set(scale)
    else:
        cube_prim.GetAttribute("xformOp:scale").Set(scale)


main()
