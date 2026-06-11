import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.storage.native import get_assets_root_path

assets_root_path = get_assets_root_path()
asset_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
stage_utils.add_reference_to_stage(usd_path=asset_path, path="/World/Arm")

arm_transform = XformPrim("/World/Arm")
arm_transform.set_world_poses(positions=[0.0, 1.0, 0.0])
arm_handle = Articulation("/World/Arm")


#! query some info (Once!)
print("Number of joints:", arm_handle.num_dofs)
print("Joint names:", arm_handle.dof_names)
positions = arm_handle.get_dof_positions()
print("Current joint positions: \n", positions)


#! use callback functions to see info at each physical step
from isaacsim.core.simulation_manager import IsaacEvents, SimulationManager


def print_joint_positions_callback(dt, context):
    positions = arm_handle.get_dof_positions()
    print("Joint positions: \n", positions)


joint_logging_callback_id = None


def enable_joint_logging():
    global joint_logging_callback_id
    if joint_logging_callback_id is None:
        joint_logging_callback_id = SimulationManager.register_callback(
            print_joint_positions_callback,
            IsaacEvents.POST_PHYSICS_STEP,
        )
        print("Joint position logging enabled.")


def disable_joint_logging():
    global joint_logging_callback_id
    if joint_logging_callback_id is not None:
        SimulationManager.deregister_callback(joint_logging_callback_id)
        joint_logging_callback_id = None
        print("Joint position logging disabled.")


enable_joint_logging()


#! control the robot using Articulation API at the joint level
# Move arm to a target pose. arm_handle from add_franka_to_stage snippet.
# Franka has 9 DOFs: 7 arm joints + 2 finger joints
arm_handle.set_dof_positions([-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04])
