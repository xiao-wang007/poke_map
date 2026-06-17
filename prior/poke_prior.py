import numpy as np
from vision.camera import (
    RESOLUTION,
    mask_to_contour,
    create_overhead_cameras_vectorized,
    get_camera_intrinsics,
    get_object_poses_vectorized,
    render_segmentation_ground_truth,
    segmentation_to_contour_current,
    render_goal_contour,
    stack_input,
    QNetwork,
    pixel_to_world,
    world_to_pixel,
)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm

def sample_pixel_on_farside(contour_pixel_coords, obj_init_CoM_pixel, direction, num_samples=20, w=0.1):
    weights = []
    for coord in contour_pixel_coords:
        offset = coord - obj_init_CoM_pixel
        #* negation makes obtuse angle (far) positive, acute angle (near) negative
        #* then cap at 0. This is a soft preference.
        far_side_score = max(0, -np.dot(offset, direction))

        #* converts geometric score into a sampling weight
        #* baseline = 1.0 ensures near-side pixels still have non-zero chance of
        #* being sampled.
        weights.append(1.0 + w * far_side_score)
    
    weights = np.array(weights)
    weights /= np.sum(weights)  # Normalize to sum to 1

    sampled_indices = np.random.choice(
        len(contour_pixel_coords), 
        size=num_samples, 
        p=weights,
        replace=False)

    sampled_pixels = contour_pixel_coords[sampled_indices]
    return sampled_pixels  


def simulate_poke(isaac_world, obj_name, world_xy, direction):
    pass


#* ================================================================
#*  compute prior 
#* ================================================================

def compute_prior(
    seg_map: np.ndarray,          # (H, W) int — from render_segmentation_ground_truth
    env_poses: dict,              # object_name → (positions, quaternions) — already available
    target_positions: dict,       # object_name → (N, 3) target poses
    K: np.ndarray,
    isaac_world,                  # Isaac Sim handle for physics stepping
    resolution: tuple = RESOLUTION,
) -> np.ndarray:
    """
    Returns: prior heatmap (H, W) float32 in [0, 1]
    """
    prior = np.zeros(resolution, dtype=np.float32)

    obj_id = 0
    for obj_name in env_poses:
        positions = env_poses[obj_name][0]
        targets = target_positions[obj_name]
        obj_id += 1

        # Extract per-object binary mask from seg_map
        obj_mask = (seg_map == obj_id)

        # Extract contour from per-object mask 
        obj_contour, centre_pixel = mask_to_contour(obj_mask)

        #* this is the metric part to build the prior heatmap
        obj_pos = positions[0, :2]
        target_pos = targets[0, :2]
        direction = normalize(target_pos - obj_pos)

        #* Sample contour pixels
        contour_pixels = np.argwhere(obj_contour)
        sampled_pixels = sample_pixel_on_farside(contour_pixels, centre_pixel, direction)

        for pixel in sampled_pixels:
            world_xy = pixel_to_world(pixel, K)
            # Run physics probe in Isaac Sim
            score = simulate_poke(isaac_world, obj_name, world_xy, direction)
            prior[tuple(pixel)] = max(prior[tuple(pixel)], score)

    return prior


#* Sub-modules ====================================================

