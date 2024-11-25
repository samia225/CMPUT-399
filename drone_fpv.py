import pybullet as p
import pybullet_data
import time
import json
import os
import numpy as np
import math
from typing import Sequence, Tuple

# Initialize PyBullet in GUI mode
p.connect(p.GUI)

# Load plane and drone
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")  # Ground plane for the drone to interact with

# Create a simple drone body (box shape)
drone_body = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.02])
drone = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=drone_body, basePosition=[0, 0, 1])

# Initial state and orientation
initial_state = [0, 0, 1, 0, 0, 0]  # x, y, z, roll, pitch, yaw
state = initial_state.copy()  # Current state
THRUST_SCALING = 0.1  # Scale down thrust to reduce rapid upward movement


def map_4d_to_12d(input_vector):
    """
    Map 4D input (pitch, roll, yaw, thrust) to a 12D velocity output.
    """
    thrust = input_vector[3] * THRUST_SCALING
    return [
        input_vector[0], input_vector[1], thrust,  # x, y, z velocity (scaled thrust)
        input_vector[0] * 0.1, input_vector[1] * 0.1, input_vector[2] * 0.1,  # Orientation roll, pitch, yaw velocities
        0, 0, 0,  # Placeholder for linear accelerations
        0, 0, 0   # Placeholder for angular accelerations
    ]


def update_drone_state(state, velocity):
    """
    Updates the drone's position and orientation based on input state and velocity vectors.
    """
    position = [state[i] + velocity[i] for i in range(3)]  # Update position
    orientation = [state[i + 3] + velocity[i + 3] for i in range(3)]  # Update orientation
    quaternion = p.getQuaternionFromEuler(orientation)

    # Debugging: Print position and orientation
    print(f"Position: {position}, Orientation: {orientation}")

    # Update PyBullet's position and orientation
    p.resetBasePositionAndOrientation(drone, position, quaternion)
    state[:3] = position
    state[3:] = orientation


def compute_view_matrix_from_cam_location(
    cam_pos: Sequence[float],
    cam_quat: Sequence[float],
    target_distance: float = 10.0
) -> Sequence[float]:
    """
    Compute the camera's view matrix based on its position and quaternion orientation.

    Parameters
    ----------
    cam_pos : Sequence[float]
        The position of the camera (x, y, z).
    cam_quat : Sequence[float]
        The orientation of the camera as a quaternion (x, y, z, w).
    target_distance : float
        The distance from the camera to the target point.

    Returns
    -------
    view_matrix : Sequence[float]
        The view matrix as a flattened list with 16 elements.
    """
    cam_rot_mat = p.getMatrixFromQuaternion(cam_quat)
    forward_vec = [cam_rot_mat[0], cam_rot_mat[3], cam_rot_mat[6]]
    cam_up_vec = [cam_rot_mat[2], cam_rot_mat[5], cam_rot_mat[8]]
    cam_target = [
        cam_pos[0] + forward_vec[0] * target_distance,
        cam_pos[1] + forward_vec[1] * target_distance,
        cam_pos[2] + forward_vec[2] * target_distance,
    ]
    view_mat = p.computeViewMatrix(cam_pos, cam_target, cam_up_vec)
    return view_mat


def load_velocities_from_file():
    filename = os.path.join(os.path.dirname(__file__), "4d_velocity.json")
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return data["velocities"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print("Error loading velocities from file:", e)
        return {}


def save_velocity_to_file(velocity_name, velocity_vector):
    filename = os.path.join(os.path.dirname(__file__), "4d_velocity.json")
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"velocities": {}}

    # Add the new velocity and save
    data["velocities"][velocity_name] = velocity_vector
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Velocity '{velocity_name}' saved successfully.")


def user_menu():
    """
    Display a menu for user to select, add, reset to previous state, or exit.
    """
    while True:
        print("\nMenu:")
        print("1. Select an existing velocity")
        print("2. Add a new velocity")
        print("3. Reset to original position")
        print("4. Exit simulation")
        choice = input("Enter your choice (1/2/3/4): ")

        if choice == "1":
            velocity_options = load_velocities_from_file()
            if not velocity_options:
                print("No velocities available. Add a new velocity first.")
                continue

            print("Available velocities:", list(velocity_options.keys()))
            selected_velocity_name = input("Enter the name of the velocity to use: ")
            if selected_velocity_name in velocity_options:
                return map_4d_to_12d(velocity_options[selected_velocity_name])
            else:
                print("Invalid selection. Try again.")

        elif choice == "2":
            velocity_name = input("Enter a name for the new velocity: ")
            velocity_vector = input("Enter 4 values for the velocity vector (pitch, roll, yaw, thrust), separated by spaces: ")
            velocity_vector = [float(v) for v in velocity_vector.split()]
            if len(velocity_vector) == 4:
                save_velocity_to_file(velocity_name, velocity_vector)
            else:
                print("Invalid input. Please enter exactly 4 values.")

        elif choice == "3":
            print("Resetting to original position:", initial_state)
            global state
            state = initial_state.copy()  
            p.resetBasePositionAndOrientation(drone, initial_state[:3], p.getQuaternionFromEuler(initial_state[3:]))
            return [0] * 12  
        elif choice == "4":
            print("Exiting simulation.")
            exit(0)

        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


class BulletCameraDevice:
    def __init__(self, z_near: float, z_far: float, res_w: int = 640, res_h: int = 480, fov_w: float = 60.0):
        self._z_near = z_near
        self._z_far = z_far
        self._width = res_w
        self._height = res_h
        self._fov_width_deg = fov_w
        self._focal_length_pix = (float(self._width) / 2) / math.tan((math.pi * self._fov_width_deg / 180) / 2)
        self._fov_height_deg = (math.atan((float(self._height) / 2) / self._focal_length_pix) * 2 / math.pi) * 180
        self._projection_matrix = self.compute_projection_matrix()

    def compute_projection_matrix(self) -> np.ndarray:
        """
        Compute projection matrix from camera attributes using PyBullet.
        """
        return p.computeProjectionMatrixFOV(
            fov=self._fov_height_deg,
            aspect=float(self._width) / float(self._height),
            nearVal=self._z_near,
            farVal=self._z_far,
        )

    def cam_capture(self, view_matrix: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture an image in PyBullet's virtual space.

        Parameters
        ----------
        view_matrix : Sequence[float]
            The camera's view matrix as a flattened list with 16 elements.

        Returns
        -------
        rgb_out : np.ndarray
            The captured RGB image.
        depth_out : np.ndarray
            The depth image in meters.
        mask_out : np.ndarray
            The segmentation mask image.
        """
        w, h, rgb, depth, mask = p.getCameraImage(
            width=self._width,
            height=self._height,
            viewMatrix=view_matrix,
            projectionMatrix=self._projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb = np.asarray(rgb).reshape(h, w, 4)[:, :, :3].astype(np.uint8)  
        depth = np.asarray(depth).reshape(h, w)
        depth_out = self._z_far * self._z_near / (self._z_far - (self._z_far - self._z_near) * depth)
        mask_out = np.asarray(mask).reshape(h, w)
        return rgb, depth_out, mask_out



camera = BulletCameraDevice(
    z_near=0.1, 
    z_far=100, 
    res_w=640, 
    res_h=480, 
    fov_w=60.0
)


fps = 30
velocity = user_menu()  # Get initial velocity using user menu

while True:
    
    update_drone_state(state, velocity)
    p.stepSimulation()

    
    pos, orn = p.getBasePositionAndOrientation(drone)
    view_matrix = compute_view_matrix_from_cam_location(cam_pos=pos, cam_quat=orn)
    rgb_image, depth_map, mask = camera.cam_capture(view_matrix)


    import matplotlib.pyplot as plt
    plt.imshow(rgb_image)
    plt.axis("off")
    plt.pause(0.001)

    if input("Press 'q' to quit or Enter to continue: ").lower() == 'q':
        break

p.disconnect()

