# Read sensors of boats, and move my boat
import numpy as np
import math
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
from qp_colreg_controller import QPColregController

# Kinematics transformation from u_otp with x and y coordinates to Throttle and steering
def map_velocity_to_differential_inputs(u_opt, max_speed=2.5):

    vx_des, vy_des = u_opt[0], u_opt[1]
    speed_des = math.hypot(vx_des, vy_des)
    
    if speed_des < 0.05:
        return np.array([0.0, 0.0], dtype=np.float32)
    
    angle_des_rad = math.atan2(vx_des, vy_des) # Calcolate the angle to point the directn
    kp_steering = 2.5
    steering = np.clip(kp_steering * angle_des_rad, -1.0, 1.0)
    
    cos_error = max(0.0, math.cos(angle_des_rad))
    throttle = np.clip((speed_des / max_speed) * cos_error, -1.0, 1.0)
    
    return np.array([throttle, steering], dtype=np.float32)

def main():
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=1.0)

    # Communication with ML-Agemts
    env_params_channel = EnvironmentParametersChannel()
    env_params_channel.set_float_parameter("eval_episode_seed", 15.0) #where is the tagret
    env_params_channel.set_float_parameter("curriculumStage", 2.0)

    print("Waiting for connection to Unity")
    
    env = UnityEnvironment(side_channels=[config_channel, env_params_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    # Initialize the controller with safety parameters
    # Initialize the controller with safety parameters and disable or enable COLREGs
    controller = QPColregController(d_safe=3.0, gamma=1.2, v_max=2.5, enable_colregs=True)

    print(f"Connected to: {behavior_name}")
    print("Avoidance logic active. Vessel (GPS) and buoy (Laser) monitoring in progress...")
    print("-" * 60)

    try:
        step_count = 0
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:
                # --- 1. Separate sensors values ---
                obs_gps = None
                obs_ray = None
                
                for sensor_data in decision_steps.obs:
                    if sensor_data.shape[1] >= 20: 
                        obs_gps = sensor_data[0]  # Boats and target (GPS) Data
                    elif sensor_data.shape[1] == 14:
                        obs_ray = sensor_data[0]  # Buey data (Laser/Raycast)

                if obs_gps is None: obs_gps = decision_steps.obs[0][0]

                # Nominal velocity
                target_dir = obs_gps[0:2]
                v_nominal = target_dir * 2.5
                v_agent_local = obs_gps[3:5] * 2.5
                
                intruders = []
                
                # --- 2. MOBILE SHIP LOGIC (from GPS) ---
                if len(obs_gps) >= 13 and obs_gps[8] < 0.99:
                    pos = obs_gps[6:8] * (obs_gps[8] * 43.0)
                    vel = (obs_gps[9:11] * 5.0) + v_agent_local
                    intruders.append((pos, vel))
                
                if len(obs_gps) >= 20 and obs_gps[15] < 0.99:
                    pos = obs_gps[13:15] * (obs_gps[15] * 43.0)
                    vel = (obs_gps[16:18] * 5.0) + v_agent_local
                    intruders.append((pos, vel))

                # --- 3. STATIC BUOY LOGIC (from corrected Raycast) ---
                if obs_ray is not None:
                    # Array of 14 elements = 7 rays * 2 values ​​for ray: [hit_it, fraction_distance]
                    angles_deg = [-90, -45, -15, 0, 15, 45, 90]
                    ray_max_dist = 20.0  # Maximum laser length in Unity
                    
                    for i in range(7):
                        has_hit = obs_ray[i * 2]         # > 0.5 if hit a buoy, 0.0 if free
                        hit_fraction = obs_ray[i * 2 + 1] # Fraction of distance (0.0 - 1.0)
                        
                        # If has_hit indicates a real laser collision, we process the obstacle
                        if has_hit > 0.5:
                            distanza_reale = hit_fraction * ray_max_dist
                            angolo_rad = math.radians(angles_deg[i])
                            
                            # Convert polar -> Cartesian (X right/left, Y forward)
                            pos_boa_x = distanza_reale * math.sin(angolo_rad)
                            pos_boa_y = distanza_reale * math.cos(angolo_rad)
                            
                            pos_boa = np.array([pos_boa_x, pos_boa_y])
                            vel_boa = np.array([0.0, 0.0]) # Le boe sono ferme
                            
                            intruders.append((pos_boa, vel_boa))

                # --- 4. QP CONTROL ---
                u_opt = controller.compute_control(np.array([0,0]), v_nominal, intruders)
                control_actions = map_velocity_to_differential_inputs(u_opt, max_speed=2.5)

                step_count += 1
                if step_count % 10 == 0:
                    for i, (p, v) in enumerate(intruders):
                        dist = np.linalg.norm(p)
                        if dist < 8.0:
                            tipo = "Nave" if np.linalg.norm(v) > 0.1 else "Boa"
                            print(f"  [WARNING] {tipo} a {dist:.1f}m: QP maneuver in progress.")
                    
                    print(f"Target: [{target_dir[0]:.2f}, {target_dir[1]:.2f}] | Gas: {control_actions[0]:.2f}, Steering: {control_actions[1]:.2f}")

                action_tuple = ActionTuple()
                action_tuple.add_continuous(np.array([control_actions], dtype=np.float32))
                env.set_actions(behavior_name, action_tuple)

            env.step() # I order Unity to advance one visual frame

    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    main()