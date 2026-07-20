# Read sensors of boats, and move my boat
import numpy as np
import math
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple
from qp_colreg_controller import QPColregController

# metric system
class MetricsTracker:
    def __init__(self, d_safe):
        self.d_safe = d_safe
        self.violations_count = 0
        self.total_steps = 0
        self.min_distances = []

    def log_step(self, intruders):
        self.total_steps += 1
        if len(intruders) > 0:
            min_dist = min([np.linalg.norm(pos) for pos, vel in intruders])
            self.min_distances.append(min_dist)
            
            if min_dist < self.d_safe:
                self.violations_count += 1

    def print_report(self):
        print("\n" + "="*40)
        print(" REPORT METRICS ")
        print(f"Total steps simulated: {self.total_steps}")
        print(f"R1 constraint violations (dist < {self.d_safe}m): {self.violations_count}")
        if self.total_steps > 0:
            print(f"Percentage of violations: {(self.violations_count / self.total_steps) * 100:.2f}%")
        
        if self.min_distances:
            print(f"Average obstacle distance: {np.mean(self.min_distances):.2f} m")
            print(f"Standard deviation distance: {np.std(self.min_distances):.2f} m")
        print("="*40 + "\n")

# Kinematics transformation from u_opt with x and y coordinates to Throttle and steering
def map_velocity_to_differential_inputs(u_opt, max_speed=2.5):

    vx_des, vy_des = u_opt[0], u_opt[1]
    speed_des = math.hypot(vx_des, vy_des)
    
    if speed_des < 0.05:
        return np.array([0.0, 0.0], dtype=np.float32)
    
    angle_des_rad = math.atan2(vx_des, vy_des) # Calculate the angle to point the direction
    kp_steering = 2.5
    steering = np.clip(kp_steering * angle_des_rad, -1.0, 1.0)
    
    # EMERGENCY ESCAPE LOGIC (Prevents engine shutdown)
    cos_error = math.cos(angle_des_rad)
    
    if cos_error < 0:
        # If the QP asks us to back up, we give 40% throttle and turn the steering wheel to full throttle to turn around.
        throttle = 0.4
    else:
        throttle = np.clip((speed_des / max_speed) * cos_error, -1.0, 1.0)
    
    return np.array([throttle, steering], dtype=np.float32)

def main():
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=1.0)

    # Communication with ML-Agents
    env_params_channel = EnvironmentParametersChannel()
    env_params_channel.set_float_parameter("eval_episode_seed", 55.0) #where is the target
    env_params_channel.set_float_parameter("curriculumStage", 2.0)

    print("Waiting for connection to Unity")
    
    env = UnityEnvironment(side_channels=[config_channel, env_params_channel])
    env.reset()

    behavior_name = list(env.behavior_specs.keys())[0]
    
    # Initialize the controller with safety parameters
    controller = QPColregController(d_safe=3.0, gamma=1.2, v_max=2.5, r_threshold=0.1, enable_colregs=True)
    
    # Initializer for metrics
    metrics = MetricsTracker(d_safe=3.0)

    # 4 GLOBAL BUOYS WITH REAL COORDINATES FROM UNITY

    global_buoys = [
        np.array([0.04, 5.16]),   # Boa 1 (X=0.04, Z=5.16)
        np.array([0.078, -4.99]), # Boa 2 (X=0.078, Z=-4.99)
        np.array([5.07, 0.04]),   # Boa 3 (X=5.07, Z=0.04)
        np.array([-4.95, 0.15])   # Boa 4 (X=-4.95, Z=0.15)
    ]
    
    # Variables to track the global position of our boat (Start X=0.5, Z=0.5)
    boat_global_x = 0.5
    boat_global_y = 0.5
    boat_global_theta = 0.0
    last_steering = 0.0
    dt = 0.1 # average Delta Time 
    
    print(f"Connected to: {behavior_name}")
    print("Avoidance logic active. Vessel (GPS) and global hardcoded buoys monitoring in progress...")
    print("-" * 60)

    try:
        step_count = 0
        while True:
            decision_steps, terminal_steps = env.get_steps(behavior_name)

            if len(decision_steps) > 0:
                # 1. Separate sensors values
                obs_gps = None
                obs_ray = None
                
                for sensor_data in decision_steps.obs:
                    if sensor_data.shape[1] >= 20: 
                        obs_gps = sensor_data[0]  # Boats and target (GPS) Data
                    elif sensor_data.shape[1] == 14:
                        obs_ray = sensor_data[0]  # Buoy data (Laser/Raycast)

                if obs_gps is None: obs_gps = decision_steps.obs[0][0]

                # Nominal velocity
                target_dir = obs_gps[0:2]
                v_nominal = target_dir * 2.5
                v_agent_local = obs_gps[3:5] * 2.5
                
                intruders = []

                # GLOBAL BOAT UPDATE AND BUOY CONVERSION

                # 1. I update the global angle of the boat using the steering
                boat_global_theta += (last_steering * 1.5) * dt
                
                # 2. I rotate the local speed to move on the global map
                vx_loc = v_agent_local[0]
                vy_loc = v_agent_local[1]
                
                vx_glob = vx_loc * math.cos(boat_global_theta) + vy_loc * math.sin(boat_global_theta)
                vy_glob = -vx_loc * math.sin(boat_global_theta) + vy_loc * math.cos(boat_global_theta)
                
                boat_global_x += vx_glob * dt
                boat_global_y += vy_glob * dt

                # 3. I CONVERT GLOBAL BUOYS INTO RELATIVE COORDINATES FOR THE QP
                for g_buoy in global_buoys:
                    dx_glob = g_buoy[0] - boat_global_x
                    dy_glob = g_buoy[1] - boat_global_y
                    
                    loc_x = dx_glob * math.cos(-boat_global_theta) - dy_glob * math.sin(-boat_global_theta)
                    loc_y = dx_glob * math.sin(-boat_global_theta) + dy_glob * math.cos(-boat_global_theta)
                    
                    pos_boa_rel = np.array([loc_x, loc_y])
                    vel_boa_rel = np.array([0.0, 0.0]) # The buoys are stationary
                    
                    intruders.append((pos_boa_rel, vel_boa_rel))

                # 2. MOBILE SHIP LOGIC (from GPS)
                if len(obs_gps) >= 13 and obs_gps[8] < 0.99:
                    pos = obs_gps[6:8] * (obs_gps[8] * 43.0)
                    vel = (obs_gps[9:11] * 5.0) + v_agent_local
                    intruders.append((pos, vel))
                
                if len(obs_gps) >= 20 and obs_gps[15] < 0.99:
                    pos = obs_gps[13:15] * (obs_gps[15] * 43.0)
                    vel = (obs_gps[16:18] * 5.0) + v_agent_local
                    intruders.append((pos, vel))

                # 3. STATIC BUOY LOGIC (from corrected Raycast - mantenuto come supporto)
                if obs_ray is not None:
                    angles_deg = [-90, -45, -15, 0, 15, 45, 90]
                    ray_max_dist = 20.0  
                    
                    for i in range(7):
                        has_hit = obs_ray[i * 2]         
                        hit_fraction = obs_ray[i * 2 + 1] 
                        
                        if has_hit > 0.5:
                            distanza_reale = hit_fraction * ray_max_dist
                            angolo_rad = math.radians(angles_deg[i])
                            
                            pos_boa_x = distanza_reale * math.sin(angolo_rad)
                            pos_boa_y = distanza_reale * math.cos(angolo_rad)
                            
                            pos_boa = np.array([pos_boa_x, pos_boa_y])
                            vel_boa = np.array([0.0, 0.0]) 
                            
                            intruders.append((pos_boa, vel_boa))

                # Recording metric data of the current frame
                metrics.log_step(intruders)

                # 4. QP CONTROL
                u_opt = controller.compute_control(np.array([0,0]), v_nominal, intruders)
                control_actions = map_velocity_to_differential_inputs(u_opt, max_speed=2.5)

                # I update the steering angle for the odometry calculation of the next frame
                last_steering = control_actions[1]

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
        # When the script is stopped, it automatically prints the metrics
        metrics.print_report()
        env.close()

if __name__ == "__main__":
    main()
