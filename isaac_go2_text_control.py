"""
Isaac Go2 with Text Command Control
Run the simulation and control the robot with natural language commands.
"""

import os
import hydra
import torch
import time
import math
import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Go2 with Text Command Control")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

from go2.go2_env import Go2RSLEnvCfg, camera_follow
import env.sim_env as sim_env
import go2.go2_sensors as go2_sensors
from text_command_controller import TextCommandController
import omni
import carb
import go2.go2_ctrl as go2_ctrl

FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")

@hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
def run_simulator(cfg):

    # Go2 Environment setup
    go2_env_cfg = Go2RSLEnvCfg()
    go2_env_cfg.scene.num_envs = cfg.num_envs
    go2_env_cfg.decimation = math.ceil(1./go2_env_cfg.sim.dt/cfg.freq)
    go2_env_cfg.sim.render_interval = go2_env_cfg.decimation
    
    # Initialize control
    go2_ctrl.init_base_vel_cmd(cfg.num_envs)
    
    # Get policy
    env, policy = go2_ctrl.get_rsl_rough_policy(go2_env_cfg)

    # Simulation environment
    if (cfg.env_name == "obstacle-dense"):
        sim_env.create_obstacle_dense_env()
    elif (cfg.env_name == "obstacle-medium"):
        sim_env.create_obstacle_medium_env()
    elif (cfg.env_name == "obstacle-sparse"):
        sim_env.create_obstacle_sparse_env()
    elif (cfg.env_name == "warehouse"):
        sim_env.create_warehouse_env()
    elif (cfg.env_name == "warehouse-forklifts"):
        sim_env.create_warehouse_forklifts_env()
    elif (cfg.env_name == "warehouse-shelves"):
        sim_env.create_warehouse_shelves_env()
    elif (cfg.env_name == "full-warehouse"):
        sim_env.create_full_warehouse_env()

    # Run simulation
    sim_step_dt = float(go2_env_cfg.sim.dt * go2_env_cfg.decimation)
    obs, _ = env.reset()
    
    # Initialize Text Command Controller
    text_controller = TextCommandController(num_envs=cfg.num_envs, env_idx=0)
    
    # Sensor setup (disabled for now to avoid errors)
    # sm = go2_sensors.SensorManager(cfg.num_envs)
    # lidar_annotators = sm.add_rtx_lidar()
    # cameras = sm.add_camera(cfg.freq)

    print("\n" + "="*60)
    print("Simulation started! Robot is ready for commands.")
    print("="*60 + "\n")


    while simulation_app.is_running() and text_controller.running:
        start_time = time.time()
        
        with torch.inference_mode():
            # Get velocity commands from text controller
            vel_commands = text_controller.update(sim_step_dt)
            
            # Update global base_vel_cmd_input for the policy
            go2_ctrl.base_vel_cmd_input = vel_commands.clone()
            
            # Get robot actions from policy
            actions = policy(obs)

            # Step the environment
            obs, _, _, _ = env.step(actions)

            # Camera follow
            if (cfg.camera_follow):
                camera_follow(env)

            # Limit loop time
            elapsed_time = time.time() - start_time
            if elapsed_time < sim_step_dt:
                sleep_duration = sim_step_dt - elapsed_time
                time.sleep(sleep_duration)
        
    
    
    # Cleanup
    text_controller.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    run_simulator()

