import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState, Imu
from scipy.spatial.transform import Rotation as R
import threading
import math
import numpy as np
import argparse
from collections import deque
import time
from humanoid.envs import YiHumanoidCfg
import torch
from pynput import keyboard

class cmd:
    vx = 0.2
    vy = 0.0
    dyaw = 0.0
    vmax = 0.3


# 创建一个线程监听键盘
def on_press(key):
    try:
        if key.char == 'w':
            cmd.vx += 0.01
            cmd.vx = min(cmd.vmax,cmd.vx)
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
        elif key.char == 's':
            cmd.vx -= 0.01
            cmd.vx = max(-cmd.vmax,cmd.vx)
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
        elif key.char == 'a':
            cmd.vy += 0.01
            cmd.vy = min(cmd.vmax,cmd.vy)
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
        elif key.char == 'd':
            cmd.vy -= 0.01
            cmd.vy = max(-cmd.vmax,cmd.vy)
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
        elif key.char == 'j':
            cmd.dyaw += 0.01
            cmd.dyaw = min(cmd.vmax,cmd.dyaw)
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
        elif key.char == 'l':
            cmd.dyaw -= 0.01
            cmd.dyaw = max(-cmd.vmax,cmd.dyaw)
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
        elif key.char == 'r':
            cmd.vx = 0.0
            cmd.vy = 0.0
            cmd.dyaw = 0.0
            print(f"vx:{cmd.vx:.2f} vy:{cmd.vy:.2f} dyaw:{cmd.dyaw:.2f}")
    except AttributeError:
        pass 

def keyboard_input_thread():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def quaternion_to_euler_array(quat):
    r = R.from_quat(quat)
    return r.as_euler('xyz')

class Sim2SimCfg(YiHumanoidCfg):
    def __init__(self, args):
        super().__init__()
        self.load_model = args.load_model

        #控制参数
        self.decimation = 10
        self.dt = 0.001

        #机器人参数
        self.robot_config = {
            'kps': np.array([100.0,100.0,100.0,50.0,50.0,
                             100.0,100.0,100.0,50.0,50.0],dtype=np.double),
            'kds': np.array([0.15,0.15,0.15,0.10,0.70,
                             0.15,0.15,0.15,0.10,0.70],dtype=np.double),
            'tau_limit':200.0*np.ones(10, dtype=np.double)
        }

class EffortController:
    def __init__(self, cfg):
        rospy.init_node('effort_controller',anonymous=True)
        rospy.set_param('/use_sim_time',True)
        self.cfg = cfg
        self.lock = threading.Lock()
        #初始化关节控制器
        self.controllers = [
            "/orca_ydescription/Lleg_yaw_joint_effort_controller/command",
            "/orca_ydescription/Lleg_roll_joint_effort_controller/command",
            "/orca_ydescription/Lleg_thigh_joint_effort_controller/command",
            "/orca_ydescription/Lleg_calf_joint_effort_controller/command",
            "/orca_ydescription/Lleg_ankle_joint_effort_controller/command",
            "/orca_ydescription/Rleg_yaw_joint_effort_controller/command",
            "/orca_ydescription/Rleg_roll_joint_effort_controller/command",
            "/orca_ydescription/Rleg_thigh_joint_effort_controller/command",
            "/orca_ydescription/Rleg_calf_joint_effort_controller/command",
            "/orca_ydescription/Rleg_ankle_joint_effort_controller/command",
        ]
        self.name = [controller.split('/')[2].replace('_effort_controller', '') for controller in self.controllers]
        # print(self.name)
        #初始化状态
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_positions_new = np.zeros((cfg.env.num_actions),dtype=np.double)
        self.joint_velocities_new = np.zeros((cfg.env.num_actions),dtype=np.double)
        self.imu_orientation = None
        self.imu_angular_velocity = None
        self.default_angle = np.zeros((cfg.env.num_actions),dtype=np.double)
        self.lock = threading.Lock()

        #初始化控制器
        self.publishers = {topic: rospy.Publisher(topic, Float64, queue_size=10) for topic in self.controllers}
        rospy.Subscriber("/orca_ydescription/joint_states", JointState, self.joint_state_callback)
        rospy.Subscriber("/imu", Imu, self.imu_callback)

        #加载策略
        self.policy = torch.jit.load(cfg.load_model)
        self.hist_obs = deque(maxlen=cfg.env.frame_stack)
        for _ in range(cfg.env.frame_stack):
            self.hist_obs.append(np.zeros(cfg.env.num_single_obs,dtype=np.float32))

        self.default_angle[0]=cfg.init_state.default_joint_angles['Lleg_yaw_joint']
        self.default_angle[1]=cfg.init_state.default_joint_angles['Lleg_roll_joint']
        self.default_angle[2]=cfg.init_state.default_joint_angles['Lleg_thigh_joint']
        self.default_angle[3]=cfg.init_state.default_joint_angles['Lleg_calf_joint']
        self.default_angle[4]=cfg.init_state.default_joint_angles['Lleg_ankle_joint']
        self.default_angle[5]=cfg.init_state.default_joint_angles['Rleg_yaw_joint']
        self.default_angle[6]=cfg.init_state.default_joint_angles['Rleg_roll_joint']
        self.default_angle[7]=cfg.init_state.default_joint_angles['Rleg_thigh_joint']
        self.default_angle[8]=cfg.init_state.default_joint_angles['Rleg_calf_joint']
        self.default_angle[9]=cfg.init_state.default_joint_angles['Rleg_ankle_joint']
        self.count = 0
        #启动键盘输入线程
        keyboard_thread = threading.Thread(target=keyboard_input_thread)
        keyboard_thread.daemon = True
        keyboard_thread.start()

    def joint_state_callback(self, msg):
        with self.lock:
            self.joint_name = np.array(msg.name, dtype=str)
            # print(self.joint_name)
            self.joint_positions = np.array(msg.position, dtype=np.double)
            self.joint_velocities = np.array(msg.velocity, dtype=np.double)
            self.joint_data = {joint:{'position': pos, 'velocity': vel}
            for joint, pos, vel in zip(self.joint_name, self.joint_positions, self.joint_velocities)}
            # print(self.joint_data)

    def imu_callback(self, msg):
        with self.lock:
            self.imu_orientation = [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ]
            # print(msg.orientation.x)
            self.imu_angular_velocity = [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ]

    def get_obs(self):
        with self.lock:
            if self.joint_positions is None or self.imu_orientation is None:
                return None
            
            r = R.from_quat(self.imu_orientation)
            euler = r.as_euler('xyz')
            omega = np.array(self.imu_angular_velocity, dtype=np.float32)
            for i, joint_name in enumerate(self.name):
                self.joint_positions_new[i]=self.joint_data[joint_name]['position']
                self.joint_velocities_new[i]=self.joint_data[joint_name]['velocity']
            return{
                'q': self.joint_positions_new,
                'dq': self.joint_velocities_new,
                'euler': euler,
                'omega': omega
            }
        
    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        return(target_q - q) * kp + (target_dq - dq) * kd
    
    def control_loop(self):
        rate = rospy.Rate(1000)
        while not rospy.is_shutdown():
            # print("Whiling")
            #使用ROS时间（仿真时间）记录起始时间
            #start_time = ropy.get_rostime()
            #if self.count % 100 == 0:
            #   rospy.loginfo(f"Running control loop: {self.count}")

            obs_data = self.get_obs()
            if obs_data is None:
                rate.sleep()
                continue

            #构造观测向量
            if self.count % self.cfg.decimation == 0:
                obs = np.zeros(self.cfg.env.num_single_obs, dtype=np.float32)

                #基础运动命令
                obs[0] = math.sin(2 * math.pi * self.count * self.cfg.dt / 0.64)
                obs[1] = math.cos(2 * math.pi * self.count * self.cfg.dt / 0.64)
                obs[2] = cmd.vx * self.cfg.normalization.obs_scales.lin_vel
                obs[3] = cmd.vy * self.cfg.normalization.obs_scales.lin_vel
                obs[4] = cmd.dyaw * self.cfg.normalization.obs_scales.ang_vel
                #关节状态
                obs[5:15] = (obs_data['q'] - self.default_angle)* self.cfg.normalization.obs_scales.dof_pos
                obs[15:25] = obs_data['dq'] * self.cfg.normalization.obs_scales.dof_vel
                obs[25:35] = self.last_action if hasattr(self, 'last_action') else np.zeros(10)

                #IMU数据
                obs[35:38] = obs_data['omega']
                obs[38:41] = obs_data['euler']

                #更新历史观测
                self.hist_obs.append(obs)
                policy_input = np.concatenate(self.hist_obs, axis=0).reshape(1, -1)

                #策略推理
                with torch.no_grad():
                    action = self.policy(torch.from_numpy(policy_input)).numpy()[0]

                action = np.clip(action, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
                self.last_action =action
                if self.count>400:
                    self.target_q = action * self.cfg.control.action_scale + self.default_angle
                else:
                    self.target_q = self.default_angle
            
            tau = self.pd_control(
                self.target_q, obs_data['q'], self.cfg.robot_config['kps'],
                0, obs_data['dq'], self.cfg.robot_config['kds']
            )
            tau = np.clip(tau, -self.cfg.robot_config['tau_limit'],self.cfg.robot_config['tau_limit'])
            print(tau)

            #发布控制命令
            for i, topic in enumerate(self.controllers):
                self.publishers[topic].publish(Float64(tau[i]))

            self.count += 1
            rate.sleep()
            # # 计算 ROS（仿真）时间下的循环运行时间
            # loop_duration = (rospy.get_rostime() - start_time).to_sec()
            # rospy.loginfo(f"Loop time: {loop_duration:.4f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str,
                        default='/home/yi/humanoid-gym/logs/YiHumanoid_ppo/exported/policies/policy_1.pt',
                        help='Path to trained policy model')
    args = parser.parse_args()

    cfg = Sim2SimCfg(args)
    controller = EffortController(cfg)

    #启动控制线程
    control_thread = threading.Thread(target=controller.control_loop)
    control_thread.start()
    print("Control thread started")
    #主线程处理callback
    rospy.spin()





