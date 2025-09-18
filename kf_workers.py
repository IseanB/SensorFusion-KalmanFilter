import os
import csv
import numpy as np
import utm
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import operator
import concurrent.futures
import multiprocessing
import sys
from scipy.interpolate import interp1d
import math
import decimal
# from kf_workers import evaluate_combo_chunk_worker


def evaluate_combo_chunk_worker(chunk, xt, Pt, class_args):
    results = []
    for combo in chunk:
        xt_bf = xt.copy()
        Pt_bf = Pt.copy()
        traj = []
        prev_time = None
        for (idx, stype, t, sdata) in combo:
            dt = t - prev_time if prev_time is not None else 0
            F = class_args['get_state_transition_matrix'](dt)
            Qt = class_args['get_process_noise_covariance_matrix'](dt)
            xt_bf = np.dot(F, xt_bf)
            Pt_bf = class_args['predict_covariance'](Pt_bf, F, Qt)
            if stype == 'GPS':
                H = class_args['get_gps_observation_matrix']()
                R = class_args['get_gps_measurement_noise_covariance_matrix']()
                Z = [sdata['easting'], sdata['northing'], sdata['altitude']]
            else:
                ax, ay, az = sdata[7], sdata[8], sdata[9]
                Vx = xt_bf[6] + ax * dt
                Vy = xt_bf[7] + ay * dt
                Vz = xt_bf[8] + az * dt
                X = xt_bf[0] + Vx * dt
                Y = xt_bf[1] + Vy * dt
                Z = xt_bf[2] + Vz * dt
                roll, pitch, yaw = sdata[1], sdata[2], sdata[3]
                ang_x, ang_y, ang_z = sdata[4], sdata[5], sdata[6]
                Z = [X, Y, Z, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]
                H = class_args['get_imu_observation_matrix']()
                R = class_args['get_imu_measurement_noise_covariance_matrix']()
            K = class_args['calculate_kalman_gain'](Pt_bf, H, R)
            y = np.array(Z) - np.dot(H, xt_bf)
            xt_bf = xt_bf + np.dot(K, y)
            Pt_bf = np.dot(np.eye(15) - np.dot(K, H), Pt_bf)
            traj.append((t, *xt_bf[:6]))
            prev_time = t
        metric = class_args['calculate_accuracy_metrics'](traj)
        results.append((metric, traj, combo, xt_bf, Pt_bf))
    return results

class Scheduler:
    """
    A class to represent a sensor scheduler.

    schedule: A function to schedule the sensors.
        RAND : Randomly select sensors.
        GREEDY : Greedily select sensors.
        RANDOMIZED_GREEDY : Randomized greedy selection of sensors.

    """

    # Main Computations

    def cov_matrix(_, S, Sigma_prev, R, H, device = 'cuda'):
        assert type(S) == list, "S should be a list"
        assert len(S) > 0, "S should not be empty"
        assert type(Sigma_prev) == np.ndarray, "Sigma_prev should be a np.ndarray array"
        assert type(R) == np.ndarray, "R should be a np.ndarray array"
        assert type(H) == np.ndarray, "H should be a np.ndarray array"
        
        result = None   
        k = len(S)
        def build_nonzeroH(H, S):
            S_sorted = sorted(S)
            H_hat = H[np.array(S_sorted) - 1, :]  # Convert to 0-based indexing
            return H_hat

        def build_nonzeroR(R, S):
            S_sorted = sorted(S)
            R_hat = R[np.ix_(np.array(S_sorted) - 1, np.array(S_sorted) - 1)]  # Convert to 0-based indexing
            return R_hat
        
        if device == 'cpu':
            
            if k == R.shape[0]:
                R_hat = R
                H_hat = H
            else:
                R_hat = build_nonzeroR(R, S)
                H_hat = build_nonzeroH(H, S)

            # one way to calculate P
            S = R_hat + np.dot(H_hat, np.dot(Sigma_prev, H_hat.T))
            K = np.dot(np.dot(Sigma_prev, H_hat.T), np.linalg.inv(S))
            Sigma_post = Sigma_prev - np.dot(np.dot(K,H_hat),Sigma_prev)
            
            # print("Sigma_post!")
            #self.x += np.dot(K, innovation)
            result = Sigma_post
        elif device == 'cuda':
            import torch

            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

            if k == R.shape[0]:
                R_hat = R
                H_hat = H
            else:
                R_hat = build_nonzeroR(R, S)
                H_hat = build_nonzeroH(H, S)

            # one way to calculate P
            R_hat = torch.tensor(R_hat, device=device)
            H_hat = torch.tensor(H_hat, device=device)
            Sigma_prev = torch.tensor(Sigma_prev, device=device)
            
            S = R_hat + torch.matmul(H_hat, torch.matmul(Sigma_prev, H_hat.T))
            K = torch.matmul(torch.matmul(Sigma_prev, H_hat.T), torch.linalg.inv(S))
            Sigma_post = Sigma_prev - torch.matmul(torch.matmul(K, H_hat), Sigma_prev)
            result = Sigma_post.cpu().numpy()   

        return result


    def gain(self, measurement=None, S_sigma=None, measurement_cov=(tuple), observation_cov=(tuple), device = 'cuda'):
        # Sample Data [(54717, 'GPS', 1697739552.3362827, {'time': 1697739552.3362827, 'easting': np.float64(0.0), 'northing': np.float64(0.0), 'zone_number': 19, 'zone_letter': 'T', 'altitude': -32.6}),
        #            (54718, 'IMU', 1697739552.3381784, ['1697739552.3381784', np.float64(-0.022514711904387082), np.float64(-0.03908667660761563), np.float64(-0.011081221176772028), -0.007647866794783674, -0.00015793529328919122, -0.003541810559943126, 0.02323812944797532, 0.13031730752538762, -0.06398634169611661])]
        assert type(measurement) != list, "only have one measurement"
        assert S_sigma is not None, "S_sigma should not be None"
        if measurement is None or len(measurement) == 0:
            return 0

        if measurement[1] == 'GPS':
            return np.trace(self.cov_matrix(S = [1], Sigma_prev=S_sigma, R=measurement_cov["GPS"], H=observation_cov["GPS"], device=device))
        elif measurement[1] == 'IMU':
            return np.trace(self.cov_matrix(S = [1], Sigma_prev=S_sigma, R=measurement_cov["IMU"], H=observation_cov["IMU"], device=device))


    def random_schedule(self, measurements=None, current_active_sensors=None, device = 'cuda'):
        assert measurements is not None, "measurements should not be None"
        assert type(current_active_sensors) == int, "current_active_sensors should be an integer"
        selected_indices = np.random.choice(len(measurements), current_active_sensors, replace=False)
        # selected_indices = sorted(selected_indices) # Needed?

        return selected_indices

    def greedy_schedule(self, measurements=None, S_sigma=None, measurement_cov=None, observation_cov=None, device = 'cuda'):
        assert measurements is not None or len(measurement) != 0, "measurements should not be None"
        S_greedy = []
        
        best_gain = -np.inf
        best_measurement = None
        
        for measurement in measurements: # arg max
            curr_gain = self.gain(measurement, S_sigma, measurement_cov, observation_cov, device=device) # gain function= gain(measurement) - gain([]) aka gain(measurement) - 0
            # print(f"Current gain: {curr_gain}, Best gain: {best_gain}")
            if curr_gain > best_gain:
                best_gain = curr_gain
                best_measurement = measurement
                    

        if best_measurement != None:
            S_greedy.append(best_measurement)

        return measurements.index(best_measurement) # index of the best measurement
    
    def randomized_greedy_schedule(self, total_num_sensors, device = 'cuda'):
        return None

    def schedule(self, func, device = 'cuda'): # WIP
        S = None
        if func == 'RAND':
            S = self.random_schedule(device)
            assert S is not None, "S should not be None after RAND"
        elif func == 'GREEDY':
            S = self.greedy_schedule(device)
            assert S is not None, "S should not be None after GREEDY"
        elif func == 'RANDOMIZED_GREEDY':
            S = self.randomized_greedy_schedule(device)
            assert S is not None, "S should not be None after RANDOMIZED_GREEDY"
        else:
            raise ValueError("func should be a string of either 'RAND', 'GREEDY', or 'RANDOMIZED_GREEDY'")

        Sigma_post = self.cov_matrix(None, S, self.prev_measurements, self.measure_cov, self.observation_cov, device)
        return Sigma_post, S
    
def plot_accuracy_error(accuracy_metrics, save_path='kf_error_plot.png'):
    """
    Plots the position error over time.

    Args:
        accuracy_metrics (dict): The dictionary returned by calculate_accuracy_metrics.
        save_path (str): Path to save the plot image.
    """
    if not accuracy_metrics:
        print("No accuracy metrics to plot.")
        return

    position_errors = accuracy_metrics['position_errors']
    num_points = len(position_errors)

    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Kalman Filter Position Error vs. GPS Ground Truth', fontsize=16)

    # Plot error for each axis
    axs[0].plot(range(num_points), position_errors[:, 0], label='Error X (East)', alpha=0.8)
    axs[0].plot(range(num_points), position_errors[:, 1], label='Error Y (North)', alpha=0.8)
    axs[0].plot(range(num_points), position_errors[:, 2], label='Error Z (Altitude)', alpha=0.8)
    axs[0].set_ylabel('Error (meters)')
    axs[0].set_title('Per-Axis Position Error')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_yscale('log')

    # Plot total Euclidean distance error
    axs[1].plot(range(num_points), accuracy_metrics['euclidean_errors'], label='Total Position Error', color='r')
    axs[1].set_xlabel('GPS Measurement Index')
    axs[1].set_ylabel('Euclidean Distance Error (meters)')
    axs[1].set_title('Total Position (Euclidean) Error')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_yscale('log')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path)
    print(f"Error plot saved to {save_path}")
    plt.show()

class KF_SensorFusion:
    def __init__(self, gps_csv_file, imu_csv_file, total_num_sensors=15, max_active_num_sensors=1):
        self.gps_csv_file = gps_csv_file
        self.imu_csv_file = imu_csv_file
        self.gps_data = []
        self.imu_data = []
        self.utm_data = []
        self.scheduler = Scheduler()
        self.processing_frequency = 120  # HZ, Degraded estimation performance around 12 HZ. 1/12

    def set_processing_frequency(self, frequency):
        self.processing_frequency = frequency

    def load_data_from_csv(self, filename, has_header=True):
        data = []
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile)
            if has_header:
                next(reader)  # Skip header row
            for row in reader:
                data.append(row)
        return data

    def load_data(self):
        self.gps_data = self.load_data_from_csv(self.gps_csv_file)
        self.imu_data = self.load_data_from_csv(self.imu_csv_file)

    def gps_to_modified_utm(self):
        '''Converts GPS data to UTM coordinates and includes altitude data.'''
        initial_easting = None
        initial_northing = None

        for entry in self.gps_data:
            time, lat_str, lon_str, alt_str = entry
            if 'nan' in lat_str.lower() or 'nan' in lon_str.lower() or 'nan' in alt_str.lower():
                continue

            lat = float(lat_str)
            lon = float(lon_str)
            alt = float(alt_str)

            # Convert latitude and longitude to UTM coordinates
            easting, northing, zone_number, zone_letter = utm.from_latlon(lat, lon)

            # If this is the first valid entry, store its easting and northing as the initial offset
            if initial_easting is None or initial_northing is None:
                initial_easting = easting
                initial_northing = northing

            # Subtract the initial offset to make the first point (0,0)
            adjusted_easting = easting - initial_easting
            adjusted_northing = northing - initial_northing

            # Format: ([UTM Format], altitude)
            self.utm_data.append({'time': float(time), 'easting': adjusted_easting, 'northing': adjusted_northing, 'zone_number': zone_number, 'zone_letter': zone_letter, 'altitude': alt })

    @staticmethod ##need to update this function based on time for valid index 
    def compute_imu_biases(gps_data, imu_data):
        first_valid_index = next((i for i, entry in enumerate(gps_data) if 'nan' not in entry[1].lower()), None)
        if first_valid_index is None:
            print("Warning: No valid GPS data found. Cannot compute IMU biases.")
            return None, None
        print(f"First valid GPS entry index: {first_valid_index}")
        stationary_imu_data = imu_data[:first_valid_index]
        angular_velocity_data = [np.array([float(entry[5]), float(entry[6]), float(entry[7])]) for entry in stationary_imu_data]
        linear_acceleration_data = [np.array([float(entry[8]), float(entry[9]), float(entry[10])]) for entry in stationary_imu_data]
        angular_velocity_bias = np.mean(angular_velocity_data, axis=0)
        linear_acceleration_bias = np.mean(linear_acceleration_data, axis=0)
        print(f"Computed Angular Velocity Bias: {angular_velocity_bias}")
        print(f"Computed Linear Acceleration Bias: {linear_acceleration_bias}")
        return angular_velocity_bias, linear_acceleration_bias, first_valid_index

    def unbias_imu_data(self, angular_velocity_bias, linear_acceleration_bias):
        unbias_imu_data = []
        for entry in self.imu_data:
            # Extract angular velocity and linear acceleration from IMU data
            angular_velocity = np.array([float(entry[5]), float(entry[6]), float(entry[7])])
            linear_acceleration = np.array([float(entry[8]), float(entry[9]), float(entry[10])])

            # Subtract biases from angular velocity and linear acceleration
            unbias_angular_velocity = angular_velocity - angular_velocity_bias
            unbias_linear_acceleration = linear_acceleration - linear_acceleration_bias

            # Extract quaternion components from IMU data
            x, y, z, w = float(entry[1]), float(entry[2]), float(entry[3]), float(entry[4])
            
            # Convert quaternion to Euler angles
            roll, pitch, yaw = self.quaternion_to_euler(x, y, z, w)

            # Create an unbias entry with Euler angles
            unbias_entry = entry[:1] + [roll, pitch, yaw] + unbias_angular_velocity.tolist() + unbias_linear_acceleration.tolist() + entry[11:]
            unbias_imu_data.append(unbias_entry)
        # Print the first two entries to check if the data is stored well
        print("First two entries in unbias_imu_data:")
        for i in range(2):
            print(unbias_imu_data[i])
        self.unbias_imu_data = unbias_imu_data

    def combine_sensor_data(self):
        """Combine GPS and IMU data and sort by timestamp, adding an index to each entry."""
        combined_data = []
        for gps_entry in self.utm_data:
            time = float(gps_entry['time'])
            combined_data.append(('GPS', time, gps_entry))
        for imu_entry in self.unbias_imu_data:
            time = float(imu_entry[0])  
            combined_data.append(('IMU', time, imu_entry))
        combined_data.sort(key=lambda x: x[1])
        self.indexed_sensor_data = [(i, *data) for i, data in enumerate(combined_data)]
        # Print the first GPS and IMU entry
        gps_printed = imu_printed = False
        for entry in self.indexed_sensor_data:
            if entry[1] == 'GPS' and not gps_printed:
                print("First GPS Entry:", entry)
                gps_printed = True
            elif entry[1] == 'IMU' and not imu_printed:
                print("First IMU Entry:", entry)
                imu_printed = True
            if gps_printed and imu_printed:
                break
            
    #take this function from internet as its standard function
    def quaternion_to_euler(self, x, y, z, w):
        """
        Convert a quaternion to Euler angles (roll, pitch, yaw).

        Parameters:
        x, y, z, w (float): Quaternion components.

        Returns:
        tuple: A tuple containing roll, pitch, yaw angles in radians.
        """
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.pi / 2 * np.sign(sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def compute_stationary_orientation(self, first_valid_index):
        """
        Compute the average orientation of the IMU when the car is stationary
        and plot the IMU frame.
        """
        stationary_imu_data = self.unbias_imu_data[:first_valid_index]

        # Calculate average roll, pitch, yaw
        avg_roll = np.mean([entry[1] for entry in stationary_imu_data])
        avg_pitch = np.mean([entry[2] for entry in stationary_imu_data])
        avg_yaw = np.mean([entry[3] for entry in stationary_imu_data])

        return avg_roll, avg_pitch, avg_yaw

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        R = Rz @ Ry @ Rx
        return R

    def plot_imu_frame(self, roll, pitch, yaw):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        axis_limit = 1.0
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        R = self.euler_to_rotation_matrix(roll, pitch, yaw)
        # Original IMU frame axes (before rotation)
        imu_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotated_axes = R @ imu_axes
        ax.quiver(0, 0, 0, rotated_axes[0, 0], rotated_axes[0, 1], rotated_axes[0, 2], length=0.1, color='red')
        ax.quiver(0, 0, 0, rotated_axes[1, 0], rotated_axes[1, 1], rotated_axes[1, 2], length=0.1, color='green')
        ax.quiver(0, 0, 0, rotated_axes[2, 0], rotated_axes[2, 1], rotated_axes[2, 2], length=0.1, color='blue')
        plt.title(f"IMU Orientation - Roll: {np.degrees(roll):.2f}, Pitch: {np.degrees(pitch):.2f}, Yaw: {np.degrees(yaw):.2f}")
        plt.show()
        
    def plot_gps_data(self):
        eastings = [data['easting'] for data in self.utm_data]
        northings = [data['northing'] for data in self.utm_data]
        plt.figure(figsize=(10, 6))
        plt.scatter(eastings, northings, c='blue', marker='o', label='raw GPS Points', s=3)  
        plt.title('GPS Data(raw) Scatter Plot')
        plt.xlabel('Easting')
        plt.ylabel('Northing')
        plt.legend()
        plt.grid(True)
        plt.show()

    #kalman filter functions for fusing IMU and GPS data
    def get_state_transition_matrix(self, dt):
        """Return the state transition matrix F based on the time difference dt."""
        # now: x, y, z, 
        #               roll, pitch, yaw, 
        #                           v_x, v_y, v_z, 
        #                                       angular_x, angular_y, angular_z,
        #                                                   a_x, a_y, a_z
        F = np.array([
            [1, 0, 0,   0, 0, 0,    dt, 0, 0,   0, 0, 0,    0.5 * dt**2, 0, 0],     # x = x + v_x*dt + 0.5*a_x*dt^2
            [0, 1, 0,   0, 0, 0,    0, dt, 0,   0, 0, 0,    0,  0.5 * dt**2, 0],    # y = y + v_y*dt + 0.5*a_y*dt^2
            [0, 0, 1,   0, 0, 0,    0, 0, dt,   0, 0, 0,    0,  0, 0.5 * dt**2],    # z = z + v_z*dt + 0.5*a_z*dt^2
            [0, 0, 0,   1, 0, 0,    0, 0, 0,    dt, 0, 0,   0, 0, 0],               # roll = roll + angular_x*dt
            [0, 0, 0,   0, 1, 0,    0, 0, 0,    0, dt, 0,   0, 0, 0],               # pitch = pitch + angular_y*dt
            [0, 0, 0,   0, 0, 1,    0, 0, 0,    0, 0, dt,   0, 0, 0],               # yaw = yaw + angular_z*dt
            [0, 0, 0,   0, 0, 0,    1, 0, 0,    0, 0, 0,    dt, 0, 0],              # v_x = v_x + a_x*dt
            [0, 0, 0,   0, 0, 0,    0, 1, 0,    0, 0, 0,    0, dt, 0],              # v_y = v_y + a_y*dt
            [0, 0, 0,   0, 0, 0,    0, 0, 1,    0, 0, 0,    0, 0, dt],              # v_z = v_z + a_z*dt
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    1, 0, 0,    0, 0, 0],              # angular_x = angular_x (constant angular velocity assumed)
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 1, 0,    0, 0, 0],              # angular_y = angular_y (constant angular velocity assumed)
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 1,    0, 0, 0],              # angular_z = angular_z (constant angular velocity assumed)
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    1, 0, 0],              # a_x = a_x (constant acceleration assumed)
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 1, 0],              # a_y = a_y (constant acceleration assumed)
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 1],              # a_z = a_z (constant acceleration assumed)
        ])
        return F

    def get_process_noise_covariance_matrix(self, dt):
        """Define the process noise covariance matrix Qt based on dt."""
        position_noise = 5 * dt
        orientation_noise = 0.05 * dt
        velocity_noise = 1 * dt
        angular_velocity_noise = 0.1 * dt
        acceleration_noise = 2 * dt

        Qt = np.diag([
            position_noise,        # x 
            position_noise,        # y 
            position_noise,        # z 
            orientation_noise,     # roll 
            orientation_noise,     # pitch 
            orientation_noise,     # yaw 
            velocity_noise,        # v_x 
            velocity_noise,        # v_y 
            velocity_noise,        # v_z 
            angular_velocity_noise,# angular_x
            angular_velocity_noise,# angular_y
            angular_velocity_noise,# angular_z
            acceleration_noise,    # a_x 
            acceleration_noise,    # a_y 
            acceleration_noise     # a_z 
        ])
        return Qt

    def predict_covariance(self, Pt, F, Qt):
        """Predict the next covariance matrix P(t+1)."""
        P_next = np.dot(np.dot(F, Pt), F.T) + Qt
        return P_next
    
    def get_gps_observation_matrix(self):
        """Return the GPS observation matrix H_GPS."""
        H_GPS = np.array([
            [1, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],   # X measurement affects the X position state
            [0, 1, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],   # Y measurement affects the Y position state
            [0, 0, 1,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0]    # Z measurement affects the Z position state
        ])
        return H_GPS
    
    def get_imu_observation_matrix(self):
        """Return the IMU observation matrix H_IMU."""
        H_IMU = np.array([
            [1, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures X
            [0, 1, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Y
            [0, 0, 1,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Z
            [0, 0, 0,   1, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Roll
            [0, 0, 0,   0, 1, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Pitch
            [0, 0, 0,   0, 0, 1,    0, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Yaw
            [0, 0, 0,   0, 0, 0,    1, 0, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Vx
            [0, 0, 0,   0, 0, 0,    0, 1, 0,    0, 0, 0,    0, 0, 0],  # IMU measures Vy
            [0, 0, 0,   0, 0, 0,    0, 0, 1,    0, 0, 0,    0, 0, 0],  # IMU measures Vz
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    1, 0, 0,    0, 0, 0],  # IMU measures angular_x
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 1, 0,    0, 0, 0],  # IMU measures angular_y
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 1,    0, 0, 0],  # IMU measures angular_z
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    1, 0, 0],  # IMU measures acceleration_x
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 1, 0],  # IMU measures acceleration_y
            [0, 0, 0,   0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 1],  # IMU measures acceleration_z
        ])
        return H_IMU

    def get_gps_measurement_noise_covariance_matrix(self):
        """Return the GPS measurement noise covariance matrix R_GPS."""
        gps_position_noise_variance = 3
        R_GPS = np.diag([gps_position_noise_variance, gps_position_noise_variance, gps_position_noise_variance])
        return R_GPS

    def get_imu_measurement_noise_covariance_matrix(self):
        """Return the IMU measurement noise covariance matrix R_IMU."""
        position_noise_variance = 50
        velocity_noise_variance = 10 
        roll_noise_variance = 0.05       
        pitch_noise_variance = 0.05       
        yaw_noise_variance = 0.05       
        angular_velocity_noise_variance = 0.1
        linear_acceleration_noise_variance = 100

        R_IMU = np.diag([
            position_noise_variance,     # X 
            position_noise_variance,     # Y 
            position_noise_variance,     # Z 
            roll_noise_variance,         # roll 
            pitch_noise_variance,        # pitch 
            yaw_noise_variance,          # yaw
            velocity_noise_variance,     # Vx 
            velocity_noise_variance,     # Vy 
            velocity_noise_variance,     # Vz 
            angular_velocity_noise_variance, # angular_x
            angular_velocity_noise_variance, # angular_y
            angular_velocity_noise_variance, # angular_z 
            linear_acceleration_noise_variance, # a_x 
            linear_acceleration_noise_variance,  # a_y 
            linear_acceleration_noise_variance  # a_z 
        ])
        return R_IMU

    def calculate_kalman_gain(self, P_next, H, R):
        # Intermediate matrix calculation
        intermediate_matrix = np.dot(np.dot(H, P_next), H.T) + R
        # Calculating the Kalman Gain
        K_next = np.dot(np.dot(P_next, H.T), np.linalg.inv(intermediate_matrix))
        return K_next

    def run_kalman_filter(self, start_idx, end_idx):
        """Run the Kalman Filter to fuse IMU and GPS data for pose estimation."""
        # Initial state [x, y, z, roll, pitch, yaw, v_x, v_y, v_z, angular_x, angular_y, angular_z, a_x, a_y, a_z]
        xt = np.array([0, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0])

        # Initial covariance matrix
        Pt = np.array([
                        [1000, 0, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],       # x
                        [0, 1000, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],       # y
                        [0, 0, 1000,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],       # z
                        [0, 0, 0,        100, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],       # roll
                        [0, 0, 0,        0, 100, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],       # pitch
                        [0, 0, 0,        0, 0, 100,     0, 0, 0,    0, 0, 0,    0, 0, 0],       # yaw
                        [0, 0, 0,        0, 0, 0,     100, 0, 0,    0, 0, 0,    0, 0, 0],       # v_x
                        [0, 0, 0,        0, 0, 0,     0, 100, 0,    0, 0, 0,    0, 0, 0],       # v_y
                        [0, 0, 0,        0, 0, 0,     0, 0, 100,    0, 0, 0,    0, 0, 0],       # v_z
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    100, 0, 0,    0, 0, 0],       # angular_x
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 100, 0,    0, 0, 0],       # angular_y
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 100,    0, 0, 0],       # angular_z
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      1000, 0, 0],    # a_x
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 1000, 0],    # a_y
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 0, 1000],    # a_z
                    ])
        I = np.eye(15)

        # Create a list to store the state (X, Y, Z, roll, pitch, yaw)
        sf_KF_state = [(0, xt[0], xt[1], xt[2], xt[3], xt[4], xt[5])]  # Storing initial state
        gps_started = False  
        for i, (index, sensor_type, time, sensor_data) in tqdm(enumerate(self.indexed_sensor_data[start_idx:end_idx])):
            # # Start processing only when the first GPS data is encountered
            if sensor_type == 'GPS' and not gps_started:
                gps_started = True
                previous_time = time  # Initialize previous_time with the first GPS time
            if not gps_started:
                continue  # Skip until first GPS data is encountered
            # Calculate dt
            dt = time - previous_time if previous_time is not None else 0

            # Prediction step for both GPS and IMU
            F = self.get_state_transition_matrix(dt)
            Qt = self.get_process_noise_covariance_matrix(dt)
            xt = np.dot(F, xt)   # State prediction
            Pt = self.predict_covariance(Pt, F, Qt)  # Covariance prediction

            # Update step based on sensor type
            if sensor_type == 'GPS':
                H_GPS = self.get_gps_observation_matrix()
                R_GPS = self.get_gps_measurement_noise_covariance_matrix()
                Zt_GPS = [sensor_data['easting'], sensor_data['northing'], sensor_data['altitude']]  # Extracting GPS measurements
                Kt = self.calculate_kalman_gain(Pt, H_GPS, R_GPS)
                y = Zt_GPS - np.dot(H_GPS, xt)
                # State update
                xt = xt + np.dot(Kt,y)
                # Covariance update
                Pt = np.dot(I - np.dot(Kt, H_GPS), Pt)
                
            elif sensor_type == 'IMU':
                # Dead reckoning for IMU data (integration of velocity and position)
                ax = sensor_data[7]
                ay = sensor_data[8]
                az = sensor_data[9]
                Vx = xt[6] + ax * dt  # Vx = Vx + ax*dt
                Vy = xt[7] + ay * dt  # Vy = Vy + ay*dt
                Vz = xt[8] + az * dt  # Vz = Vz + az*dt
                X = xt[0] + Vx * dt  # X = X + Vx*dt
                Y = xt[1] + Vy * dt  # Y = Y + Vy*dt
                Z = xt[2] + Vz * dt  # Y = Y + Vy*dt
                roll = sensor_data[1]
                pitch = sensor_data[2]
                yaw = sensor_data[3]
                ang_x = sensor_data[4]
                ang_y = sensor_data[5]
                ang_z = sensor_data[6]

                Zt_IMU = [X, Y, Z, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]  # IMU measurements
                H_IMU = self.get_imu_observation_matrix()
                R_IMU = self.get_imu_measurement_noise_covariance_matrix()
                Kt = self.calculate_kalman_gain(Pt, H_IMU, R_IMU)
                y = np.array(Zt_IMU) - np.dot(H_IMU, xt)
                xt = xt + np.dot(Kt, y)
                Pt = np.dot(I - np.dot(Kt, H_IMU), Pt)

            sf_KF_state.append((time, xt[0], xt[1], xt[2], xt[3], xt[4], xt[5]))  # Append updated state (X, Y, Z, Roll, Pitch, Yaw)
            previous_time = time  # Update previous time
        return sf_KF_state

    def run_kalman_filter_scheduled(self, start_idx, end_idx):
        """Run the Kalman Filter to fuse IMU and GPS data for pose estimation."""
        # Initialize state and covariance
        xt = np.zeros(15)
        Pt = np.array([
                        [1000, 0, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 1000, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 1000,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        100, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 100, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 100,     0, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     100, 0, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 100, 0,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 100,    0, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    100, 0, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 100, 0,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 100,    0, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      1000, 0, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 1000, 0],
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 0, 1000],
                    ])
        I = np.eye(15)

        # Find the initial GPS measurement in the subset to initialize xt and previous_time.
        gps_started = False
        previous_time = None
        for (index, sensor_type, time, sensor_data) in self.indexed_sensor_data[start_idx:end_idx]:
            if sensor_type == 'GPS':
                xt[0] = sensor_data['easting']
                xt[1] = sensor_data['northing']
                xt[2] = sensor_data['altitude']
                previous_time = time
                gps_started = True
                break
        if not gps_started:
            print("WARN: No GPS measurement found in the sensor subset.")
            return None

        # Store initial state (using the GPS time)
        sf_KF_state = [(previous_time, xt[0], xt[1], xt[2], xt[3], xt[4], xt[5])]
        
        sensor_data_queue = []
        for i, (index, sensor_type, time, sensor_data) in tqdm(enumerate(self.indexed_sensor_data[start_idx:end_idx])):
            # Only process measurements occurring after initialization.
            if time < previous_time:
                continue

            # Collect sensor data over the sampling period.
            if time - previous_time < 1/self.processing_frequency:
                sensor_data_queue.append((index, sensor_type, time, sensor_data))
                continue

            # Ensure there is at least one measurement for update.
            if len(sensor_data_queue) == 0:
                sensor_data_queue.append((index, sensor_type, time, sensor_data))
            
            # Choose the best measurement using the scheduler.
            selected_measurement_index = self.scheduler.greedy_schedule(
                sensor_data_queue,
                S_sigma=Pt,
                measurement_cov={
                    "IMU": self.get_imu_measurement_noise_covariance_matrix(),
                    "GPS": self.get_gps_measurement_noise_covariance_matrix()
                },
                observation_cov={
                    "IMU": self.get_imu_observation_matrix(),
                    "GPS": self.get_gps_observation_matrix()
                },
                device='cpu'
            )
            sensor_data_queue = []
            (index, sensor_type, time, sensor_data) = self.indexed_sensor_data[selected_measurement_index]
            
            # Calculate dt
            dt = time - previous_time if previous_time is not None else 0

            # Prediction step for both GPS and IMU.
            F = self.get_state_transition_matrix(dt)
            Qt = self.get_process_noise_covariance_matrix(dt)
            xt = np.dot(F, xt)
            Pt = self.predict_covariance(Pt, F, Qt)

            # Update step based on sensor type.
            if sensor_type == 'GPS':
                H_GPS = self.get_gps_observation_matrix()
                R_GPS = self.get_gps_measurement_noise_covariance_matrix()
                Zt_GPS = [sensor_data['easting'], sensor_data['northing'], sensor_data['altitude']]
                Kt = self.calculate_kalman_gain(Pt, H_GPS, R_GPS)
                y = Zt_GPS - np.dot(H_GPS, xt)
                xt = xt + np.dot(Kt, y)
                Pt = np.dot(I - np.dot(Kt, H_GPS), Pt)
            elif sensor_type == 'IMU':
                ax = sensor_data[7]
                ay = sensor_data[8]
                az = sensor_data[9]
                Vx = xt[6] + ax * dt
                Vy = xt[7] + ay * dt
                Vz = xt[8] + az * dt
                X = xt[0] + Vx * dt
                Y = xt[1] + Vy * dt
                Z = xt[2] + Vz * dt
                roll = sensor_data[1]
                pitch = sensor_data[2]
                yaw = sensor_data[3]
                ang_x = sensor_data[4]
                ang_y = sensor_data[5]
                ang_z = sensor_data[6]
                Zt_IMU = [X, Y, Z, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]
                H_IMU = self.get_imu_observation_matrix()
                R_IMU = self.get_imu_measurement_noise_covariance_matrix()
                Kt = self.calculate_kalman_gain(Pt, H_IMU, R_IMU)
                y = np.array(Zt_IMU) - np.dot(H_IMU, xt)
                xt = xt + np.dot(Kt, y)
                Pt = np.dot(I - np.dot(Kt, H_IMU), Pt)
            
            sf_KF_state.append((time, xt[0], xt[1], xt[2], xt[3], xt[4], xt[5]))
            previous_time = time

        return sf_KF_state

    def calculate_accuracy_metrics(self, sf_KF_state_with_time):
        utm_data = self.get_utm_data()
        if not sf_KF_state_with_time or not utm_data:
            print("State list or UTM data is empty. Cannot calculate accuracy.")
            return None

        gps_times = np.array([entry['time'] for entry in utm_data])
        gps_positions = np.array([[entry['easting'], entry['northing'], entry['altitude']] for entry in utm_data])

        kf_times = np.array([state[0] for state in sf_KF_state_with_time])
        kf_positions = np.array([state[1:4] for state in sf_KF_state_with_time])

        # Interpolate GPS positions to KF times
        interp_east = interp1d(gps_times, gps_positions[:, 0], kind='linear', fill_value="extrapolate")
        interp_north = interp1d(gps_times, gps_positions[:, 1], kind='linear', fill_value="extrapolate")
        interp_alt = interp1d(gps_times, gps_positions[:, 2], kind='linear', fill_value="extrapolate")
        gps_interp_positions = np.stack([interp_east(kf_times), interp_north(kf_times), interp_alt(kf_times)], axis=1)

        position_errors = kf_positions - gps_interp_positions
        euclidean_errors = np.linalg.norm(position_errors, axis=1)
        total_position_rmse = np.sqrt(np.mean(euclidean_errors**2))

        # print(f"Accuracy Calculation Complete:")
        # print(f"  - Compared {len(sf_KF_state_with_time)} KF states to closest GPS points.")
        # print(f"  - RMSE X: {rmse_x:.4f} meters")
        # print(f"  - RMSE Y: {rmse_y:.4f} meters")
        # print(f"  - RMSE Z: {rmse_z:.4f} meters")
        # print(f"  - Overall Position RMSE: {total_position_rmse:.4f} meters")
        # return {
        #         'total_position_rmse': total_position_rmse,
        #         'position_errors': position_errors,
        #         'euclidean_errors': euclidean_errors,
        #         'kf_times': kf_times,
        #         'kf_positions': kf_positions,
        #         'gps_interp_positions': gps_interp_positions
        #     }
        return total_position_rmse

    def run_sampled_brute_force_kalman_filter(self, start_idx=0, end_idx=None, overwrite_sampling_freq=None, max_combos_in_memory=1000):
        from itertools import product, islice
        class_args = {
            'get_state_transition_matrix': self.get_state_transition_matrix,
            'get_process_noise_covariance_matrix': self.get_process_noise_covariance_matrix,
            'predict_covariance': self.predict_covariance,
            'get_gps_observation_matrix': self.get_gps_observation_matrix,
            'get_gps_measurement_noise_covariance_matrix': self.get_gps_measurement_noise_covariance_matrix,
            'get_imu_observation_matrix': self.get_imu_observation_matrix,
            'get_imu_measurement_noise_covariance_matrix': self.get_imu_measurement_noise_covariance_matrix,
            'calculate_kalman_gain': self.calculate_kalman_gain,
            'calculate_accuracy_metrics': self.calculate_accuracy_metrics,
        }

        xt = np.zeros(15)
        Pt = np.diag([1000]*3 + [100]*3 + [100]*3 + [100]*3 + [1000]*3)
        I = np.eye(15)

        if end_idx is None:
            end_idx = len(self.indexed_sensor_data)

        gps_started = False
        previous_time = None
        sensor_data_queues = []

        # Step 1: Collect all sensor_data_queues (one per sampling period)
        sensor_data_queue = []
        for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
            if not gps_started and sensor_type == 'GPS':
                xt[0] = sensor_data['easting']
                xt[1] = sensor_data['northing']
                xt[2] = sensor_data['altitude']
                gps_started = True
                previous_time = time
            if not gps_started:
                print("WARN: No GPS measurement found in the sensor subset.")
                continue

            if (overwrite_sampling_freq is not None):
                processing_frequency = overwrite_sampling_freq
            else:
                processing_frequency = self.processing_frequency

            if time - previous_time < 1/processing_frequency:
                sensor_data_queue.append((index, sensor_type, time, sensor_data))
                continue

            if len(sensor_data_queue) == 0:
                sensor_data_queue.append((index, sensor_type, time, sensor_data))

            sensor_data_queues.append(sensor_data_queue.copy())
            sensor_data_queue = []
            previous_time = time

        # Step 2: Brute force all possible combinations (one sensor per queue)
        queue_lengths = [len(q) for q in sensor_data_queues]
        if any(l == 0 for l in queue_lengths):
            print("Warning: At least one queue is empty, no combinations possible.")
            return None

        log_total_combos = sum(math.log10(l) for l in queue_lengths)
        if log_total_combos > 12:
            total_combos = 1
            for l in queue_lengths:
                total_combos *= l
            total_combos_decimal = decimal.Decimal(total_combos)
            print(f"Total combinations: {total_combos_decimal:.2E} (too large to enumerate in memory)")
        else:
            total_combos = 1
            for l in queue_lengths:
                total_combos *= l
            print(f"Total combinations: {total_combos}")

        best_metric = float('inf')
        best_trajectory = None
        best_sensors = None
        best_state = None
        best_cov = None

        def combo_generator():
            return product(*sensor_data_queues)

        processed = 0
        chunk_size = max_combos_in_memory
        combos = combo_generator()
        print("Beginning brute force search...")
        print(f"Total combinations to process: {total_combos}")

        # Use all available CPU cores
        # num_workers = multiprocessing.cpu_count()
        print("Total number of cpu cores available: ", multiprocessing.cpu_count())
        num_workers = 50
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            chunk_list = []
            while True:
                chunk = list(islice(combos, chunk_size))
                if not chunk:
                    break
                futures.append(executor.submit(evaluate_combo_chunk_worker, chunk, xt, Pt, class_args))
                chunk_list.append(len(chunk))  # Track chunk sizes for progress

            # Use tqdm to show progress
            with tqdm(total=total_combos, desc="Processing combinations") as pbar:
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    results = future.result()
                    for metric, traj, combo, xt_bf, Pt_bf in results:
                        if metric is not None and metric < best_metric:
                            best_metric = metric
                            best_trajectory = traj
                            best_sensors = combo
                            best_state = xt_bf
                            best_cov = Pt_bf
                    processed += chunk_list[i]
                    pbar.update(chunk_list[i])

        print("Brute force search complete.")
        return {
            'selected_sensor_measurements': best_sensors,
            'final_state': best_state,
            'final_covariance': best_cov,
            'trajectory': best_trajectory,
            'accuracy_metric': best_metric
        }
    
    def run_dead_reckoning_for_IMU(self):
        """Run dead reckoning to generate IMU estimates."""
        # Initial state and covariance

        # [x, y, z, roll, pitch, yaw, v_x, v_y, v_z, angular_x, angular_y, angular_z, a_x, a_y, a_z]
        xt = np.array([0, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0])
        Pt = np.array([
                        [1000, 0, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],       # x
                        [0, 1000, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],       # y
                        [0, 0, 1000,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],       # z
                        [0, 0, 0,        100, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],       # roll
                        [0, 0, 0,        0, 100, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],       # pitch
                        [0, 0, 0,        0, 0, 100,     0, 0, 0,    0, 0, 0,    0, 0, 0],       # yaw
                        [0, 0, 0,        0, 0, 0,     100, 0, 0,    0, 0, 0,    0, 0, 0],       # v_x
                        [0, 0, 0,        0, 0, 0,     0, 100, 0,    0, 0, 0,    0, 0, 0],       # v_y
                        [0, 0, 0,        0, 0, 0,     0, 0, 100,    0, 0, 0,    0, 0, 0],       # v_z
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    100, 0, 0,    0, 0, 0],       # angular_x
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 100, 0,    0, 0, 0],       # angular_y
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 100,    0, 0, 0],       # angular_z
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      1000, 0, 0],    # a_x
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 1000, 0],    # a_y
                        [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 0, 1000],    # a_z
                    ])
        I = np.eye(15)  # 8x8 Identity matrix

        # List to store dead-reckoned IMU estimates
        deadreckoned_IMU_estimates = []
# unbias_entry = entry[:1] + [roll, pitch, yaw] + unbias_angular_velocity.tolist() + unbias_linear_acceleration.tolist() + entry[11:]             
        previous_time = None
        # for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data):
            
        return deadreckoned_IMU_estimates

    #### Visualization functions ####

    def plot_kf_states_2d(self, sf_KF_state, gps_alpha=0.1, kf_alpha=0.5, save_path='kf_plot.png'):
        """Plot the estimated states from the Kalman Filter against GPS data."""
        X = [state[1] for state in sf_KF_state]
        Y = [state[2] for state in sf_KF_state]
        # Extracting Easting and Northing from GPS data
        eastings = [data['easting'] for data in self.utm_data]
        northings = [data['northing'] for data in self.utm_data]
        plt.figure(figsize=(12, 6))
        # Plot Estimated Trajectory (X, Y) and GPS data (translated Easting, translated Northing)
        plt.scatter(X, Y, label='Estimated Trajectory (X, Y)', marker='o', s=1, alpha=kf_alpha)
        plt.scatter(eastings, northings, label='GPS Data (Easting, Northing)', marker='o', s=0.5, alpha=gps_alpha)
        plt.xlabel('X / Easting')
        plt.ylabel('Y / Northing')
        plt.title('Estimated Trajectory and GPS Data')
        plt.legend()
        plt.grid(True)
        plt.xlim(-2000, 3000)
        plt.ylim(-1000, 3000)
        # plt.tight_layout()
        plt.savefig(save_path)
        # plt.show()

    def animate_kf_states_2d(self, sf_KF_state, save_path='kf_animation.mp4', skip_rate=200):
        """Animate the estimated states from the Kalman Filter against GPS data."""
        X = np.array([state[1] for state in sf_KF_state])
        Y = np.array([state[2] for state in sf_KF_state])
        
        eastings = np.array([data['easting'] for data in self.utm_data])
        northings = np.array([data['northing'] for data in self.utm_data])

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Estimated Trajectory and GPS Data')
        ax.set_xlabel('X / Easting')
        ax.set_ylabel('Y / Northing')
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.grid(True)

        est_line, = ax.plot([], [], '-', color='blue', linewidth=1.5, label='Estimated Trajectory')
        gps_line, = ax.plot([], [], 'o', color='red', markersize=0.1, alpha=0.2, label='GPS Data')
        ax.legend()

        num_frames = len(X)//skip_rate

        def init():
            est_line.set_data([], [])
            gps_line.set_data([], [])
            return est_line, gps_line

        def update(frame):
            idx = frame * skip_rate
            
            print(f"Rendering frame {frame}/{num_frames}", end='\r')
            est_line.set_data(X[:idx], Y[:idx])
            gps_line.set_data(eastings[:idx], northings[:idx])
            return est_line, gps_line


        ani = animation.FuncAnimation(
            fig, update, frames=num_frames, init_func=init,
            interval=1, blit=True, repeat=False
        )

        ani.save(save_path, fps=30, dpi=150)
        plt.close(fig)
        
    def plot_kf_states_3d(self, sf_KF_state):
        """Plot the estimated states from the Kalman Filter against GPS data."""
        # Estimatated States
        X = [state[0] for state in sf_KF_state]
        Y = [state[1] for state in sf_KF_state]
        Z = [state[2] for state in sf_KF_state]
        
        # GPS data
        eastings = [data['easting'] for data in self.utm_data]
        northings = [data['northing'] for data in self.utm_data]
        altitude = [data['altitude'] for data in self.utm_data]
        
        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(X, Y, Z, label='KF Estimated Path', color='blue')
        ax.plot(eastings, northings, altitude, label='GPS Path', color='red', linestyle='--')

        ax.set_xlabel('Easting / X')
        ax.set_ylabel('Northing / Y')
        ax.set_zlabel('Altitude / Z')
        ax.set_title('3D Trajectory Comparison: Kalman Filter vs. GPS')
        ax.legend()

        plt.show()

    def plot_deadreckoned_imu_with_gps_2d(self, deadreckoned_IMU_estimates):
        """Plot dead-reckoned IMU estimates against raw GPS data."""
        # Extracting X, Y from dead-reckoned IMU data
        X_IMU = [state[0] for state in deadreckoned_IMU_estimates]
        Y_IMU = [state[1] for state in deadreckoned_IMU_estimates]
        # Extracting Easting and Northing from GPS data
        eastings = [data['easting'] for data in self.utm_data]
        northings = [data['northing'] for data in self.utm_data]
        plt.figure(figsize=(12, 6))
        plt.scatter(X_IMU, Y_IMU, label='Dead-Reckoned IMU Trajectory (X, Y)', marker='o', s=3)
        plt.scatter(eastings, northings, label='GPS Data (Easting, Northing)', marker='x', s=3)
        plt.xlabel('X / Easting')
        plt.ylabel('Y / Northing')
        plt.title('Dead-Reckoned IMU Trajectory and GPS Data')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def plot_theta_comparisons(self, sf_KF_state, deadreckoned_IMU_estimates):
        """Plot theta values from different sources for comparison."""
        # (X, Y, Z, roll, pitch, yaw)
        roll_KF = [state[3] for state in sf_KF_state]
        roll_IMU_deadreckoned = [state[3] for state in deadreckoned_IMU_estimates]
        # Extract from IMU data
        # unbias_entry = entry[:1] + [roll, pitch, yaw] + unbias_angular_velocity.tolist() + unbias_linear_acceleration.tolist() + entry[11:]
        roll_IMU_raw = [np.degrees(float(entry[1])) for entry in self.unbias_imu_data]

        # (X, Y, Z, roll, pitch, yaw)
        pitch_KF = [state[4] for state in sf_KF_state]
        pitch_IMU_deadreckoned = [state[4] for state in deadreckoned_IMU_estimates]
        # Extract from IMU data
        # unbias_entry = entry[:1] + [roll, pitch, yaw] + unbias_angular_velocity.tolist() + unbias_linear_acceleration.tolist() + entry[11:]
        pitch_IMU_raw = [np.degrees(float(entry[2])) for entry in self.unbias_imu_data]

        # (X, Y, Z, roll, pitch, yaw)
        yaw_KF = [state[5] for state in sf_KF_state]
        yaw_IMU_deadreckoned = [state[5] for state in deadreckoned_IMU_estimates]
        # Extract from IMU data
        # unbias_entry = entry[:1] + [roll, pitch, yaw] + unbias_angular_velocity.tolist() + unbias_linear_acceleration.tolist() + entry[11:]
        yaw_IMU_raw = [np.degrees(float(entry[3])) for entry in self.unbias_imu_data]

        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Roll plot
        axs[0].plot(roll_KF, label='Kalman Filter Roll', marker='o', markersize=2)
        axs[0].plot(roll_IMU_deadreckoned, label='Dead Reckoned IMU Roll', marker='o', markersize=2)
        axs[0].plot(roll_IMU_raw, label='Raw IMU Roll', marker='o', markersize=2)
        axs[0].set_ylabel('Roll (Degrees)')
        axs[0].set_title('Roll Comparison')
        axs[0].legend()
        axs[0].grid(True)

        # Pitch plot
        axs[1].plot(pitch_KF, label='Kalman Filter Pitch', marker='o', markersize=2)
        axs[1].plot(pitch_IMU_deadreckoned, label='Dead Reckoned IMU Pitch', marker='o', markersize=2)
        axs[1].plot(pitch_IMU_raw, label='Raw IMU Pitch', marker='o', markersize=2)
        axs[1].set_ylabel('Pitch (Degrees)')
        axs[1].set_title('Pitch Comparison')
        axs[1].legend()
        axs[1].grid(True)

        # Yaw plot
        axs[2].plot(yaw_KF, label='Kalman Filter Yaw', marker='o', markersize=2)
        axs[2].plot(yaw_IMU_deadreckoned, label='Dead Reckoned IMU Yaw', marker='o', markersize=2)
        axs[2].plot(yaw_IMU_raw, label='Raw IMU Yaw', marker='o', markersize=2)
        axs[2].set_ylabel('Yaw (Degrees)')
        axs[2].set_xlabel('Time Step')
        axs[2].set_title('Yaw Comparison')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()

    def animate_kf_states_3d(self, sf_KF_state, save_path='kf_animation_3d.mp4', skip_rate=100):
        """Animate the estimated states from the Kalman Filter against GPS data in 3D."""
        X = np.array([state[0] for state in sf_KF_state])
        Y = np.array([state[1] for state in sf_KF_state])
        Z = np.array([state[2] for state in sf_KF_state])

        eastings = np.array([data['easting'] for data in self.utm_data])
        northings = np.array([data['northing'] for data in self.utm_data])
        altitudes = np.array([data['altitude'] for data in self.utm_data])  # Default to 0 if missing

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D Estimated Trajectory and GPS Data')
        ax.set_xlabel('X / Easting')
        ax.set_ylabel('Y / Northing')
        ax.set_zlabel('Z / Altitude')
        ax.set_xlim(-5000, 5000)
        ax.set_ylim(-5000, 5000)
        ax.set_zlim(-100, 500)
        ax.grid(True)

        est_line, = ax.plot3D([], [], [], '-', color='blue', linewidth=1.5, label='Estimated Trajectory')
        gps_line, = ax.plot3D([], [], [], 'o', color='red', markersize=0.1, alpha=0.3, label='GPS Data')
        ax.legend()

        num_frames = len(X) // skip_rate

        def init():
            est_line.set_data([], [])
            est_line.set_3d_properties([])
            gps_line.set_data([], [])
            gps_line.set_3d_properties([])
            return est_line, gps_line

        def update(frame):
            idx = frame * skip_rate
            print(f"Rendering frame {frame}/{num_frames}", end='\r')

            est_line.set_data(X[:idx], Y[:idx])
            est_line.set_3d_properties(Z[:idx])
            gps_line.set_data(eastings[:idx], northings[:idx])
            gps_line.set_3d_properties(altitudes[:idx])
            return est_line, gps_line

        ani = animation.FuncAnimation(
            fig, update, frames=num_frames, init_func=init,
            interval=1, blit=False, repeat=False  # blit=False is safer for 3D
        )

        ani.save(save_path, fps=30, dpi=150)
        plt.close(fig)
    
    def get_utm_data(self):
        """Returns the UTM data."""
        return self.utm_data


def plot_brute_force_vs_standard(utm_data, best_set, standard_kf_trajectory, start_idx, start_offset):
    # Get the timestamps from indexed_sensor_data for the start and end indices
    start_time = sensor_fusion.indexed_sensor_data[start_idx][2]
    end_time = sensor_fusion.indexed_sensor_data[start_idx + start_offset][2]

    # Find the closest GPS entries in UTM data for these times
    gps_times = [entry['time'] for entry in utm_data]
    def find_closest_gps_idx(target_time):
        return min(range(len(gps_times)), key=lambda i: abs(gps_times[i] - target_time))
    gps_idx_start = find_closest_gps_idx(start_time)
    gps_idx_end = find_closest_gps_idx(end_time)
    gps_start = utm_data[gps_idx_start]
    gps_end = utm_data[gps_idx_end]
    center_easting = (gps_start['easting'] + gps_end['easting']) / 2
    center_northing = (gps_start['northing'] + gps_end['northing']) / 2

    # Extract trajectories
    bf_traj = best_set['trajectory']
    std_traj = standard_kf_trajectory
    bf_X = [state[1] for state in bf_traj]
    bf_Y = [state[2] for state in bf_traj]
    std_X = [state[1] for state in std_traj]
    std_Y = [state[2] for state in std_traj]

    # GPS for reference (using the range between the two GPS indices)
    gps_X = [utm_data[i]['easting'] for i in range(min(gps_idx_start, gps_idx_end), max(gps_idx_start, gps_idx_end) + 1)]
    gps_Y = [utm_data[i]['northing'] for i in range(min(gps_idx_start, gps_idx_end), max(gps_idx_start, gps_idx_end) + 1)]

    plt.figure(figsize=(10, 8))
    plt.scatter(gps_X, gps_Y, label='GPS Reference', c='black', s=10, alpha=0.5)
    plt.plot(bf_X, bf_Y, label='Brute Force KF', color='blue', linewidth=2)
    plt.plot(std_X, std_Y, label='Standard KF', color='orange', linewidth=2, linestyle='--')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Brute Force vs Standard KF Trajectory')
    plt.legend()
    plt.grid(True)
    # Center window around the midpoint of the GPS segment
    window_size = 5  # meters, adjust as needed
    plt.xlim(center_easting - window_size, center_easting + window_size)
    plt.ylim(center_northing - window_size, center_northing + window_size)
    plt.tight_layout()
    plt.show()

def plot_kf_centered_comparison(utm_data, best_set, standard_kf_trajectory):
    # Extract standard KF trajectory and its center
    std_traj = standard_kf_trajectory
    std_X = [state[1] for state in std_traj]
    std_Y = [state[2] for state in std_traj]
    center_easting = (min(std_X) + max(std_X)) / 2
    center_northing = (min(std_Y) + max(std_Y)) / 2

    # Extract brute force KF trajectory
    bf_traj = best_set['trajectory']
    bf_X = [state[1] for state in bf_traj]
    bf_Y = [state[2] for state in bf_traj]

    # GPS for reference (using time range of the standard KF trajectory)
    gps_times = [entry['time'] for entry in utm_data]
    std_times = [state[0] for state in std_traj]
    def find_closest_gps_idx(target_time):
        return min(range(len(gps_times)), key=lambda i: abs(gps_times[i] - target_time))
    gps_idx_start = find_closest_gps_idx(std_times[0])
    gps_idx_end = find_closest_gps_idx(std_times[-1])
    gps_X = [utm_data[i]['easting'] for i in range(min(gps_idx_start, gps_idx_end), max(gps_idx_start, gps_idx_end) + 1)]
    gps_Y = [utm_data[i]['northing'] for i in range(min(gps_idx_start, gps_idx_end), max(gps_idx_start, gps_idx_end) + 1)]

    plt.figure(figsize=(10, 8))
    plt.scatter(gps_X, gps_Y, label='GPS Reference', c='black', s=10, alpha=0.5)
    plt.plot(bf_X, bf_Y, label='Brute Force KF', color='blue', linewidth=2)
    plt.plot(std_X, std_Y, label='Standard KF', color='orange', linewidth=2, linestyle='--')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Brute Force vs Standard KF Trajectory')
    plt.legend()
    plt.grid(True)
    # Center window around the midpoint of the standard KF segment
    window_size = 500  # meters, adjust as needed
    plt.xlim(center_easting - window_size, center_easting + window_size)
    plt.ylim(center_northing - window_size, center_northing + window_size)
    plt.tight_layout()
    plt.show()

# main code
if __name__ == "__main__":
    # Define file paths for GPS and IMU data
    gps_csv_file = 'gps_data.csv'
    imu_csv_file = 'imu_data.csv'

    # Create an instance of KF_SensorFusion
    sensor_fusion = KF_SensorFusion(gps_csv_file, imu_csv_file)

    # Load data from CSV files
    sensor_fusion.load_data()

    # Convert GPS data to UTM coordinates and appened Altitude
    sensor_fusion.gps_to_modified_utm()

    # Compute biases from IMU data
    angular_velocity_bias, linear_acceleration_bias, first_valid_index = sensor_fusion.compute_imu_biases(sensor_fusion.gps_data, sensor_fusion.imu_data)

    # Unbias IMU data
    sensor_fusion.unbias_imu_data(angular_velocity_bias, linear_acceleration_bias)

    # Compute stationary orientation of the IMU
    avg_roll, avg_pitch, avg_yaw = sensor_fusion.compute_stationary_orientation(first_valid_index)

    # Combine GPS and IMU data and sort by timestamp
    sensor_fusion.combine_sensor_data()

    # Run Kalman filter for sensor fusion
    # sf_KF_state = sensor_fusion.run_kalman_filter_scheduled() # Output (Time Steps x 6) where stored values are (X, Y, Z, roll, pitch, yaw)

    # deadreckoned_IMU_estimates = sensor_fusion.run_dead_reckoning_for_IMU() # Output (Time Steps x 6) where stored values are (X, Y, Z, roll, pitch, yaw)


    # Evaluate Kalman Filter performance with brute force search
    # mi could be GPS measurement or IMU measurement # i is a index/number

    # [m1,m2,m3, ... m100000] # orignal measurements for trajectory

    # [m54717, ..., 54767] # selected portion of trajectory

    # # Ex. Subsets of measurments sampled and to be evaluated
    # 4/50 seconds
    # [[mm54717_GPS, m54718_IMU, m54719_IMU], [m54720_IMU, m54721_IMU], [m54722_IMU, m54723_IMU], [m54724_GPS, m54725_IMU, m54726_IMU]]
    


    # [mm54717_GPS, m54720_IMU, m54722_IMU, m54724_GPS] --> calculate accuracy metric --> update best if better
    # [mm54717_GPS, m54720_IMU, m54722_IMU, m54725_IMU] --> calculate accuracy metric --> update best if better
    # [mm54717_GPS, m54720_IMU, m54722_IMU, m54726_IMU] --> calculate accuracy metric --> update best if better

    # [mm54717_GPS, m54720_IMU, m54723_IMU, m54724_GPS] --> calculate accuracy metric --> update best if better
    # [mm54717_GPS, m54720_IMU, m54723_IMU, m54725_IMU] --> calculate accuracy metric --> update best if better
    # [mm54717_GPS, m54720_IMU, m54723_IMU, m54726_IMU] --> calculate accuracy metric --> update best if better
    # # subset size is 2 or 3

    # start_offset * 1/sampling_frq = seconds of trajectory being processed/evaluated

    start_idx = 54717  # i1_t=0 i2_t=0.0001 g1_t=0.0006 i3 i4 ... i100 g40
    start_offset = 100
    sampling_frq = 50

    # UIOP
    # I = GPS
    # U = IMU
    # P = Radar
    # O = Images

    # TIME                  # SENSOR MEASSURMENT SUBSET
    # 0 - 1/50 (0 - 0.02) seconds      # U U I (0.019) I (0.0001)
    # 1/50 - 2/50 seconds   # U U U P O O
    # 2/50 - 3/50 seconds   # U P O I
    # 3/50 - 4/50 seconds   # U U I P

    # best_set = sensor_fusion.run_sampled_brute_force_kalman_filter(start_idx, start_idx + start_offset, sampling_frq)
    # print("Brute Force - Accuracy Metric (Total Position RMSE):", best_set['accuracy_metric']) # Accuracy Metric (Total Position RMSE): 1.2345
    
    output = sensor_fusion.run_kalman_filter(start_idx, start_idx + start_offset) # ONLY USE IMU, BROKEN
    print("Regular KF - Accuracy Metric (Total Position RMSE):", sensor_fusion.calculate_accuracy_metrics(output)) # Accuracy
                # sf_KF_state.append((time, xt[0], xt[1], xt[2], xt[3], xt[4], xt[5]))  # Append initial state

    sensor_fusion.set_processing_frequency(sampling_frq) # Set to 50 Hz for comparison
    standard_kf_trajectory = sensor_fusion.run_kalman_filter_scheduled(start_idx, start_idx + start_offset)
    
    print("Accuracy Metric (Total Position RMSE):", sensor_fusion.calculate_accuracy_metrics(standard_kf_trajectory)) # Accuracy
    # plot_kf_centered_comparison(sensor_fusion.get_utm_data(), best_set, standard_kf_trajectory)