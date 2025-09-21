import os
import csv
import numpy as np
import utm
import matplotlib.pyplot as plt
import signal
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import operator
import concurrent.futures
from multiprocessing import Pool
import multiprocessing as mp
import sys
from scipy.interpolate import interp1d
import math
import decimal


def evaluate_combo_chunk_worker(chunk, xt, Pt, class_args, prev_time, last_time):
    """Worker function with better error handling and memory management."""
    results = []
    
    try:
        for combo in chunk:
            try:
                xt_bf = xt.copy()
                Pt_bf = Pt.copy()
                traj = []
                log_det = []
                # Use the passed-in prev_time for the first element of the combo
                current_prev_time = prev_time 
                
                for (idx, stype, t, sdata) in combo:
                    dt = t - current_prev_time if current_prev_time is not None else 0
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
                        Z_pos = xt_bf[2] + Vz * dt
                        roll, pitch, yaw = sdata[1], sdata[2], sdata[3]
                        ang_x, ang_y, ang_z = sdata[4], sdata[5], sdata[6]
                        Z = [X, Y, Z_pos, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]
                        H = class_args['get_imu_observation_matrix']()
                        R = class_args['get_imu_measurement_noise_covariance_matrix']()
                    
                    K = class_args['calculate_kalman_gain'](Pt_bf, H, R)
                    y = np.array(Z) - np.dot(H, xt_bf)
                    xt_bf = xt_bf + np.dot(K, y)
                    Pt_bf = np.dot(np.eye(15) - np.dot(K, H), Pt_bf)
                    traj.append((t, *xt_bf[:6]))
                    current_prev_time = t
                    sign_log_det, value_log_det = np.linalg.slogdet(Pt_bf)
                    log_det.append(sign_log_det * value_log_det)
                
                if current_prev_time is not None and current_prev_time < last_time:
                    dt = last_time - current_prev_time
                    F = class_args['get_state_transition_matrix'](dt)
                    Qt = class_args['get_process_noise_covariance_matrix'](dt)
                    xt_bf = np.dot(F, xt_bf)
                    Pt_bf = class_args['predict_covariance'](Pt_bf, F, Qt)
                    traj.append((last_time, *xt_bf[:6]))
                    sign_log_det, value_log_det = np.linalg.slogdet(Pt_bf)
                    log_det.append(sign_log_det * value_log_det)

                # Calculate metric for this combination
                # try:
                #     metric_result = class_args['calculate_accuracy_metrics'](traj)
                #     if metric_result and 'total_position_rmse' in metric_result:
                #         metric = metric_result['total_position_rmse']
                #     else:
                #         metric = float('inf')  # Invalid result
                # except Exception as metric_error:
                #     print(f"Error calculating metrics: {metric_error}")
                metric = float('inf')
                
                # Only keep essential data to reduce memory usage
                results.append((metric, traj, combo, xt_bf.copy(), None, log_det, len(combo)))  # Don't return covariance to save memory
                
            except Exception as combo_error:
                print(f"Error processing combination: {combo_error}")
                # Continue with next combination instead of failing entire chunk
                continue
                
    except Exception as chunk_error:
        print(f"Error processing chunk: {chunk_error}")
        return []  # Return empty results for this chunk
    
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


    def random_schedule(self, num_measurements=None):
        assert num_measurements is not None, "measurements should not be None"
        selected_indices = np.random.choice(num_measurements)
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
        # print("First two entries in unbias_imu_data:")
        # for i in range(2):
        #     print(unbias_imu_data[i])
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
        # gps_printed = imu_printed = False
        # for entry in self.indexed_sensor_data:
        #     if entry[1] == 'GPS' and not gps_printed:
        #         print("First GPS Entry:", entry)
        #         gps_printed = True
        #     elif entry[1] == 'IMU' and not imu_printed:
        #         print("First IMU Entry:", entry)
        #         imu_printed = True
        #     if gps_printed and imu_printed:
        #         break
            
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
    
    def run_kalman_filter_full(self, start_idx=None, end_idx=None, initial_pt=None, initial_state=None, print_output=False):
        """
        Run the Kalman Filter over all available sensor data and store the resulting state
        trajectory, covariance matrices, and log determinant of covariance as ground truth.
        This version is refactored for robustness to prevent list length mismatches.
        """
        if not hasattr(self, 'indexed_sensor_data') or not self.indexed_sensor_data:
            print("Sensor data has not been combined or is empty.")
            return [], [], []

        # --- Initialization Step ---
        if start_idx is None or start_idx < 0:
            start_idx = 0
            print("Overriding start_idx to 0.")
        if end_idx is None or end_idx > len(self.indexed_sensor_data):
            end_idx = len(self.indexed_sensor_data)
            print("Overriding end_idx to the length of indexed_sensor_data.")

        xt = np.zeros(15)
        I = np.eye(15)
        if initial_pt is not None and initial_state is not None:
            Pt = initial_pt
            xt[0:6] = initial_state[1:7]  # Set initial position and orientation
            prev_time = initial_state[0]

            start_idx_offset = start_idx
            print("Using provided initial covariance matrix.")
        else:
            Pt = np.diag([10000]*3 + [1000]*3 + [1000]*3 + [1000]*3 + [10000]*3)
            # Find the first GPS measurement to initialize the state robustly
            start_idx_offset = -1
            prev_time = None
            for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:]):
                if sensor_type == 'GPS':
                    xt[0] = sensor_data['easting']
                    xt[1] = sensor_data['northing']
                    xt[2] = sensor_data['altitude']
                    prev_time = time
                    start_idx_offset = i
                    break
            
            if start_idx_offset == -1:
                print("No GPS data found to initialize the filter.")
                return [], [], []

        # Initialize all lists with the state *at* the first GPS point
        sf_KF_state = [(prev_time, *xt[:6])]
        sf_KF_covariance = [Pt.copy()]
        sign, log_det = np.linalg.slogdet(Pt)
        sf_KF_log_det = [log_det]
        total_sensor_count = 0

        # --- Main Loop Step ---
        # Loop through all data points *after* the initializing GPS point
        for (index, sensor_type, time, sensor_data) in self.indexed_sensor_data[start_idx_offset + 1:end_idx]:
            dt = time - prev_time
            if dt < 0: # Sanity check for out-of-order data
                prev_time = time
                continue

            # Prediction Step
            F = self.get_state_transition_matrix(dt)
            Qt = self.get_process_noise_covariance_matrix(dt)
            xt = np.dot(F, xt)
            Pt = self.predict_covariance(Pt, F, Qt)

            # Update Step
            if sensor_type == 'GPS':
                H = self.get_gps_observation_matrix()
                R = self.get_gps_measurement_noise_covariance_matrix()
                Z = [sensor_data['easting'], sensor_data['northing'], sensor_data['altitude']]
            else: # IMU
                ax, ay, az = sensor_data[7], sensor_data[8], sensor_data[9]
                Vx, Vy, Vz = xt[6] + ax * dt, xt[7] + ay * dt, xt[8] + az * dt
                X, Y, Z_pos = xt[0] + Vx * dt, xt[1] + Vy * dt, xt[2] + Vz * dt
                roll, pitch, yaw = sensor_data[1], sensor_data[2], sensor_data[3]
                ang_x, ang_y, ang_z = sensor_data[4], sensor_data[5], sensor_data[6]
                Z = [X, Y, Z_pos, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]
                H = self.get_imu_observation_matrix()
                R = self.get_imu_measurement_noise_covariance_matrix()
            
            K = self.calculate_kalman_gain(Pt, H, R)
            y = np.array(Z) - np.dot(H, xt)
            xt = xt + np.dot(K, y)
            Pt = np.dot(I - np.dot(K, H), Pt)

            # Append results for this time step to all three lists
            sf_KF_state.append((time, *xt[:6]))
            sf_KF_covariance.append(Pt.copy())
            sign, log_det = np.linalg.slogdet(Pt)
            sf_KF_log_det.append(log_det)
            
            total_sensor_count += 1

            prev_time = time

        self._ground_truth = sf_KF_state
        self._ground_truth_cov = sf_KF_covariance
        if print_output:
            print(f"---------------------------- Full Kalman Filter ----------------------------\n Processed {total_sensor_count} measurements out of {end_idx - start_idx_offset} in total, from index {start_idx_offset} to {end_idx}")
        
        return sf_KF_state, sf_KF_log_det, Pt

    def get_GT(self):
        """Return the ground truth trajectory computed by the Kalman Filter."""
        if hasattr(self, '_ground_truth'):
            return self._ground_truth
        else:
            print("Ground truth not computed yet. Please run run_kalman_filter_full() first.")
            return None

    def run_kalman_filter(self, start_idx, end_idx):
        """Run the Kalman Filter to fuse IMU and GPS data for pose estimation."""
        # Initial state [x, y, z, roll, pitch, yaw, v_x, v_y, v_z, angular_x, angular_y, angular_z, a_x, a_y, a_z]
        xt = np.array([0, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0])

        # Initial covariance matrix
        Pt = np.array([
            [10000, 0, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 10000, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 10000,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        1000, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 1000, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 1000,     0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     1000, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 1000, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 1000,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    1000, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 1000, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 1000,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      10000, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 10000, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 0, 10000],
        ])
        I = np.eye(15)

        # Create a list to store the state (X, Y, Z, roll, pitch, yaw)
        sf_KF_state = [(0, xt[0], xt[1], xt[2], xt[3], xt[4], xt[5])]  # Storing initial state
        sf_KF_covariance = [Pt.copy()]
        gps_started = False  
        for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
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
            sf_KF_covariance.append(Pt.copy())
            previous_time = time  # Update previous time
        return sf_KF_state, sf_KF_covariance

    def run_kalman_filter_scheduled(self, start_idx=None, end_idx=None, initial_pt=None, initial_state=None, selection_method=None,print_output=False):
        """Run the Kalman Filter to fuse IMU and GPS data for pose estimation."""
        if start_idx is None or start_idx < 0:
            start_idx = 0
            print("Overriding start_idx to 0.")
        if end_idx is None or end_idx > len(self.indexed_sensor_data):
            end_idx = len(self.indexed_sensor_data)
            print("Overriding end_idx to the length of indexed_sensor_data.")
        if selection_method not in ['random', 'greedy']:
            print("Invalid selection_method. Choose either 'random' or 'greedy'.")
            return None
        
        # Initialize state and covariance
        xt = np.zeros(15)
        I = np.eye(15)
        total_sensor_count = 0
        if initial_state is not None:
            Pt = initial_pt
            xt[0:6] = initial_state[1:7]  # Set initial position and orientation
            start_idx_offset = start_idx
            previous_time = initial_state[0]
        else:
            Pt = np.diag([10000]*3 + [1000]*3 + [1000]*3 + [1000]*3 + [10000]*3)

            # Find the initial GPS measurement in the subset to initialize xt and previous_time.
            gps_started = False
            previous_time = None
            start_idx_offset = 0
            for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
                if sensor_type == 'GPS':
                    xt[0] = sensor_data['easting']
                    xt[1] = sensor_data['northing']
                    xt[2] = sensor_data['altitude']
                    previous_time = time
                    gps_started = True
                    start_idx_offset = start_idx + i
                    break
            
            if not gps_started:
                print("WARN: No GPS measurement found in the sensor subset.")
                return None, None

        # Store initial state and log determinant
        sf_KF_state = [(previous_time, *xt[:6])]
        sign, initial_log_det = np.linalg.slogdet(Pt)
        sf_KF_log_det = [initial_log_det]
        
        sensor_data_queue = []
        # Start the loop *after* the initial GPS point
        if end_idx == -1:
            end_idx = len(self.indexed_sensor_data)
        for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx_offset + 1 : end_idx]):
            
            # Collect sensor data over the sampling period.
            if time - previous_time < 1/self.processing_frequency:
                sensor_data_queue.append((index, sensor_type, time, sensor_data))
                continue

            if not sensor_data_queue:
                # If the gap was too large, process the current measurement by itself
                sensor_data_queue.append((index, sensor_type, time, sensor_data))
            
            # Choose the best measurement using the scheduler.
            assert len(sensor_data_queue) > 0, "Sensor data queue is empty!!!"
            if selection_method == 'random':
                selected_measurement_global_index = self.scheduler.random_schedule(
                                    num_measurements=len(sensor_data_queue),
                                )
            elif selection_method == 'greedy':
                selected_measurement_global_index = self.scheduler.greedy_schedule(
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
            
            # Get the full data for the selected measurement
            (selected_idx, selected_stype, selected_time, selected_sdata) = sensor_data_queue[selected_measurement_global_index]
            sensor_data_queue = [] # Clear the queue for the next window
            
            # Calculate dt using the timestamp of the *selected* measurement
            dt = selected_time - previous_time

            # Prediction step
            F = self.get_state_transition_matrix(dt)
            Qt = self.get_process_noise_covariance_matrix(dt)
            xt = np.dot(F, xt)
            Pt = self.predict_covariance(Pt, F, Qt)

            # Update step based on the selected sensor type
            if selected_stype == 'GPS':
                H = self.get_gps_observation_matrix()
                R = self.get_gps_measurement_noise_covariance_matrix()
                Z = [selected_sdata['easting'], selected_sdata['northing'], selected_sdata['altitude']]
            else: # IMU
                ax, ay, az = selected_sdata[7], selected_sdata[8], selected_sdata[9]
                Vx, Vy, Vz = xt[6] + ax * dt, xt[7] + ay * dt, xt[8] + az * dt
                X, Y, Z_pos = xt[0] + Vx * dt, xt[1] + Vy * dt, xt[2] + Vz * dt
                roll, pitch, yaw = selected_sdata[1], selected_sdata[2], selected_sdata[3]
                ang_x, ang_y, ang_z = selected_sdata[4], selected_sdata[5], selected_sdata[6]
                Z = [X, Y, Z_pos, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]
                H = self.get_imu_observation_matrix()
                R = self.get_imu_measurement_noise_covariance_matrix()
            
            K = self.calculate_kalman_gain(Pt, H, R)
            y = np.array(Z) - np.dot(H, xt)
            xt = xt + np.dot(K, y)
            Pt = np.dot(I - np.dot(K, H), Pt)
            
            # Append results for this time step
            sf_KF_state.append((selected_time, *xt[:6]))
            sign, log_det = np.linalg.slogdet(Pt)
            sf_KF_log_det.append(log_det)
            
            previous_time = selected_time
            total_sensor_count += 1

        if print_output and selection_method == "random":
            print("---------------------------- Random Scheduled Kalman Filter ----------------------------")
        elif print_output and selection_method == "greedy":
            print("---------------------------- Greedy Scheduled Kalman Filter ----------------------------")
        if print_output:
            print(f"Scheduled KF processed {total_sensor_count} measurements from index {start_idx_offset + 1} to {end_idx}")

        return sf_KF_state, sf_KF_log_det, Pt
    
    def run_adaptive_threshold_kalman_filter(self, start_idx=None, end_idx=None, R_threshold=None, initial_pt=None, initial_state=None, print_output=False):
        """Run the Kalman Filter to fuse IMU and GPS data for pose estimation."""
        # Initialize state and covariance
        xt = np.zeros(15)
        I = np.eye(15)
        total_sensor_count = 0

        # --- Initialization Step ---
        if start_idx is None or start_idx < 0:
            start_idx = 0
            print("Overriding start_idx to 0.")
        if end_idx is None or end_idx > len(self.indexed_sensor_data):
            end_idx = len(self.indexed_sensor_data)
            print("Overriding end_idx to the length of indexed_sensor_data.")
        if R_threshold is None:
            R_threshold = -inf
            print("Overriding R_threshold to - infinity (i.e. max information utilization).")

        if initial_pt is not None and initial_state is not None:
            Pt = initial_pt
            xt[0:6] = initial_state[1:7]  # Set initial position and orientation
            previous_time = initial_state[0]
            start_idx_offset = start_idx 
        else:
            Pt = np.diag([10000]*3 + [1000]*3 + [1000]*3 + [1000]*3 + [10000]*3)

            # Find the initial GPS measurement in the subset to initialize xt and previous_time.
            gps_started = False
            previous_time = None
            start_idx_offset = -1
            for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
                if sensor_type == 'GPS':
                    xt[0] = sensor_data['easting']
                    xt[1] = sensor_data['northing']
                    xt[2] = sensor_data['altitude']
                    previous_time = time
                    gps_started = True
                    start_idx_offset = start_idx + i
                    break
            
            if not gps_started:
                print("WARN: No GPS measurement found in the sensor subset.")
                return None, None

        # Store initial state and log determinant
        sf_KF_state = [(previous_time, *xt[:6])]
        sign, initial_log_det = np.linalg.slogdet(Pt)
        sf_KF_log_det = [initial_log_det]
        
        sensor_data_queue = []

        # Start the loop *after* the initial GPS point
        if end_idx == -1:
            end_idx = len(self.indexed_sensor_data)
        for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx_offset + 1 : end_idx]):

            # Calculate dt using the timestamp of the *selected* measurement
            dt = time - previous_time

            # Prediction step
            F = self.get_state_transition_matrix(dt)
            Qt = self.get_process_noise_covariance_matrix(dt)
            xt = np.dot(F, xt)
            Pt = self.predict_covariance(Pt, F, Qt)

            curr_log_det_sign, curr_log_det = np.linalg.slogdet(Pt)

            if (curr_log_det * curr_log_det_sign > R_threshold): # If too uncertain, process point
                # Update step based on the selected sensor type
                if sensor_type == 'GPS':
                    H = self.get_gps_observation_matrix()
                    R = self.get_gps_measurement_noise_covariance_matrix()
                    Z = [sensor_data['easting'], sensor_data['northing'], sensor_data['altitude']]
                else: # IMU
                    ax, ay, az = sensor_data[7], sensor_data[8], sensor_data[9]
                    Vx, Vy, Vz = xt[6] + ax * dt, xt[7] + ay * dt, xt[8] + az * dt
                    X, Y, Z_pos = xt[0] + Vx * dt, xt[1] + Vy * dt, xt[2] + Vz * dt
                    roll, pitch, yaw = sensor_data[1], sensor_data[2], sensor_data[3]
                    ang_x, ang_y, ang_z = sensor_data[4], sensor_data[5], sensor_data[6]
                    Z = [X, Y, Z_pos, roll, pitch, yaw, Vx, Vy, Vz, ang_x, ang_y, ang_z, ax, ay, az]
                    H = self.get_imu_observation_matrix()
                    R = self.get_imu_measurement_noise_covariance_matrix()
                
                K = self.calculate_kalman_gain(Pt, H, R)
                y = np.array(Z) - np.dot(H, xt)
                xt = xt + np.dot(K, y)
                Pt = np.dot(I - np.dot(K, H), Pt)
                
                # Append results for this time step
                sf_KF_state.append((time, *xt[:6]))
                sign, log_det = np.linalg.slogdet(Pt)
                sf_KF_log_det.append(log_det)
                
                previous_time = time
                total_sensor_count += 1
        
        if print_output:
            print(f"---------------------------- Adaptive Kalman Filter ----------------------------\n Processed {total_sensor_count} measurements out of {end_idx - start_idx_offset} in total, from index {start_idx_offset} to {end_idx}")

        return sf_KF_state, sf_KF_log_det, Pt
    

    def calculate_accuracy_metrics(self, candidate_trajectory):
        """
        Compare a candidate trajectory against the ground truth trajectory.
        This version assumes the candidate trajectory is sorted by time.
        """
        if not candidate_trajectory:
            print("Candidate trajectory is empty. Cannot calculate accuracy.")
            return None

        if not (hasattr(self, "_ground_truth") and self._ground_truth):
            print("Ground truth trajectory is not available. Run run_kalman_filter_full() first.")
            return None

        # --- MODIFIED SECTION ---
        # Directly use the first and last timestamps, assuming the trajectory is sorted.
        candidate_start = candidate_trajectory[0][0]
        candidate_end = candidate_trajectory[-1][0]
        # --- END MODIFIED SECTION ---

        # Extract ground truth section corresponding to the time window
        gt_full = self._ground_truth
        gt_section = [state for state in gt_full if candidate_start <= state[0] <= candidate_end]
        if len(gt_section) < 2:
            gt_times = np.array([state[0] for state in gt_full])
            gt_positions = np.array([state[1:4] for state in gt_full])
        else:
            gt_times = np.array([state[0] for state in gt_section])
            gt_positions = np.array([state[1:4] for state in gt_section])

        candidate_times = np.array([state[0] for state in candidate_trajectory])
        candidate_positions = np.array([state[1:4] for state in candidate_trajectory])

        # Interpolate the ground truth positions at the candidate timestamps.
        interp_east = interp1d(gt_times, gt_positions[:, 0], kind='linear', fill_value="extrapolate")
        interp_north = interp1d(gt_times, gt_positions[:, 1], kind='linear', fill_value="extrapolate")
        interp_alt = interp1d(gt_times, gt_positions[:, 2], kind='linear', fill_value="extrapolate")
        gt_interp_positions = np.stack([interp_east(candidate_times),
                                        interp_north(candidate_times),
                                        interp_alt(candidate_times)], axis=1)

        # Compute error metrics.
        position_errors = candidate_positions - gt_interp_positions
        euclidean_errors = np.linalg.norm(position_errors, axis=1)
        total_position_rmse = np.sqrt(np.mean(euclidean_errors**2))

        return {
            'total_position_rmse': total_position_rmse,
            'position_errors': position_errors,
            'euclidean_errors': euclidean_errors,
            'candidate_times': candidate_times,
            'candidate_positions': candidate_positions,
            'ground_truth_interp': gt_interp_positions,
            'gt_start_time': candidate_start,
            'gt_end_time': candidate_end
        }

    # def run_sampled_brute_force_kalman_filter(self, start_idx=0, end_idx=None, overwrite_sampling_freq=None, R_threshold=None, max_combos_in_memory=500):
    #     """
    #     Zombie-resistant brute force implementation.
    #     """
    #     from itertools import product, islice

    #     if R_threshold is None:
    #         raise ValueError("R_threshold must be specified for brute force KF.")
        
    #     # Set up signal handling to prevent zombies
    #     def signal_handler(signum, frame):
    #         print(f"\nReceived signal {signum}, cleaning up...")
    #         sys.exit(1)
        
    #     signal.signal(signal.SIGINT, signal_handler)
    #     signal.signal(signal.SIGTERM, signal_handler)
        
    #     class_args = {
    #         'get_state_transition_matrix': self.get_state_transition_matrix,
    #         'get_process_noise_covariance_matrix': self.get_process_noise_covariance_matrix,
    #         'predict_covariance': self.predict_covariance,
    #         'get_gps_observation_matrix': self.get_gps_observation_matrix,
    #         'get_gps_measurement_noise_covariance_matrix': self.get_gps_measurement_noise_covariance_matrix,
    #         'get_imu_observation_matrix': self.get_imu_observation_matrix,
    #         'get_imu_measurement_noise_covariance_matrix': self.get_imu_measurement_noise_covariance_matrix,
    #         'calculate_kalman_gain': self.calculate_kalman_gain,
    #         'calculate_accuracy_metrics': self.calculate_accuracy_metrics,
    #     }

    #     # Initialize state and covariance
    #     xt = np.zeros(15)
    #     Pt = np.diag([1000]*3 + [100]*3 + [100]*3 + [100]*3 + [1000]*3)

    #     if end_idx is None:
    #         end_idx = len(self.indexed_sensor_data)

    #     # Find first GPS measurement to initialize
    #     gps_started = False
    #     previous_time = None
    #     sensor_data_queues = []
    #     sensor_data_queue = []
        
    #     for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
    #         if not gps_started and sensor_type == 'GPS':
    #             xt[0] = sensor_data['easting']
    #             xt[1] = sensor_data['northing']  
    #             xt[2] = sensor_data['altitude']
    #             gps_started = True
    #             previous_time = time
    #         if not gps_started:
    #             continue
                
    #         processing_frequency = overwrite_sampling_freq if (overwrite_sampling_freq is not None) else self.processing_frequency
            
    #         # Collect measurements within each time window
    #         if time - previous_time < 1/processing_frequency:
    #             sensor_data_queue.append((index, sensor_type, time, sensor_data))
    #             continue
                
    #         if len(sensor_data_queue) == 0:
    #             sensor_data_queue.append((index, sensor_type, time, sensor_data))
                
    #         sensor_data_queues.append(sensor_data_queue.copy())
    #         sensor_data_queue = []
    #         previous_time = time

    #     # Step 2: Brute force all possible combinations (one sensor per queue)
    #     queue_lengths = [len(q) for q in sensor_data_queues]
    #     if any(l == 0 for l in queue_lengths):
    #         print("Warning: At least one queue is empty, no combinations possible.")
    #         return None

    #     total_combos = 1
    #     for l in queue_lengths:
    #         total_combos *= l
    #     print(f"Total combinations: {total_combos}")
            
    #     # Conservative settings to prevent zombie accumulation
    #     num_workers = 50  # Very conservative
    #     chunk_size = max(50, max_combos_in_memory // num_workers - 1)
        
    #     print(f"Using {num_workers} workers with chunk size {chunk_size}")
        
    #     best_metric = float('inf')
    #     best_trajectory = None
    #     best_sensors = None
    #     best_state = None
    #     best_cov = None
    #     best_log_det = None
    #     best_num_measurements_used = float('inf')
        
    #     def combo_generator():
    #         return product(*sensor_data_queues)
        
    #     # Use regular multiprocessing.Pool instead of ProcessPoolExecutor
    #     # This gives us better control over process lifecycle
    #     combos = combo_generator()
    #     processed = 0
        
    #     try:
    #         print("Starting multiprocessing pool...")
    #         with Pool(processes=num_workers) as pool:
    #             with tqdm(total=total_combos, desc="Brute force progress") as pbar:
    #                 while True:
    #                     chunk = list(islice(combos, chunk_size))
    #                     if not chunk:
    #                         break

    #                     # --- MODIFICATION START ---
    #                     # Filter combos based on the number of measurements used.
    #                     # This assumes a measurement is a placeholder if its stype is 'NONE'.
    #                     # You may need to adjust the placeholder value.
    #                     filtered_chunk = [
    #                         c for c in chunk 
    #                         if sum(1 for (_, stype, _, _) in c if stype != 'NONE') < best_num_measurements_used
    #                     ]

    #                     # If the entire chunk is filtered out, just update the progress and continue.
    #                     if not filtered_chunk:
    #                         pbar.update(len(chunk))
    #                         continue
    #                     # --- MODIFICATION END ---
                        
    #                     # Submit work and get results immediately (blocking)
    #                     # This prevents accumulation of zombie processes
    #                     try:
    #                         result = pool.apply_async(
    #                             evaluate_combo_chunk_worker,
    #                             (chunk, xt, Pt, class_args)
    #                         )
                            
    #                         # Get result with timeout to prevent hanging
    #                         chunk_results = result.get(timeout=300)  # 5 minute timeout
                            
    #                         for metric, traj, combo, xt_bf, Pt_bf, log_det, num_measurements_used in chunk_results:
    #                             # Ensuring we minimize sensor usage while also mantaining certain level of accurcay
    #                             if metric is not None and max(log_det) < R_threshold and num_measurements_used < best_num_measurements_used:
    #                                 best_metric = metric
    #                                 best_trajectory = traj
    #                                 best_sensors = combo
    #                                 best_state = xt_bf
    #                                 best_cov = Pt_bf
    #                                 best_log_det = log_det
    #                                 best_num_measurements_used = num_measurements_used
                            
    #                         processed += len(chunk)
    #                         pbar.update(len(chunk))
                            
    #                         # Force garbage collection after each chunk
    #                         import gc
    #                         gc.collect()
                            
    #                     except mp.TimeoutError:
    #                         print(f"\nChunk timed out, skipping...")
    #                         continue
    #                     except Exception as e:
    #                         print(f"\nError processing chunk: {e}")
    #                         continue
                
    #             print(f"\nProcessed {processed} combinations total")
                
    #         # Explicitly wait for all processes to terminate
    #         print("Waiting for all processes to terminate...")
            
    #     except KeyboardInterrupt:
    #         print("\nInterrupted by user, cleaning up...")
    #         return None
    #     except Exception as e:
    #         print(f"Error in multiprocessing: {e}")
    #         return None
    #     finally:
    #         # Additional cleanup
    #         import gc
    #         gc.collect()
        
    #     print("Brute force search complete.")
    #     return {
    #         'selected_sensors': best_sensors,
    #         'final_state': best_state,
    #         'final_covariance': best_cov,
    #         'trajectory': best_trajectory,
    #         'accuracy_metric': best_metric,
    #         'log_determinants': best_log_det,
    #         'num_measurements_used': best_num_measurements_used,
    #     }
    
    def run_brute_force_kalman_filter_no_sampling(self, start_idx=0, end_idx=None, R_threshold=None, initial_pt=None, initial_state=None, max_combos_in_memory=500):
        """
        Brute force implementation that processes all sensor measurements without sampling frequency constraints.
        Each sensor measurement becomes a separate choice in the combination space.
        """
        from itertools import combinations, islice

        if R_threshold is None:
            raise ValueError("R_threshold must be specified for brute force KF.")
        if start_idx is None or start_idx < 0:
            start_idx = 0
            print("Overriding start_idx to 0.")
        if end_idx is None or end_idx > len(self.indexed_sensor_data):
            end_idx = len(self.indexed_sensor_data)
            print("Overriding end_idx to the length of indexed_sensor_data.")
        if R_threshold is None:
            R_threshold = -inf
            print("Overriding R_threshold to - infinity (i.e. max information utilization).")

        # Set up signal handling to prevent zombies
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, cleaning up...")
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        class_args = {
            'get_state_transition_matrix': self.get_state_transition_matrix,
            'get_process_noise_covariance_matrix': self.get_process_noise_covariance_matrix,
            'predict_covariance': self.predict_covariance,
            'get_gps_observation_matrix': self.get_gps_observation_matrix,
            'get_gps_measurement_noise_covariance_matrix': self.get_gps_measurement_noise_covariance_matrix,
            'get_imu_observation_matrix': self.get_imu_observation_matrix,
            'get_imu_measurement_noise_covariance_matrix': self.get_imu_measurement_noise_covariance_matrix,
            'calculate_kalman_gain': self.calculate_kalman_gain,
            # 'calculate_accuracy_metrics': self.calculate_accuracy_metrics,
        }
        sensor_measurements = []

        # Initialize state and covariance
        xt = np.zeros(15)
        if initial_pt is not None and initial_state is not None:
            Pt = initial_pt
            xt[0:6] = initial_state[1:7]  # Set initial position and orientation
            prev_time = initial_state[0]
            start_idx_offset = start_idx 

            for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
                sensor_measurements.append((index, sensor_type, time, sensor_data))
        else:
            Pt = np.array([
                [10000, 0, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 10000, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 10000,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        1000, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 1000, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 1000,     0, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     1000, 0, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 1000, 0,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 1000,    0, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 0,    1000, 0, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 1000, 0,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 1000,    0, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      10000, 0, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 10000, 0],
                [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 0, 10000],
            ])
            

            if end_idx is None:
                end_idx = len(self.indexed_sensor_data)

            # Find first GPS measurement to initialize
            gps_started = False
            
            for i, (index, sensor_type, time, sensor_data) in enumerate(self.indexed_sensor_data[start_idx:end_idx]):
                if not gps_started and sensor_type == 'GPS':
                    xt[0] = sensor_data['easting']
                    xt[1] = sensor_data['northing']  
                    xt[2] = sensor_data['altitude']
                    gps_started = True
                    prev_time = time
                if not gps_started:
                    continue
                    
                # Add each sensor measurement as a potential choice
                sensor_measurements.append((index, sensor_type, time, sensor_data))

            if len(sensor_measurements) == 0:
                print("No sensor measurements found after GPS initialization.")
                return None

        last_time = sensor_measurements[-1][2]

        n_measurements = len(sensor_measurements)
        total_combos = 2 ** n_measurements
        print(f"Total combinations: {total_combos} (2^{n_measurements})")

        # Conservative settings to prevent zombie accumulation
        num_workers = 50
        chunk_size = max(50, max_combos_in_memory // num_workers - 1)
        
        print(f"Using {num_workers} workers with chunk size {chunk_size}")
        
        for k in range(1, n_measurements + 1):
            print(f"\nProcessing combinations with {k} measurements...")
            
            num_combos_k = math.comb(n_measurements, k)
            print(f"Total combinations of size {k}: {num_combos_k}")
            
            combos_k_iter = combinations(sensor_measurements, k)
            
            try:
                with Pool(processes=num_workers) as pool:
                    with tqdm(total=num_combos_k, desc=f"Size {k} combos") as pbar:
                        while True:
                            chunk = list(islice(combos_k_iter, chunk_size))
                            if not chunk:
                                break

                            try:
                                result = pool.apply_async(
                                    evaluate_combo_chunk_worker,
                                    (chunk, xt, Pt, class_args, prev_time, last_time)
                                )
                                
                                chunk_results = result.get(timeout=300)
                                
                                valid_results_in_chunk = []
                                for res in chunk_results:
                                    metric, traj, combo, xt_bf, Pt_bf, log_det, num_used = res
                                    if metric is not None and max(log_det) < R_threshold:
                                        valid_results_in_chunk.append(res)
                                
                                # If we found any valid results, find the best one in the chunk, return, and exit
                                if valid_results_in_chunk:
                                    best_in_chunk = min(valid_results_in_chunk, key=lambda x: x[0]) # Best by metric
                                    
                                    metric, traj, combo, xt_bf, Pt_bf, log_det, num_used = best_in_chunk
                                    
                                    print(f"\nFound valid combination with {k} measurements meeting R_threshold.")
                                    print("Brute force search complete.")
                                    return {
                                        'selected_sensors': combo,
                                        'final_state': xt_bf,
                                        'final_covariance': Pt_bf,
                                        'trajectory': traj,
                                        'accuracy_metric': metric,
                                        'log_determinants': log_det,
                                        'num_measurements_used': num_used,
                                    }
                                
                                pbar.update(len(chunk))
                                
                            except mp.TimeoutError:
                                print(f"\nChunk timed out, skipping...")
                                pbar.update(len(chunk))
                                continue
                            except Exception as e:
                                print(f"\nError processing chunk: {e}")
                                pbar.update(len(chunk))
                                continue

            except KeyboardInterrupt:
                print("\nInterrupted by user, cleaning up...")
                return None
            except Exception as e:
                print(f"Error in multiprocessing: {e}")
                return None

        print("\nBrute force search complete. No combination met the R_threshold.")
        return None
    
    def run_dead_reckoning_for_IMU(self):
        """Run dead reckoning to generate IMU estimates."""
        # Initial state and covariance

        # [x, y, z, roll, pitch, yaw, v_x, v_y, v_z, angular_x, angular_y, angular_z, a_x, a_y, a_z]
        xt = np.array([0, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0,    0, 0, 0])
        Pt = np.array([
            [10000, 0, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 10000, 0,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 10000,     0, 0, 0,       0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        1000, 0, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 1000, 0,     0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 1000,     0, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     1000, 0, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 1000, 0,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 1000,    0, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    1000, 0, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 1000, 0,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 1000,    0, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      10000, 0, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 10000, 0],
            [0, 0, 0,        0, 0, 0,     0, 0, 0,    0, 0, 0,      0, 0, 10000],
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

    def plot_covariance_evolution(self, sf_KF_state, sf_KF_covariance, save_path='covariance_evolution.png'):
        """
        Plot the evolution of covariance over time for the Kalman Filter.
        
        Args:
            sf_KF_state: List of state tuples (time, x, y, z, roll, pitch, yaw)
            sf_KF_covariance: List of 15x15 covariance matrices
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if len(sf_KF_state) != len(sf_KF_covariance):
            print("State and covariance lists must have the same length")
            return
        
        # Extract timestamps
        times = np.array([state[0] for state in sf_KF_state])
        times = (times - times[0]) / 60  # Convert to minutes from start
        
        # Extract diagonal elements (variances) for key states
        position_vars = np.array([[cov[0,0], cov[1,1], cov[2,2]] for cov in sf_KF_covariance])  # x, y, z variances
        orientation_vars = np.array([[cov[3,3], cov[4,4], cov[5,5]] for cov in sf_KF_covariance])  # roll, pitch, yaw variances
        velocity_vars = np.array([[cov[6,6], cov[7,7], cov[8,8]] for cov in sf_KF_covariance])  # vx, vy, vz variances
        
        # Convert variances to standard deviations for better interpretation
        position_stds = np.sqrt(position_vars)
        orientation_stds = np.sqrt(orientation_vars) * 180 / np.pi  # Convert to degrees
        velocity_stds = np.sqrt(velocity_vars)
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Kalman Filter Uncertainty Evolution Over Time', fontsize=16)
        
        # Position uncertainties
        axes[0,0].plot(times, position_stds[:, 0], label='X (East)', color='red', alpha=0.8)
        axes[0,0].plot(times, position_stds[:, 1], label='Y (North)', color='green', alpha=0.8)
        axes[0,0].plot(times, position_stds[:, 2], label='Z (Altitude)', color='blue', alpha=0.8)
        axes[0,0].set_ylabel('Position Std Dev (meters)')
        axes[0,0].set_title('Position Uncertainty')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_yscale('log')
        
        # Orientation uncertainties
        axes[0,1].plot(times, orientation_stds[:, 0], label='Roll', color='red', alpha=0.8)
        axes[0,1].plot(times, orientation_stds[:, 1], label='Pitch', color='green', alpha=0.8)
        axes[0,1].plot(times, orientation_stds[:, 2], label='Yaw', color='blue', alpha=0.8)
        axes[0,1].set_ylabel('Orientation Std Dev (degrees)')
        axes[0,1].set_title('Orientation Uncertainty')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_yscale('log')
        
        # Velocity uncertainties
        axes[1,0].plot(times, velocity_stds[:, 0], label='Vx', color='red', alpha=0.8)
        axes[1,0].plot(times, velocity_stds[:, 1], label='Vy', color='green', alpha=0.8)
        axes[1,0].plot(times, velocity_stds[:, 2], label='Vz', color='blue', alpha=0.8)
        axes[1,0].set_xlabel('Time (minutes)')
        axes[1,0].set_ylabel('Velocity Std Dev (m/s)')
        axes[1,0].set_title('Velocity Uncertainty')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_yscale('log')
        
        # Overall uncertainty metric (trace of position covariance)
        total_position_uncertainty = np.sqrt(position_vars.sum(axis=1))
        axes[1,1].plot(times, total_position_uncertainty, color='purple', linewidth=2)
        axes[1,1].set_xlabel('Time (minutes)')
        axes[1,1].set_ylabel('Total Position Uncertainty (m)')
        axes[1,1].set_title('Overall Position Uncertainty')
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_yscale('log')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Covariance evolution plot saved to {save_path}")
        # plt.show()

    def plot_covariance_heatmap(self, sf_KF_covariance, time_indices=None, save_path='covariance_heatmap.png'):
        """
        Plot heatmaps of covariance matrices at specific time points.
        
        Args:
            sf_KF_covariance: List of 15x15 covariance matrices
            time_indices: List of time indices to plot (default: start, middle, end)
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        if time_indices is None:
            n_cov = len(sf_KF_covariance)
            time_indices = [0, n_cov//2, n_cov-1]  # Start, middle, end
        
        # State variable labels
        state_labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 
                    'Vx', 'Vy', 'Vz', 'ωx', 'ωy', 'ωz', 'ax', 'ay', 'az']
        
        fig, axes = plt.subplots(1, len(time_indices), figsize=(5*len(time_indices), 4))
        if len(time_indices) == 1:
            axes = [axes]
        
        for i, time_idx in enumerate(time_indices):
            cov_matrix = sf_KF_covariance[time_idx]
            
            # Create correlation matrix for better visualization
            std_devs = np.sqrt(np.diag(cov_matrix))
            correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)
            
            im = axes[i].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            axes[i].set_title(f'Correlation Matrix at Step {time_idx}')
            axes[i].set_xticks(range(15))
            axes[i].set_yticks(range(15))
            axes[i].set_xticklabels(state_labels, rotation=45)
            axes[i].set_yticklabels(state_labels)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[i], shrink=0.8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Covariance heatmap saved to {save_path}")
        # plt.show()

    def plot_uncertainty_ellipses_2d(self, sf_KF_state, sf_KF_covariance, confidence_level=0.95, 
                                    skip_rate=10, save_path='uncertainty_ellipses.png'):
        """
        Plot 2D trajectory with uncertainty ellipses showing position confidence.
        
        Args:
            sf_KF_state: List of state tuples (time, x, y, z, roll, pitch, yaw)
            sf_KF_covariance: List of 15x15 covariance matrices
            confidence_level: Confidence level for ellipses (0.95 = 95%)
            skip_rate: Plot every skip_rate-th ellipse to avoid clutter
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import numpy as np
        from scipy.stats import chi2
        
        # Extract positions
        positions = np.array([[state[1], state[2]] for state in sf_KF_state])  # x, y positions
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, alpha=0.7, label='Estimated Trajectory')
        
        # Chi-square value for confidence level
        chi2_val = chi2.ppf(confidence_level, df=2)  # 2 DOF for x,y position
        
        # Plot uncertainty ellipses
        for i in range(0, len(sf_KF_covariance), skip_rate):
            # Extract 2x2 position covariance submatrix
            pos_cov = sf_KF_covariance[i][:2, :2]  # x,y covariance
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(pos_cov)
            
            # Compute ellipse parameters
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = 2 * np.sqrt(chi2_val * eigenvals[0])
            height = 2 * np.sqrt(chi2_val * eigenvals[1])
            
            # Create ellipse
            ellipse = Ellipse(positions[i], width, height, angle=angle, 
                            fill=False, edgecolor='red', alpha=0.5, linewidth=1)
            ax.add_patch(ellipse)
        
        # Add GPS reference data if available
        if hasattr(self, 'utm_data'):
            eastings = [data['easting'] for data in self.utm_data]
            northings = [data['northing'] for data in self.utm_data]
            ax.scatter(eastings, northings, c='green', marker='.', s=1, alpha=0.3, label='GPS Reference')
        
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.set_title(f'2D Trajectory with {confidence_level*100:.0f}% Confidence Ellipses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty ellipses plot saved to {save_path}")
        # plt.show()


def plot_brute_force_centered_comparison(utm_data, best_set, standard_kf_trajectory):
    # Extract standard KF trajectory and its center
    std_traj = standard_kf_trajectory
    std_X = [state[1] for state in std_traj]
    std_Y = [state[2] for state in std_traj]
    center_easting = (min(std_X) + max(std_X)) / 2
    center_northing = (min(std_Y) + max(std_Y)) / 2

    # Extract brute force KF trajectory (this will be the primary trajectory for centering)
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

    # Calculate bounds based primarily on the best set trajectory (brute force KF)
    min_x_bf, max_x_bf = min(bf_X), max(bf_X)
    min_y_bf, max_y_bf = min(bf_Y), max(bf_Y)
    
    # Add minimal padding (5% of the range) to maximize screen usage
    x_range = max_x_bf - min_x_bf
    y_range = max_y_bf - min_y_bf
    padding = 0.9  # 40% padding to maximize screen usage
    
    x_padding = x_range * padding
    y_padding = y_range * padding

    plt.figure(figsize=(10, 8))
    plt.plot(gps_X, gps_Y, label='GPS Ground Truth', color='black', alpha=0.5, linestyle='--')
    plt.plot(std_X, std_Y, label='Standard KF', color='orange', linewidth=2, linestyle='--')
    plt.plot(bf_X, bf_Y, label='Brute Force KF (Best Set)', color='blue', linewidth=1, linestyle='--')  # Thicker line for emphasis
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Brute Force KF Trajectory Centered (Best Set Optimized)')
    plt.legend()
    plt.grid(True)
    
    # Set limits to show entire trajectory with minimal padding, centered on best set
    plt.xlim(min_x_bf - x_padding, max_x_bf + x_padding)
    plt.ylim(min_y_bf - y_padding, max_y_bf + y_padding)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("brute_force_centered_trajectory_plot.png")

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

    # Calculate bounds for all trajectories
    all_X = std_X + bf_X + gps_X
    all_Y = std_Y + bf_Y + gps_Y
    
    min_x, max_x = min(all_X), max(all_X)
    min_y, max_y = min(all_Y), max(all_Y)
    
    # Add padding (10% of the range)
    x_range = max_x - min_x
    y_range = max_y - min_y
    padding = 0.9  # 90% padding
    
    x_padding = x_range * padding
    y_padding = y_range * padding

    plt.figure(figsize=(10, 8))
    plt.scatter(gps_X, gps_Y, label='GPS Reference', c='black', alpha=0.5, linestyle='-')
    plt.plot(bf_X, bf_Y, label='Brute Force KF', color='blue', linewidth=2)
    plt.plot(std_X, std_Y, label='Standard KF', color='orange', linewidth=2, linestyle='--')
    plt.xlabel('Easting')
    plt.ylabel('Northing')
    plt.title('Brute Force vs Standard KF Trajectory')
    plt.legend()
    plt.grid(True)
    
    # Set limits to show entire trajectory with padding
    plt.xlim(min_x - x_padding, max_x + x_padding)
    plt.ylim(min_y - y_padding, max_y + y_padding)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("kf_centered_trajectory_plot.png")

def plot_log_determinant(sf_KF_state, sf_KF_log_det, save_path='log_determinant_evolution.png'):
    """
    Plots the log determinant of the covariance matrix over time.

    Args:
        sf_KF_state (list): The list of state tuples containing timestamps.
        sf_KF_log_det (list): The list of log determinant values.
        save_path (str): The path to save the plot image.
    """
    if not sf_KF_state or not sf_KF_log_det:
        print("State or log determinant data is empty. Cannot plot.")
        return

    # Extract timestamps and make them relative to the start time
    times = np.array([state[0] for state in sf_KF_state])
    relative_times = times - times[0]

    # Create the plot
    plt.figure(figsize=(14, 7))
    plt.plot(relative_times, sf_KF_log_det, color='purple', linewidth=2)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log Determinant of Covariance (log det(P))')
    plt.title('Evolution of Covariance Uncertainty Over Time')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    # plt.ylim(-30, -21)
    
    # Save the plot
    plt.savefig(save_path, dpi=300)
    print(f"Log determinant plot saved to {save_path}")
    # plt.show()

def find_start_idx_for_time_offset(sensor_fusion, target_seconds):
    """Find the index where the trajectory reaches target_seconds from start."""
    if not hasattr(sensor_fusion, 'indexed_sensor_data') or not sensor_fusion.indexed_sensor_data:
        print("Sensor data not available. Run combine_sensor_data() first.")
        return None
    
    # Get the first timestamp as reference
    first_timestamp = 1697739552.3362827  # (idx, sensor_type, time, data)
    target_timestamp = first_timestamp + target_seconds
    
    # Search for the closest index
    for i, (index, sensor_type, time, sensor_data) in enumerate(sensor_fusion.indexed_sensor_data):
        if time >= target_timestamp:
            print(f"Found index {i} at time {time:.3f}s (offset: {time - first_timestamp:.3f}s)")
            return i
    
    print(f"Target time {target_seconds}s not found in data")
    return None

# Usage:

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
    # avg_roll, avg_pitch, avg_yaw = sensor_fusion.compute_stationary_orientation(first_valid_index)

    # Combine GPS and IMU data and sort by timestamp
    sensor_fusion.combine_sensor_data()

    # deadreckoned_IMU_estimates = sensor_fusion.run_dead_reckoning_for_IMU() # Output (Time Steps x 6) where stored values are (X, Y, Z, roll, pitch, yaw)


    ################# SETUP #################

    start_idx = find_start_idx_for_time_offset(sensor_fusion, 133.0)
    converging_buffer = 2000
    start_offset = 40
    sampling_frq = 20
    r_value = -25

    sensor_fusion.set_processing_frequency(sampling_frq)

    ################# MAX INFORMATION UTILIZATION FOR TIME SEGMENT [start_idx, start_idx + start_offset] #################

    # sf_KF_state_offset, _, pt_offset = sensor_fusion.run_kalman_filter_full(end_idx=start_idx) # Run full to get covariance convergence
    # sf_KF_state, sf_KF_log_det, sf_KF_pt = sensor_fusion.run_kalman_filter_full(start_idx=start_idx, end_idx=start_idx + start_offset, initial_pt=pt_offset, initial_state=sf_KF_state_offset[-1], print_output=True)
    
    # plot_log_determinant(sf_KF_state, sf_KF_log_det, save_path="Full_KF_log_determinant_evolution.png")

    ################# GREEDY INFORMATION SCHEDULING UTILIZATION ################# BROKEN
    # sensor_fusion.set_processing_frequency(sampling_frq)

    # sf_KF_state_offset, _, pt_offset = sensor_fusion.run_kalman_filter_scheduled(end_idx=start_idx, selection_method='greedy')
    # sf_KF_state, sf_KF_log_det, sf_KF_pt = sensor_fusion.run_kalman_filter_scheduled(start_idx=start_idx, end_idx=start_idx + start_offset, initial_pt=pt_offset, initial_state=sf_KF_state_offset[-1], selection_method='greedy', print_output=True)
    
    # plot_log_determinant(sf_KF_state, sf_KF_log_det, save_path="Greedy_Scheduling_KF_log_determinant_evolution.png")
 
    # ################# RANDOM INFORMATION SCHEDULING UTILIZATION ################# BROKEN
    # sensor_fusion.set_processing_frequency(sampling_frq)

    # sf_KF_state_offset, _, pt_offset = sensor_fusion.run_kalman_filter_scheduled(end_idx=start_idx, selection_method='random')
    # sf_KF_state, sf_KF_log_det, sf_KF_pt = sensor_fusion.run_kalman_filter_scheduled(start_idx=start_idx, end_idx=start_idx + start_offset, initial_pt=pt_offset, initial_state=sf_KF_state_offset[-1], selection_method='random', print_output=True)
    
    # plot_log_determinant(sf_KF_state, sf_KF_log_det, save_path="Random_Scheduling_KF_log_determinant_evolution.png")
 

    ################# ADAPTIVE INFORMATION UTILIZATION #################

    sf_KF_state_offset, _, pt_offset = sensor_fusion.run_adaptive_threshold_kalman_filter(end_idx=start_idx, R_threshold=r_value)
    sf_KF_state, sf_KF_log_det, sf_KF_pt = sensor_fusion.run_adaptive_threshold_kalman_filter(start_idx=start_idx, end_idx=start_idx + start_offset, R_threshold=r_value, initial_pt=pt_offset, initial_state=sf_KF_state_offset[-1], print_output=True)
    plot_log_determinant(sf_KF_state, sf_KF_log_det, save_path="Adaptive_KF_log_determinant_evolution.png")


    ################# BRUTE FORCE INFORMATION SCHEDULING UTILIZATION #################
    result = sensor_fusion.run_brute_force_kalman_filter_no_sampling(start_idx=start_idx, end_idx=start_idx + start_offset, initial_pt=pt_offset, initial_state=sf_KF_state_offset[-1], R_threshold=r_value)
    plot_log_determinant(result['trajectory'], result['log_determinants'], save_path="Brute_Force_KF_log_determinant_evolution.png")

    # Plotting
    # plot_kf_centered_comparison(sensor_fusion.get_utm_data(), best_set, sensor_fusion.get_GT())
    # plot_brute_force_centered_comparison(sensor_fusion.get_utm_data(), best_set, sensor_fusion.get_GT())



# kill $(pgrep -f "kf_workers.py")
# ps aux | grep -E "(zombie|<defunct>)" | grep iseanps 
# pkill -9 -u isean python3
# pkill -u isean python3
# python3 kf_workers.py
