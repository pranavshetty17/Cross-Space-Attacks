%% ESKF using MATLAB Built-in insfilterErrorState with External GPS Data
% Required: Navigation Toolbox
%%%%%%% Simultaneous Lidar and GPS spoofing for ALOAM

%% 1. Constants and Noise Parameters
dt = 0.1;               % Time step (10 Hz)
imuRate = 1/dt;         % IMU Rate set to match dt (10 Hz)
gyroNoiseStd  = 0.01;     
accNoiseStd   = 2.0;      
gpsNoiseStd   = 1.0;   
lidarNoiseStd = 1.0;
g = [0; 0; -9.81]; % Gravity in World Frame

%% 2. Load KITTI Ground Truth (00.txt)
data = load('00.txt');   
N = size(data,1);
R_gt = zeros(3,3,N);
t_gt = zeros(N,3);
for i = 1:N
    T_mat = reshape(data(i,:), [4,3])';
    R_gt(:,:,i) = T_mat(:,1:3);
    t_gt(i,:)   = T_mat(:,4)';
end
% Shift GT to origin
gt_start = t_gt(1,:);
t_gt = t_gt - gt_start;

%% 3. Load External GPS Data from CSV
gps_filepath = '/home/matlab_R2026a_Linux/Scripts/Cross_Space_Attacks/new_traj_0_2.csv';
if ~exist(gps_filepath, 'file')
    error('GPS CSV file not found at: %s', gps_filepath);
end
gps_table = readtable(gps_filepath);

% FIX 1: Normalize GPS timestamps to start at 0.0 seconds
t_gps   = gps_table.Time_s - gps_table.Time_s(1);
gps_pos = [gps_table.GPS_X, gps_table.GPS_Y, gps_table.GPS_Z];

fprintf('Successfully loaded %d GPS samples from CSV.\n', height(gps_table));

%% 4. Prepare IMU Data (Derived from GT)
time = (0:N-1)' * dt;
vel_gt = [zeros(1,3); diff(t_gt)/dt];
acc_world = [zeros(1,3); diff(vel_gt)/dt];
acc_body = zeros(N,3);
gyro = zeros(N,3);
for i = 2:N
    acc_body(i,:) = R_gt(:,:,i)' * (acc_world(i,:)' - g);
    dR = R_gt(:,:,i-1)' * R_gt(:,:,i);
    omega_mat = (dR - dR') / (2*dt);
    gyro(i,:) = [omega_mat(3,2), omega_mat(1,3), omega_mat(2,1)];
end
gyro_meas = gyro + gyroNoiseStd * randn(size(gyro));
acc_meas  = acc_body + accNoiseStd * randn(size(acc_body));

%% 5. Load Aligned LiDAR Data
lidar = readmatrix('spoofed_traj_transformed.csv');

% FIX 2: Normalize LiDAR timestamps to start at 0.0 seconds
t_lidar = lidar(:,1) - lidar(1,1);
p_lidar = lidar(:,2:4);
q_lidar = lidar(:,5:8); % [qx qy qz qw]

%% 6. Initialize MATLAB Built-in ESKF
q0 = [q_lidar(1,4), q_lidar(1,1:3)]; % [qw qx qy qz]
p0 = p_lidar(1,:);                  % [x y z]
v0 = [0, 0, 0];                     % [vx vy vz]
bg0 = [0, 0, 0];                    % Gyro Bias
ba0 = [0, 0, 0];                    % Accel Bias

stateVec = [q0, p0, v0, bg0, ba0];
if length(stateVec) < 17
    stateVec(17) = 0; 
end

% FIX 3: Set IMUSampleRate explicitly to 1/dt (10 Hz)
filter = insfilterErrorState('State', stateVec, 'IMUSampleRate', imuRate);

% Noise parameters
filter.GyroscopeNoise     = gyroNoiseStd^2;
filter.AccelerometerNoise = accNoiseStd^2;
filter.GyroscopeBiasNoise = 1e-8;
filter.AccelerometerBiasNoise = 1e-8;
R_gps_val   = gpsNoiseStd^2;
R_lidar_val = lidarNoiseStd^2;

%% 7. Main Fusion Loop
est_p = zeros(N,3);
est_q = zeros(N,4);

% Pre-assign step k = 1
est_p(1,:) = p0;
est_q(1,:) = [q0(2), q0(3), q0(4), q0(1)]; % [qx, qy, qz, qw]

for k = 2:N
    % 1. Predict state using IMU
    predict(filter, acc_meas(k,:), gyro_meas(k,:));
    
    % 2. GPS Correction (Find nearest GPS measurement in time)
    [min_gps_dt, idx_gps] = min(abs(t_gps - time(k)));
    if min_gps_dt < (dt / 2) % Apply correction if timestamp matches step
        z_gps = gps_pos(idx_gps, :);
        correct(filter, 5:7, z_gps, R_gps_val * eye(3)); 
    end
    
    % 3. LiDAR Correction (Only update if timestamp matches step)
    [min_lidar_dt, idx_ts] = min(abs(t_lidar - time(k)));
    if min_lidar_dt < (dt / 2)
        z_lidar = p_lidar(idx_ts, :);
        correct(filter, 5:7, z_lidar, R_lidar_val * eye(3));
    end
    
    % 4. State Extraction
    current_state = filter.State;
    est_p(k,:) = current_state(5:7)'; 
    
    % Store orientation as [qx, qy, qz, qw]
    qw = current_state(1);
    qx = current_state(2);
    qy = current_state(3);
    qz = current_state(4);
    est_q(k,:) = [qx, qy, qz, qw]; 
end

%% 8. Trajectory Comparison Plot (Publication Ready)
fig1 = figure('Name', 'Built-in ESKF Result', 'Color', 'w');
fig1.Position = [50, 50, 1600, 1000];

plot3(t_gt(:,1), t_gt(:,2), t_gt(:,3), 'g-', 'LineWidth', 3.5); hold on;
plot3(est_p(:,1), est_p(:,2), est_p(:,3), 'c--', 'LineWidth', 4.0);
grid on; axis equal;
xlabel('X (m)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Y (m)', 'FontSize', 18, 'FontWeight', 'bold');
zlabel('Z (m)', 'FontSize', 18, 'FontWeight', 'bold');
title('Trajectory Comparison (MATLAB insfilterErrorState)', 'FontSize', 22, 'FontWeight', 'bold');

leg1 = legend({'Ground Truth', 'Built-in ESKF'}, 'Location', 'northeast', 'FontSize', 16, 'FontWeight', 'bold');
set(leg1, 'Box', 'on', 'Color', [1 1 1 0.85], 'EdgeColor', [0.3 0.3 0.3]);
set(gca, 'FontSize', 16, 'LineWidth', 2.0);

%% 9. Statistical Analysis
% Position Error
error_pos_mag = sqrt(sum((est_p - t_gt).^2, 2));
mean_pos = mean(error_pos_mag);
max_pos  = max(error_pos_mag);
sd_pos   = std(error_pos_mag);

% Orientation Error (Geodesic Distance)
error_ori_deg = zeros(N, 1);
for k = 1:N
    R_gt_k = R_gt(:,:,k);
    q_curr = est_q(k, :);
    q_m = [q_curr(4), q_curr(1), q_curr(2), q_curr(3)]; % [w, x, y, z]
    R_est_k = quat2rotm(q_m);
    
    R_diff = R_est_k' * R_gt_k;
    angle_rad = acos(max(-1, min(1, (trace(R_diff) - 1) / 2)));
    error_ori_deg(k) = rad2deg(angle_rad);
end
mean_ori = mean(error_ori_deg);
max_ori  = max(error_ori_deg);
sd_ori   = std(error_ori_deg);

%% Print Statistics
fprintf('\n========================================\n');
fprintf('       ESKF PERFORMANCE STATISTICS       \n');
fprintf('========================================\n');
fprintf('POSITION ERROR (meters):\n');
fprintf('  Mean: %.4f | Max: %.4f | SD: %.4f\n', mean_pos, max_pos, sd_pos);
fprintf('----------------------------------------\n');
fprintf('ORIENTATION ERROR (degrees):\n');
fprintf('  Mean: %.4f | Max: %.4f | SD: %.4f\n', mean_ori, max_ori, sd_ori);
fprintf('========================================\n');

%% 10. Error Histograms
fig2 = figure('Color', 'w', 'Name', 'Position Error Histogram');
fig2.Position = [100, 100, 1200, 800];
histogram(error_pos_mag, 50, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'w');
xline(mean_pos, 'r--', 'LineWidth', 3.0, 'Label', ['Mean: ' num2str(mean_pos, '%.2f') 'm'], ...
    'FontSize', 16, 'LabelHorizontalAlignment', 'right', 'FontWeight', 'bold');
grid on;
title('Position Error Magnitude Distribution', 'FontSize', 22, 'FontWeight', 'bold');
xlabel('Error (meters)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'FontSize', 16, 'LineWidth', 2.0);

fig3 = figure('Color', 'w', 'Name', 'Orientation Error Histogram');
fig3.Position = [150, 150, 1200, 800];
histogram(error_ori_deg, 50, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'w');
xline(mean_ori, 'k--', 'LineWidth', 3.0, 'Label', ['Mean: ' num2str(mean_ori, '%.2f') '°'], ...
    'FontSize', 16, 'LabelHorizontalAlignment', 'right', 'FontWeight', 'bold');
grid on;
title('Orientation Error Magnitude Distribution', 'FontSize', 22, 'FontWeight', 'bold');
xlabel('Angle Error (degrees)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 18, 'FontWeight', 'bold');
set(gca, 'FontSize', 16, 'LineWidth', 2.0);

%% 11. Sensor Fusion Overview (All Sensors Included - In-Plot Legend)
fig4 = figure('Color', 'w', 'Name', 'Comprehensive Sensor Comparison');
fig4.Position = [50, 50, 1800, 1100];

plot3(t_gt(:,1), t_gt(:,2), t_gt(:,3), 'g-', 'LineWidth', 4.0, 'DisplayName', 'Ground Truth'); 
hold on;
scatter3(gps_pos(:,1), gps_pos(:,2), gps_pos(:,3), 45, [1 0.5 0], 'filled', ...
    'MarkerFaceAlpha', 0.6, 'DisplayName', 'GPS');
plot3(p_lidar(:,1), p_lidar(:,2), p_lidar(:,3), 'r-', 'LineWidth', 2.5, 'DisplayName', 'LiDAR');
plot3(est_p(:,1), est_p(:,2), est_p(:,3), 'c--', 'LineWidth', 4.5, 'DisplayName', 'ESKF Fusion');

grid on; axis equal; view([-45, 30]); 
xlabel('X (meters)', 'FontSize', 18, 'FontWeight', 'bold');
ylabel('Y (meters)', 'FontSize', 18, 'FontWeight', 'bold');
zlabel('Z (meters)', 'FontSize', 18, 'FontWeight', 'bold');
title('Sensor Fusion Overview: GT vs. GPS vs. LiDAR vs. ESKF', 'FontSize', 22, 'FontWeight', 'bold');

leg4 = legend('Location', 'northeast', 'FontSize', 16, 'FontWeight', 'bold');
set(leg4, 'Box', 'on', 'Color', [1 1 1 0.85], 'EdgeColor', [0.3 0.3 0.3]);
set(gca, 'FontSize', 16, 'LineWidth', 2.0);

