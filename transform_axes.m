% 1. Load original LiDAR [time, x, y, z, qx, qy, qz, qw]
data = readmatrix('spoofed_traj.txt');
N = size(data, 1);

% 2. Define KITTI LiDAR-to-GT Frame Matrix

T = [ 0, -1,  0; 
    0,  0, -1; 
    1,  0,  0];

transformed_data = data;

for i = 1:N
    % 1. Position Transformation
    p_old = data(i, 2:4)'; 
    p_new = T * p_old;
    transformed_data(i, 2:4) = p_new';

    % 2. Quaternion Transformation
    qx = data(i, 5); qy = data(i, 6); qz = data(i, 7); qw = data(i, 8);

    % Reorder to MATLAB [w x y z]
    q_in = [qw, qx, qy, qz];
    R_old = quat2rotm(q_in);

    % Apply basis transformation: R_new = T * R_old * T'
    R_new = T * R_old * T';

    % Back to [w x y z]
    q_out = rotm2quat(R_new);

    % Reorder to your CSV [qx qy qz qw]
    transformed_data(i, 5:8) = [q_out(2), q_out(3), q_out(4), q_out(1)];
end

% 3. Save the new file
writematrix(transformed_data, 'spoofed_traj_transformed.csv');
fprintf('Success: Transformed %d rows into spoofed_traj_transformed.csv\n', N);


%% Plot Comparison: Ground Truth vs Original LiDAR vs Transformed LiDAR
% This script aligns the origins and compares the trajectories.

% 1. Load Ground Truth (KITTI 00.txt)
gt_data = load('00.txt');
N_gt = size(gt_data, 1);
t_gt = zeros(N_gt, 3);
for i = 1:N_gt
    % Extract the 3x4 transformation matrix
    T = reshape(gt_data(i,:), [4, 3])';
    % Extract the translation vector (last column)
    t_gt(i,:) = T(:, 4)';
end

% Zero out the Ground Truth Origin
t_gt = t_gt - t_gt(1, :);

% 2. Load LiDAR Data
% Format: [time, x, y, z, qx, qy, qz, qw]
original = readmatrix('spoofed_traj.txt');
transformed = readmatrix('spoofed_traj_transformed.csv');

% Extract coordinates
p_orig = original(:, 2:4);
p_trans = transformed(:, 2:4);

% 3. Create the 3D Plot
figure('Color', 'w', 'Name', 'KITTI Sequence 00 Alignment Check');
hold on;

% Plot Ground Truth (Black)
plot3(t_gt(:,1), t_gt(:,2), t_gt(:,3), 'g-', 'LineWidth', 2);

% Plot Original LiDAR (Blue)
plot3(p_orig(:,1), p_orig(:,2), p_orig(:,3), 'b--', 'LineWidth', 1);

% Plot Transformed LiDAR (Red)
plot3(p_trans(:,1), p_trans(:,2), p_trans(:,3), 'r-', 'LineWidth', 1.5);

% Formatting
grid on;
axis equal;
xlabel('X (meters)');
ylabel('Y (meters)');
zlabel('Z (meters)');
title('Trajectory Comparison: KITTI GT vs LiDAR');
legend('Ground Truth (KITTI 00)', 'Original LiDAR', 'Transformed', 'Location', 'best');
view(3);

% Mark the start point
scatter3(0, 0, 0, 100, 'k', 'filled', 'DisplayName', 'Start Point');

hold off;



%% 4. Calculate Error
% Determine the length to compare (shortest of the two)
num_points = min(size(t_gt, 1), size(p_trans, 1));

% Calculate the difference between aligned trajectories
% (Assumes both datasets are sampled at 10Hz starting at the same time)
errors = p_trans(1:num_points, :) - t_gt(1:num_points, :);

% 1. RMSE per axis
rmse_x = sqrt(mean(errors(:,1).^2));
rmse_y = sqrt(mean(errors(:,2).^2));
rmse_z = sqrt(mean(errors(:,3).^2));

% 2. Absolute Trajectory Error (ATE) - Euclidean distance error
ate = mean(sqrt(sum(errors.^2, 2)));

% 3. Print the results
fprintf('\n--- Trajectory Error Analysis ---\n');
fprintf('RMSE X: %.4f meters\n', rmse_x);
fprintf('RMSE Y: %.4f meters\n', rmse_y);
fprintf('RMSE Z: %.4f meters\n', rmse_z);
fprintf('Total ATE: %.4f meters\n', ate);
fprintf('---------------------------------\n');
