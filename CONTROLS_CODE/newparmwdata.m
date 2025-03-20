% 2DOF System Parameter Estimation - Multiple Trials (with time shifting and combined plots)
% This script analyzes data from three experiments (with 10 trials each) to determine the
% six parameters of the 2DOF system (m1, m2, k1, k2, d1, d2)
clear all; clc; close all;
format long; % For high precision output

% Number of trials to process
numTrials = 10;

% Initialize arrays to store parameters from each trial
k1_values = zeros(numTrials, 1);
k2_values = zeros(numTrials, 1);
m1_values = zeros(numTrials, 1);
m2_values = zeros(numTrials, 1);
d1_values = zeros(numTrials, 1);
d2_values = zeros(numTrials, 1);

% Set step input amplitude (adjust as needed)
U = 0.5; % Volts - input step amplitude

% Time shift value (seconds)
timeShift = 4.5;

% Loop through all trials
for trial = 1:numTrials
    fprintf('\n============= PROCESSING TRIAL %d =============\n', trial);
    
    %% Experiment 1: 1DOF System with first cart only (encoder 1)
    fprintf('\n=== Experiment 1, Trial %d (First Cart Only) ===\n', trial);
    
    % Load the data with new format x1.y.mat
    filename = sprintf('x1.%d.mat', trial);
    data1 = load(filename);
    time1 = data1.time;
    enc1_exp1 = data1.enc1;

    % Filter data if needed
    timeLimit1 = 8; % seconds
    timeIdx1 = time1 <= timeLimit1;
    filteredTime1 = time1(timeIdx1);
    filteredEnc1_exp1 = enc1_exp1(timeIdx1);

    % Time shift for plotting - subtract timeShift from all time values
    shiftedTime1 = filteredTime1 - timeShift;

    % Find peaks for oscillation analysis
    [peaks1, peakLocations1] = findpeaks(filteredEnc1_exp1, 'MinPeakHeight', max(filteredEnc1_exp1)*0.5);
    peakTimes1 = filteredTime1(peakLocations1);
    shiftedPeakTimes1 = peakTimes1 - timeShift; % Apply time shift to peak times

    % Extract key parameters
    t0_exp1 = peakTimes1(1); % Time of first peak (unshifted for calculations)
    y0_exp1 = peaks1(1);     % Value of first peak
    n1 = 2; % Number of oscillations to analyze

    if length(peakTimes1) > n1
        tn_exp1 = peakTimes1(n1+1); % Time of (n1+1)th peak (unshifted for calculations)
        yn_exp1 = peaks1(n1+1);     % Value of (n1+1)th peak
    else
        error('Experiment 1, Trial %d: Not enough peaks detected.', trial);
    end

    % Estimate steady-state value
    steadyStateIdx1 = filteredTime1 > (max(filteredTime1) - 1); % Last second of data
    y_inf_exp1_raw = mean(filteredEnc1_exp1(steadyStateIdx1));
    y_inf_exp1 = round(y_inf_exp1_raw); % Round to nearest integer for encoder count

    % Store data for combined plotting (only for first trial)
    if trial == 1
        % Save experiment 1 data for combined plot
        exp1_time = shiftedTime1;
        exp1_data = filteredEnc1_exp1;
        exp1_peaks_time = shiftedPeakTimes1;
        exp1_peaks = peaks1;
        exp1_t0 = t0_exp1 - timeShift;
        exp1_y0 = y0_exp1;
        exp1_tn = tn_exp1 - timeShift;
        exp1_yn = yn_exp1;
        exp1_ss = y_inf_exp1;
    end

    % Calculate system parameters for experiment 1
    omega_d_exp1 = 2*pi*n1/(tn_exp1 - t0_exp1);
    beta_omega_n_exp1 = (1/(tn_exp1 - t0_exp1))*log((y0_exp1 - y_inf_exp1)/(yn_exp1 - y_inf_exp1));
    omega_n_exp1 = sqrt(omega_d_exp1^2 + beta_omega_n_exp1^2);
    beta_exp1 = beta_omega_n_exp1/omega_n_exp1;

    % Calculate model parameters (combined k1+k2)
    k_combined = U/y_inf_exp1;     % Combined stiffness (k1+k2) in V/count
    m1 = k_combined/(omega_n_exp1^2); % Mass m1 in V·s²/count
    d1 = k_combined*2*beta_exp1/omega_n_exp1; % Damping d1 in V·s/count

    % Store m1 and d1 values for this trial
    m1_values(trial) = m1;
    d1_values(trial) = d1;

    % Display experiment 1 results
    fprintf('Identified points:\n');
    fprintf('t0 = %.4f s, tn = %.4f s\n', t0_exp1, tn_exp1);
    fprintf('y0 = %.4f counts, yn = %.4f counts\n', y0_exp1, yn_exp1);
    fprintf('Steady state (raw mean) = %.4f counts\n', y_inf_exp1_raw);
    fprintf('Steady state (rounded) = %d counts\n', y_inf_exp1);
    fprintf('Calculated parameters:\n');
    fprintf('Damped frequency (ω_d) = %.4f rad/s\n', omega_d_exp1);
    fprintf('Decay term (βω_n) = %.4f\n', beta_omega_n_exp1);
    fprintf('Natural frequency (ω_n) = %.4f rad/s\n', omega_n_exp1);
    fprintf('Damping ratio (β) = %.4f\n', beta_exp1);
    fprintf('Combined stiffness (k1+k2) = %.6f V/count\n', k_combined);
    fprintf('Mass m1 = %.6f V·s²/count\n', m1);
    fprintf('Damping d1 = %.6f V·s/count\n\n', d1);

    %% Experiment 2: Full 2DOF System Analysis for k1 and k2
    fprintf('=== Experiment 2, Trial %d (Full 2DOF System) ===\n', trial);
    
    % Load the data with new format x2.y.mat
    filename = sprintf('x2.%d.mat', trial);
    data2 = load(filename);
    time2 = data2.time;
    enc1_exp2 = data2.enc1;
    enc2_exp2 = data2.enc2;

    % Filter data if needed
    timeLimit2 = 8; % seconds
    timeIdx2 = time2 <= timeLimit2;
    filteredTime2 = time2(timeIdx2);
    filteredEnc1_exp2 = enc1_exp2(timeIdx2);
    filteredEnc2_exp2 = enc2_exp2(timeIdx2);

    % Time shift for plotting - subtract timeShift from all time values
    shiftedTime2 = filteredTime2 - timeShift;

    % Calculate steady state values
    steadyStateIdx2 = filteredTime2 > (max(filteredTime2) - 1); % Last second of data
    ss_enc1_exp2 = mean(filteredEnc1_exp2(steadyStateIdx2));
    ss_enc1_exp2_rounded = round(ss_enc1_exp2); % Round to nearest integer
    ss_enc2_exp2 = mean(filteredEnc2_exp2(steadyStateIdx2));
    ss_enc2_exp2_rounded = round(ss_enc2_exp2); % Round to nearest integer

    % Store data for combined plotting (only for first trial)
    if trial == 1
        % Save experiment 2 data for combined plot
        exp2_time = shiftedTime2;
        exp2_data1 = filteredEnc1_exp2;
        exp2_data2 = filteredEnc2_exp2;
        exp2_ss1 = ss_enc1_exp2;
        exp2_ss2 = ss_enc2_exp2;
    end

    % Calculate individual spring constants using steady state analysis
    % For a 2DOF system at steady state with force F applied to first cart:
    % k1*x1 + k2*(x1-x2) = F  (for cart 1)
    % k2*(x2-x1) = 0       (for cart 2)
    % At steady state, both carts have the same position, so: x1 = x2
    % Therefore: k1*x1 = F or k1 = F/x1
    k1 = U/ss_enc1_exp2_rounded; % V/count

    % Since we know combined stiffness (k1+k2) from experiment 1,
    % we can calculate k2:
    k2 = k_combined - k1; % V/count

    % Store k1 and k2 values for this trial
    k1_values(trial) = k1;
    k2_values(trial) = k2;

    % Display experiment 2 results
    fprintf('Steady state values:\n');
    fprintf('Encoder 1 steady state (raw) = %.4f counts\n', ss_enc1_exp2);
    fprintf('Encoder 1 steady state (rounded) = %d counts\n', ss_enc1_exp2_rounded);
    fprintf('Encoder 2 steady state (raw) = %.4f counts\n', ss_enc2_exp2);
    fprintf('Encoder 2 steady state (rounded) = %d counts\n', ss_enc2_exp2_rounded);
    fprintf('Individual spring constants:\n');
    fprintf('k1 = %.6f V/count\n', k1);
    fprintf('k2 = %.6f V/count\n\n', k2);

    %% Experiment 3: 1DOF System with second cart only (encoder 2)
    fprintf('=== Experiment 3, Trial %d (Second Cart Only) ===\n', trial);
    
    % Load the data with new format x3.y.mat
    filename = sprintf('x3.%d.mat', trial);
    data3 = load(filename);
    time3 = data3.time;
    enc2_exp3 = data3.enc2;

    % Filter data if needed
    timeLimit3 = 8; % seconds
    timeIdx3 = time3 <= timeLimit3;
    filteredTime3 = time3(timeIdx3);
    filteredEnc2_exp3 = enc2_exp3(timeIdx3);

    % Find peaks for oscillation analysis - IMPROVED PEAK DETECTION
    [peaks3, peakLocations3] = findpeaks(filteredEnc2_exp3, 'MinPeakHeight', max(filteredEnc2_exp3)*0.2, 'MinPeakDistance', 10);
    peakTimes3 = filteredTime3(peakLocations3);
    
    % Display number of peaks found for debugging
    fprintf('Number of peaks found in Experiment 3, Trial %d: %d\n', trial, length(peaks3));

    % Extract key parameters
    t0_exp3 = peakTimes3(1); % Time of first peak
    y0_exp3 = peaks3(1);     % Value of first peak
    n3 = 4; % Number of oscillations to analyze - increased to 4 for better accuracy

    if length(peakTimes3) > n3
        tn_exp3 = peakTimes3(n3+1); % Time of (n3+1)th peak
        yn_exp3 = peaks3(n3+1);     % Value of (n3+1)th peak
    else
        error('Experiment 3, Trial %d: Not enough peaks detected. Only %d peaks found, need at least %d.', trial, length(peakTimes3), n3+1);
    end

    % Estimate steady-state value
    steadyStateIdx3 = filteredTime3 > (max(filteredTime3) - 1); % Last second of data
    y_inf_exp3_raw = mean(filteredEnc2_exp3(steadyStateIdx3));
    y_inf_exp3 = round(y_inf_exp3_raw); % Round to nearest integer for encoder count

    % Store data for combined plotting (only for first trial)
    if trial == 1
        % Save experiment 3 data for combined plot
        exp3_time = filteredTime3;
        exp3_data = filteredEnc2_exp3;
        exp3_peaks_time = peakTimes3;
        exp3_peaks = peaks3;
        exp3_t0 = t0_exp3;
        exp3_y0 = y0_exp3;
        exp3_tn = tn_exp3;
        exp3_yn = yn_exp3;
        exp3_ss = y_inf_exp3;
    end

    % Calculate system parameters for experiment 3
    omega_d_exp3 = 2*pi*n3/(tn_exp3 - t0_exp3);
    beta_omega_n_exp3 = (1/(tn_exp3 - t0_exp3))*log((y0_exp3 - y_inf_exp3)/(yn_exp3 - y_inf_exp3));
    omega_n_exp3 = sqrt(omega_d_exp3^2 + beta_omega_n_exp3^2);
    beta_exp3 = beta_omega_n_exp3/omega_n_exp3;

    % Calculate model parameters (second cart)
    m2 = k2/(omega_n_exp3^2);       % Mass m2 in V·s²/count
    d2 = k2*2*beta_exp3/omega_n_exp3; % Damping d2 in V·s/count

    % Store m2 and d2 values for this trial
    m2_values(trial) = m2;
    d2_values(trial) = d2;

    % Display experiment 3 results
    fprintf('Identified points:\n');
    fprintf('t0 = %.4f s, tn = %.4f s\n', t0_exp3, tn_exp3);
    fprintf('y0 = %.4f counts, yn = %.4f counts\n', y0_exp3, yn_exp3);
    fprintf('Steady state (raw mean) = %.4f counts\n', y_inf_exp3_raw);
    fprintf('Steady state (rounded) = %d counts\n', y_inf_exp3);
    fprintf('Calculated parameters:\n');
    fprintf('Damped frequency (ω_d) = %.4f rad/s\n', omega_d_exp3);
    fprintf('Decay term (βω_n) = %.4f\n', beta_omega_n_exp3);
    fprintf('Natural frequency (ω_n) = %.4f rad/s\n', omega_n_exp3);
    fprintf('Damping ratio (β) = %.4f\n', beta_exp3);
    fprintf('Mass m2 = %.6f V·s²/count\n', m2);
    fprintf('Damping d2 = %.6f V·s/count\n\n', d2);
end

%% Create combined plot of experiments 1, 2, and 3 side by side (after first trial is complete)
figure('Position', [100, 100, 1200, 400]);

% Experiment 1: First subplot
subplot(1, 3, 1);
% Find indices for 0-3s time range
timeIdx1_range = exp1_time >= 0 & exp1_time <= 3;
plot(exp1_time(timeIdx1_range), exp1_data(timeIdx1_range), 'b');
hold on;
% Add only peaks that fall within the 0-3s range
peakInRange = exp1_peaks_time >= 0 & exp1_peaks_time <= 3;
plot(exp1_peaks_time(peakInRange), exp1_peaks(peakInRange), 'ro', 'MarkerSize', 6);
% Add key points if they fall within range
if exp1_t0 >= 0 && exp1_t0 <= 3
    plot(exp1_t0, exp1_y0, 'gs', 'MarkerSize', 8, 'LineWidth', 1.5);
end
if exp1_tn >= 0 && exp1_tn <= 3
    plot(exp1_tn, exp1_yn, 'bs', 'MarkerSize', 8, 'LineWidth', 1.5);
end
yline(exp1_ss, 'k--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Encoder 1 Reading (counts)');
title('Experiment 1: First Cart Only');
grid on;
xlim([0 3]);
ylim auto;

% Experiment 2: Second subplot
subplot(1, 3, 2);
% Find indices for 0-3s time range
timeIdx2_range = exp2_time >= 0 & exp2_time <= 3;
plot(exp2_time(timeIdx2_range), exp2_data1(timeIdx2_range), 'b');
hold on;
plot(exp2_time(timeIdx2_range), exp2_data2(timeIdx2_range), 'r');
yline(exp2_ss1, 'b--', 'LineWidth', 1.5);
yline(exp2_ss2, 'r--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Encoder Reading (counts)');
title('Experiment 2: Both Carts');
legend('Cart 1', 'Cart 2', 'Location', 'best');
grid on;
xlim([0 3]);
ylim auto;

% Experiment 3: Third subplot
subplot(1, 3, 3);
plot(exp3_time, exp3_data, 'r');
hold on;
plot(exp3_peaks_time, exp3_peaks, 'ro', 'MarkerSize', 6);
plot(exp3_t0, exp3_y0, 'gs', 'MarkerSize', 8, 'LineWidth', 1.5);
plot(exp3_tn, exp3_yn, 'bs', 'MarkerSize', 8, 'LineWidth', 1.5);
yline(exp3_ss, 'k--', 'LineWidth', 1.5);
xlabel('Time (s)');
ylabel('Encoder 2 Reading (counts)');
title('Experiment 3: Second Cart Only');
grid on;
xlim auto;
ylim auto;

sgtitle('2DOF System Experimental Results');

%% Calculate average parameter values
k1_avg = mean(k1_values);
k2_avg = mean(k2_values);
m1_avg = mean(m1_values);
m2_avg = mean(m2_values);
d1_avg = mean(d1_values);
d2_avg = mean(d2_values);

% Calculate standard deviations
k1_std = std(k1_values);
k2_std = std(k2_values);
m1_std = std(m1_values);
m2_std = std(m2_values);
d1_std = std(d1_values);
d2_std = std(d2_values);

%% Display summary of all trials and average values
fprintf('\n======= 2DOF SYSTEM PARAMETER SUMMARY =======\n');
fprintf('Trial\t   k1\t\t   k2\t\t   m1\t\t   m2\t\t   d1\t\t   d2\n');
fprintf('---------------------------------------------------------------------\n');

for trial = 1:numTrials
    fprintf('%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
        trial, k1_values(trial), k2_values(trial), m1_values(trial), ...
        m2_values(trial), d1_values(trial), d2_values(trial));
end

fprintf('---------------------------------------------------------------------\n');
fprintf('Avg\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
    k1_avg, k2_avg, m1_avg, m2_avg, d1_avg, d2_avg);
fprintf('Std\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n', ...
    k1_std, k2_std, m1_std, m2_std, d1_std, d2_std);
fprintf('=========================================\n\n');

% Create a table with the parameter values for each trial
trialTable = table((1:numTrials)', k1_values, k2_values, m1_values, m2_values, d1_values, d2_values, ...
    'VariableNames', {'Trial', 'k1 (V/count)', 'k2 (V/count)', ...
    'm1 (V·s²/count)', 'm2 (V·s²/count)', 'd1 (V·s/count)', 'd2 (V·s/count)'});
disp(trialTable);

% Create a table with the final average parameter values
avgTable = table(...
    [m1_avg; m2_avg], [d1_avg; d2_avg], [k1_avg; k2_avg], ...
    [m1_std; m2_std], [d1_std; d2_std], [k1_std; k2_std], ...
    'VariableNames', {'Mass (V·s²/count)', 'Damping (V·s/count)', 'Stiffness (V/count)', ...
                      'Mass StdDev', 'Damping StdDev', 'Stiffness StdDev'}, ...
    'RowNames', {'Cart 1', 'Cart 2'});

disp('Average Parameter Values with Standard Deviations:');
disp(avgTable);

% Create bar plots to visualize parameter variations across trials
figure;
subplot(3,2,1);
bar(1:numTrials, k1_values);
title('k1 Values Across Trials');
xlabel('Trial');
ylabel('k1 (V/count)');
grid on;

subplot(3,2,2);
bar(1:numTrials, k2_values);
title('k2 Values Across Trials');
xlabel('Trial');
ylabel('k2 (V/count)');
grid on;

subplot(3,2,3);
bar(1:numTrials, m1_values);
title('m1 Values Across Trials');
xlabel('Trial');
ylabel('m1 (V·s²/count)');
grid on;

subplot(3,2,4);
bar(1:numTrials, m2_values);
title('m2 Values Across Trials');
xlabel('Trial');
ylabel('m2 (V·s²/count)');
grid on;

subplot(3,2,5);
bar(1:numTrials, d1_values);
title('d1 Values Across Trials');
xlabel('Trial');
ylabel('d1 (V·s/count)');
grid on;

subplot(3,2,6);
bar(1:numTrials, d2_values);
title('d2 Values Across Trials');
xlabel('Trial');
ylabel('d2 (V·s/count)');
grid on;

sgtitle('Parameter Variation Across Trials');

% Save the average parameters to a .mat file for future use
save('identified_parameters_avg.mat', 'k1_avg', 'k2_avg', 'm1_avg', 'm2_avg', 'd1_avg', 'd2_avg', ...
    'k1_std', 'k2_std', 'm1_std', 'm2_std', 'd1_std', 'd2_std', ...
    'k1_values', 'k2_values', 'm1_values', 'm2_values', 'd1_values', 'd2_values');

% Generate parameters.m file for use with maelab.m
parameters_file = fopen('parameters33.m', 'w');
fprintf(parameters_file, '%% Parameters for 2DOF system (generated automatically)\n');
fprintf(parameters_file, 'm1 = %.8f; %% Cart 1 mass\n', m1_avg);
fprintf(parameters_file, 'm2 = %.8f; %% Cart 2 mass\n', m2_avg);
fprintf(parameters_file, 'k1 = %.8f; %% Spring 1 stiffness\n', k1_avg);
fprintf(parameters_file, 'k2 = %.8f; %% Spring 2 stiffness\n', k2_avg);
fprintf(parameters_file, 'd1 = %.8f; %% Damping 1\n', d1_avg);
fprintf(parameters_file, 'd2 = %.8f; %% Damping 2\n', d2_avg);
fclose(parameters_file);

fprintf('\nParameters file "parameters.m" has been created for use with maelab.m\n');