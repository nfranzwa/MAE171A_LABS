% Parameter Comparison Plot
% This script compares experimental data with model predictions
% using both optimized and experimentally determined parameters

clear all; clc; close all;

%% Load experimental data
try
    data = load('x2.1.mat');
    fprintf('Successfully loaded x2.1.mat file\n');
    
    % Trim data to start at 5 seconds (as in the optimization script)
    trim_start_time = 5;
    exp_time_original = data.time;  
    exp_enc1_original = data.enc1;
    
    % Trim data
    trim_idx = exp_time_original >= trim_start_time;
    exp_time_trimmed = exp_time_original(trim_idx) - trim_start_time; % Shift time to start at 0
    exp_enc1_trimmed = exp_enc1_original(trim_idx);
    
    % Further trim data to only include points up to max_plot_time for plotting
    max_plot_time = 4;
    plot_idx = exp_time_trimmed <= max_plot_time;
    exp_time_plot = exp_time_trimmed(plot_idx);
    exp_enc1_plot = exp_enc1_trimmed(plot_idx);
catch ME
    error('Error loading data file: %s', ME.message);
end

%% Define parameters

% Optimized parameters from the code
optimized_params = [
    9.98789139168119e-07,  % m1
    2.10661018510654e-06,  % d1
    1.87882727967939e-04,  % k1
    8.94530018250171e-07,  % m2
    4.44901886590331e-06,  % d2
    1.58744321567414e-04   % k2
];

% Experimentally determined parameters from the table
experimental_params = [
    7.78584175929454e-07,  % m1
    3.58010235562671e-06,  % d1
    0.0001834029533006,    % k1
    8.75651342142609e-07,  % m2
    8.09020038557544e-07,  % d2
    0.0001640739906126     % k2
];

%% Configuration for simulation
step_size = 0.5;          % Step input amplitude in Volts
dwell_time = 5000;        % Dwell time in msec
encoderout = 1;           % Using encoder 1 
DOFs = 2;                 % Using 2DOF system
signofsystem = 1;         % Sign of encoder

%% Simulate models

% Function to simulate model
function [sim_time, sim_enc1, sim_enc2] = simulate_model(params, step_amplitude, dwell_time, encoderout, DOFs, signofsystem)
    % Extract parameters
    m1 = params(1);
    d1 = params(2);
    k1 = params(3);
    m2 = params(4);
    d2 = params(5);
    k2 = params(6);
    
    % Create the time vector just like MAELAB
    sim_dwell_time_sec = dwell_time * 1e-3; % Convert dwell time to seconds
    sim_time = linspace(0, 2*sim_dwell_time_sec, 900)';
    
    % Create step input just like MAELAB (first half up, second half down)
    ustep = [step_amplitude*ones(450,1); zeros(450,1)];
    
    % Build models based on DOFs setting
    if DOFs == 2
        % Full 2DOF model
        deng0 = [m1*m2, (m1*d2+m2*d1), (k2*m1+(k1+k2)*m2+d1*d2), ((k1+k2)*d2+k2*d1), k1*k2];
        
        % Create transfer functions for both encoders
        num_enc1 = signofsystem * [m2, d2, k2];  % Encoder 1 (x1)
        num_enc2 = signofsystem * [k2];          % Encoder 2 (x2)
    else
        % 1DOF model
        if encoderout == 1
            deng0 = [m1, d1, k1];
            num_enc1 = signofsystem * [1];  % Encoder 1 output
            num_enc2 = zeros(size(num_enc1)); % Not used
        else
            deng0 = [m2, d2, k2];
            num_enc1 = zeros(1); % Not used
            num_enc2 = signofsystem * [1];  % Encoder 2 output
        end
    end
    
    % Add DC motor dynamics (actuator) - MAELAB uses 1/209
    motor_time_constant = 1/209;
    deng = conv(deng0, [motor_time_constant, 1]);
    
    % Simulate both encoders
    sim_enc1 = lsim(num_enc1, deng, ustep, sim_time);
    sim_enc2 = lsim(num_enc2, deng, ustep, sim_time);
end

% Simulate with optimized parameters
[sim_time_opt, sim_enc1_opt, ~] = simulate_model(optimized_params, step_size, dwell_time, encoderout, DOFs, signofsystem);

% Simulate with experimentally determined parameters
[sim_time_exp, sim_enc1_exp, ~] = simulate_model(experimental_params, step_size, dwell_time, encoderout, DOFs, signofsystem);

%% Create comparison plot
figure('Name', 'Encoder 1 Response Comparison', 'Position', [100, 100, 900, 600]);

% Plot experimental data
plot(exp_time_plot, exp_enc1_plot, 'b-', 'LineWidth', 2);
hold on;

% Plot optimized model response
plot(sim_time_opt, sim_enc1_opt, 'r--', 'LineWidth', 1.5);

% Plot experimentally determined model response
plot(sim_time_exp, sim_enc1_exp, 'g-.', 'LineWidth', 1.5);

% Add legend and labels
legend('Experimental Data', 'Optimized Parameters', 'Experimentally Determined Parameters');
title('Comparison of Encoder 1 Responses');
xlabel('Time (s)');
ylabel('Encoder 1 Reading (counts)');
xlim([0, max_plot_time]);
grid on;

% % Add a text box with the parameters
% annotation('textbox', [0.15, 0.7, 0.25, 0.2], ...
%            'String', {'Parameters:', ...
%                       sprintf('Opt. m1: %.3e', optimized_params(1)), ...
%                       sprintf('Opt. d1: %.3e', optimized_params(2)), ...
%                       sprintf('Opt. k1: %.3e', optimized_params(3)), ...
%                       sprintf('Opt. m2: %.3e', optimized_params(4)), ...
%                       sprintf('Opt. d2: %.3e', optimized_params(5)), ...
%                       sprintf('Opt. k2: %.3e', optimized_params(6))}, ...
%            'FitBoxToText', 'on', ...
%            'BackgroundColor', [1 1 1 0.7]);

% % Calculate error metrics
% opt_enc1_interp = interp1(sim_time_opt, sim_enc1_opt, exp_time_plot, 'linear', 'extrap');
% exp_param_enc1_interp = interp1(sim_time_exp, sim_enc1_exp, exp_time_plot, 'linear', 'extrap');
% 
% % MSE calculation
% opt_mse = mean((opt_enc1_interp - exp_enc1_plot).^2);
% exp_param_mse = mean((exp_param_enc1_interp - exp_enc1_plot).^2);
% 
% % Add a text box with error metrics
% annotation('textbox', [0.65, 0.8, 0.25, 0.1], ...
%            'String', {'Error Metrics (MSE):', ...
%                       sprintf('Optimized: %.2f', opt_mse), ...
%                       sprintf('Experimental: %.2f', exp_param_mse)}, ...
%            'FitBoxToText', 'on', ...
%            'BackgroundColor', [1 1 1 0.7]);
% 
% fprintf('Plot created successfully comparing experimental de ata with models using different parameter sets\n');
% fprintf('MSE for optimized parameters: %.2f\n', opt_mse);
% fprintf('MSE for experimental parameters: %.2f\n', exp_param_mse);
% 
% % Save the figure
% saveas(gcf, 'encoder1_response_comparison.png');
% fprintf('Figure saved as encoder1_response_comparison.png\n');