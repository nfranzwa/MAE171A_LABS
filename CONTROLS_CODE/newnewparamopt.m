%% Enhanced Parameter Optimizer for Exact Plot Matching (<0.5% error)
% This script focuses on finding parameters that produce plots that match 
% the experimental encoder data to within 0.5% error with improved error handling

clear all; clc; close all;

%% Custom Initial parameters
initial_params = [
    1.06408675316770e-06,  % m1
    2.19156418903721e-06,  % d1
    1.88203847883458e-04,  % k1
    7.81950170595321e-07,  % m2
    4.64537678794473e-06,  % d2
    1.43896172739815e-04   % k2
];

%% Configuration
trial_num = 1;           % Trial number to use (x2.1.mat)
step_size = 0.5;         % Step input amplitude in Volts
dwell_time = 5000;       % Dwell time in msec
trim_start_time = 5;     % Time in seconds to trim from beginning of experimental data
max_plot_time = 4;       % Maximum time to show in plots (seconds)
num_optimization_attempts = 6;  % Increased number of optimization attempts for better results

% MAELAB configuration (must match what you'll use in MAELAB)
encoderout = 1;          % Using encoder 1 (options: 1 or 2)
DOFs = 2;                % Using 2DOF system
signofsystem = 1;        % Sign of encoder (1 or -1)

% Enhanced optimization configuration
max_iterations = 100;    % Increased max iterations for more thorough search
error_threshold = 0.005;  % 0.5% error threshold
min_bound_distance = 0.001; % Minimum distance between parameter bounds

% More aggressive bounds to enhance exploration
bound_lower_factor = 0.05;  % Lower bound is 5% of initial value
bound_upper_factor = 10;    % Upper bound is 10x initial value

%% Load and trim experimental data
fprintf('Loading and trimming experimental data...\n');
filename = sprintf('x2.%d.mat', trial_num);
try
    data = load(filename);
    exp_time_original = data.time;  
    exp_enc1_original = data.enc1;  % Encoder 1 data
    exp_enc2_original = data.enc2;  % Encoder 2 data
    
    % Trim data to remove points before trim_start_time
    trim_idx = exp_time_original >= trim_start_time;
    exp_time_trimmed = exp_time_original(trim_idx) - trim_start_time; % Shift time to start at 0
    exp_enc1_trimmed = exp_enc1_original(trim_idx);
    exp_enc2_trimmed = exp_enc2_original(trim_idx);
    
    % Further trim data to only include points up to max_plot_time for plotting
    plot_idx = exp_time_trimmed <= max_plot_time;
    exp_time_plot = exp_time_trimmed(plot_idx);
    exp_enc1_plot = exp_enc1_trimmed(plot_idx);
    exp_enc2_plot = exp_enc2_trimmed(plot_idx);
    
    fprintf('Successfully loaded and trimmed encoder data from %s\n', filename);
    fprintf('Trimmed first %.1f seconds of data, shifted time to start at 0\n', trim_start_time);
    fprintf('Plots will only show 0-%.1f seconds of data\n', max_plot_time);
    
    % Plot trimmed experimental data
    figure('Name', 'Trimmed Experimental Data', 'Position', [100, 100, 900, 600]);
    
    subplot(2,1,1);
    plot(exp_time_plot, exp_enc1_plot, 'b-', 'LineWidth', 1.5);
    title(sprintf('Trimmed Encoder 1 Data (0-%.1f sec)', max_plot_time));
    xlabel('Time (s)');
    ylabel('Encoder 1 Reading (counts)');
    xlim([0, max_plot_time]);
    grid on;
    
    subplot(2,1,2);
    plot(exp_time_plot, exp_enc2_plot, 'r-', 'LineWidth', 1.5);
    title(sprintf('Trimmed Encoder 2 Data (0-%.1f sec)', max_plot_time));
    xlabel('Time (s)');
    ylabel('Encoder 2 Reading (counts)');
    xlim([0, max_plot_time]);
    grid on;
    
catch ME
    error('Error loading data file: %s', ME.message);
end

%% Simulate initial model
fprintf('\nSimulating initial model...\n');

% Convert dwell time from msec to seconds
sim_dwell_time = max_plot_time; % Limit simulation to max_plot_time seconds

% Initialize transfer function with initial parameters
[sim_time_init, sim_enc1_init, sim_enc2_init] = simulate_model(initial_params, step_size, sim_dwell_time, encoderout, DOFs, signofsystem);

% Plot comparison with initial parameters
figure('Name', 'Initial Model vs Trimmed Experimental Data', 'Position', [150, 150, 900, 600]);

% Encoder 1 comparison
subplot(2,1,1);
plot(exp_time_plot, exp_enc1_plot, 'b-', 'LineWidth', 1.5);
hold on;
plot(sim_time_init, sim_enc1_init, 'g--', 'LineWidth', 1.5);
title('Encoder 1: Initial Model vs Trimmed Experiment');
xlabel('Time (s)');
ylabel('Encoder 1 Reading (counts)');
legend('Experimental (Trimmed)', 'Initial Model');
xlim([0, max_plot_time]);
grid on;

% Encoder 2 comparison
subplot(2,1,2);
plot(exp_time_plot, exp_enc2_plot, 'r-', 'LineWidth', 1.5);
hold on;
plot(sim_time_init, sim_enc2_init, 'g--', 'LineWidth', 1.5);
title('Encoder 2: Initial Model vs Trimmed Experiment');
xlabel('Time (s)');
ylabel('Encoder 2 Reading (counts)');
legend('Experimental (Trimmed)', 'Initial Model');
xlim([0, max_plot_time]);
grid on;

% Calculate initial error (using both MSE and percentage-based metrics)
initial_enc1_interp = interp1(sim_time_init, sim_enc1_init, exp_time_plot, 'linear', 'extrap');
initial_enc2_interp = interp1(sim_time_init, sim_enc2_init, exp_time_plot, 'linear', 'extrap');

% Calculate MSE error
initial_enc1_mse = mean((initial_enc1_interp - exp_enc1_plot).^2);
initial_enc2_mse = mean((initial_enc2_interp - exp_enc2_plot).^2);
initial_total_mse = initial_enc1_mse + 0.5 * initial_enc2_mse;

% Calculate percentage error - with safeguards for division by zero
try
    initial_enc1_pct = mean(abs((initial_enc1_interp - exp_enc1_plot) ./ max(abs(exp_enc1_plot), 1e-10)) * 100);
catch
    initial_enc1_pct = 100; % Default high error if calculation fails
end

try
    initial_enc2_pct = mean(abs((initial_enc2_interp - exp_enc2_plot) ./ max(abs(exp_enc2_plot), 1e-10)) * 100);
catch
    initial_enc2_pct = NaN; % Set to NaN if calculation fails
end

fprintf('\nInitial model error metrics:\n');
fprintf('MSE error: %.4e\n', initial_total_mse);
fprintf('Percentage error - Encoder 1: %.4f%%\n', initial_enc1_pct);
if isnan(initial_enc2_pct)
    fprintf('Percentage error - Encoder 2: NaN%%\n');
else
    fprintf('Percentage error - Encoder 2: %.4f%%\n', initial_enc2_pct);
end

%% Prepare for optimization
fprintf('\nPerforming optimization with high precision targeting...\n');

% Package parameters for optimization function
opt_data.exp_time_plot = exp_time_plot;
opt_data.exp_enc1_plot = exp_enc1_plot;
opt_data.exp_enc2_plot = exp_enc2_plot;
opt_data.step_size = step_size;
opt_data.dwell_time = sim_dwell_time; 
opt_data.encoderout = encoderout;
opt_data.DOFs = DOFs;
opt_data.signofsystem = signofsystem;
opt_data.target_pct_error = error_threshold * 100; % 0.5% in percentage terms

% Define initial parameter bounds (more aggressive for better exploration)
lb_initial = initial_params * bound_lower_factor;  % Lower bound: 5% of initial values
ub_initial = initial_params * bound_upper_factor;  % Upper bound: 10x initial values

% Apply minimum bound distance constraint
for i = 1:length(initial_params)
    if (ub_initial(i) - lb_initial(i)) < min_bound_distance
        mean_value = (ub_initial(i) + lb_initial(i)) / 2;
        half_range = min_bound_distance / 2;
        lb_initial(i) = max(0, mean_value - half_range);
        ub_initial(i) = mean_value + half_range;
    end
end

% Store best parameters and errors
best_params = initial_params;
best_error_mse = initial_total_mse;
best_error_pct_enc1 = initial_enc1_pct;
best_error_pct_enc2 = initial_enc2_pct;

% Define robust optimization methods that are less prone to failures
opt_methods = {
    'fminsearch',        % Basic Nelder-Mead works well for many optimization problems
    'particleswarm',     % Particle swarm for global search
    'robust_pattern',    % Our custom robust wrapper for patternsearch
    'adaptive_ga'        % Multi-stage genetic algorithm approach
};

% Flag for saving intermediate results
save_intermediates = true;
intermediate_params = cell(num_optimization_attempts, 1);
intermediate_errors = zeros(num_optimization_attempts, 3); % [MSE, Enc1%, Enc2%]

%% Run optimization
fprintf('\n=== Starting Parameter Optimization ===\n');

for attempt = 1:num_optimization_attempts
    fprintf('\n--- Optimization Attempt %d/%d ---\n', attempt, num_optimization_attempts);
    
    % Choose optimization method
    opt_method = opt_methods{mod(attempt-1, length(opt_methods))+1};
    fprintf('Using %s algorithm\n', opt_method);
    
    % Adjust bounds progressively as we iterate
    if attempt > 1 && best_error_pct_enc1 < initial_enc1_pct * 0.8
        % If we've improved significantly, narrow the search around the best point
        range_factor = 0.5 - 0.4 * min(1, best_error_pct_enc1 / initial_enc1_pct);
        lb = best_params * (1 - range_factor);
        ub = best_params * (1 + range_factor);
        fprintf('Narrowing search around best params (range factor: %.2f)\n', range_factor);
    else
        % Otherwise, keep exploring the full range
        lb = lb_initial;
        ub = ub_initial;
    end
    
    % Apply minimum bound distance constraint again
    for i = 1:length(initial_params)
        if (ub(i) - lb(i)) < min_bound_distance
            mean_value = (ub(i) + lb(i)) / 2;
            half_range = min_bound_distance / 2;
            lb(i) = max(0, mean_value - half_range);
            ub(i) = mean_value + half_range;
        end
    end
    
    % Add some randomness to starting point for better exploration
    if attempt > 1
        % Random factor between 0.8 and 1.2 for diversity
        rand_factor = 0.8 + 0.4*rand(size(initial_params)); 
        if mod(attempt, 3) == 0
            % Every 3rd attempt, use a more diverse starting point
            start_params = initial_params .* (0.6 + 0.8*rand(size(initial_params)));
        else
            % Otherwise, start near the best params found so far
            start_params = best_params .* rand_factor;
        end
        
        % Ensure starting parameters are within bounds
        start_params = max(min(start_params, ub), lb);
    else
        start_params = initial_params;
    end
    
    % Define robust cost function with scalar output guarantee
    robust_cost_function = @(params) safe_cost_function(params, opt_data);
    
    % Run optimization based on selected method with error handling
    try
        if strcmp(opt_method, 'robust_pattern')
            options = optimoptions('patternsearch', 'Display', 'iter', ...
                'MaxIterations', max_iterations, 'MeshTolerance', 1e-8, ...
                'FunctionTolerance', 1e-6, 'UseCompletePoll', true, ...
                'UseParallel', true);  % Enable parallel computing if available
            [current_params, current_error] = robust_patternsearch(robust_cost_function, start_params, [], [], [], [], lb, ub, [], options);
            
        elseif strcmp(opt_method, 'fminsearch')
            % fminsearch doesn't handle bounds directly, so we'll use a wrapper
            wrapped_cost_fn = @(params) cost_function_with_bounds(params, robust_cost_function, lb, ub);
            options = optimset('Display', 'iter', 'MaxIter', max_iterations, ...
                'TolFun', 1e-6, 'TolX', 1e-6);
            [unbounded_params, current_error] = fminsearch(wrapped_cost_fn, start_params, options);
            % Project back to feasible region
            current_params = max(min(unbounded_params, ub), lb);
            % Re-evaluate to get actual error at projected point
            current_error = robust_cost_function(current_params);
            
        elseif strcmp(opt_method, 'particleswarm')
            options = optimoptions('particleswarm', 'Display', 'iter', ...
                'MaxIterations', max_iterations, 'SwarmSize', 50, ...  % Increased swarm size
                'FunctionTolerance', 1e-4, 'UseParallel', true);  % Enable parallel computing
            [current_params, current_error] = particleswarm(robust_cost_function, length(start_params), lb, ub, options);
            
        elseif strcmp(opt_method, 'adaptive_ga')
            % Custom multi-stage genetic algorithm approach
            % First stage: Global exploration
            ga_options1 = optimoptions('ga', 'Display', 'iter', ...
                'MaxGenerations', round(max_iterations/3), ...
                'PopulationSize', 100, 'UseParallel', true);
            [ga_params1, ~] = ga(robust_cost_function, length(start_params), [], [], [], [], lb, ub, [], ga_options1);
            
            % Second stage: Local refinement
            % Narrow down the search around the first stage results
            lb2 = ga_params1 * 0.9;
            ub2 = ga_params1 * 1.1;
            
            ga_options2 = optimoptions('ga', 'Display', 'iter', ...
                'MaxGenerations', round(max_iterations/2), ...
                'PopulationSize', 50, 'UseParallel', true, ...
                'FunctionTolerance', 1e-6);
            [current_params, current_error] = ga(robust_cost_function, length(ga_params1), [], [], [], [], lb2, ub2, [], ga_options2);
        end
    catch ME
        warning('Optimization failed: %s\nUsing fallback method...', ME.message);
        
        % Fallback to a more robust method if the chosen one fails
        try
            options = optimoptions('particleswarm', 'Display', 'iter', ...
                'MaxIterations', round(max_iterations/2), 'SwarmSize', 30, ...
                'FunctionTolerance', 1e-3);
            [current_params, current_error] = particleswarm(robust_cost_function, length(start_params), lb, ub, options);
        catch ME2
            warning('Fallback optimization also failed: %s\nSkipping to next attempt.', ME2.message);
            continue; % Skip to next attempt
        end
    end
    
    % Evaluate the result with percentage error metric
    [current_error_mse, current_pct_enc1, current_pct_enc2] = evaluate_params(current_params, opt_data);
    
    % Save intermediate results
    if save_intermediates
        intermediate_params{attempt} = current_params;
        intermediate_errors(attempt, :) = [current_error_mse, current_pct_enc1, current_pct_enc2];
    end
    
    % Check if this is the best result so far (based on max percentage error)
    % If Encoder 2 gives NaN, just compare Encoder 1
    if isnan(current_pct_enc2) && isnan(best_error_pct_enc2)
        is_better = current_pct_enc1 < best_error_pct_enc1;
    elseif isnan(current_pct_enc2)
        is_better = current_pct_enc1 < best_error_pct_enc1;
    elseif isnan(best_error_pct_enc2)
        is_better = current_pct_enc1 < best_error_pct_enc1;
    else
        current_max_pct = max(current_pct_enc1, current_pct_enc2);
        best_max_pct = max(best_error_pct_enc1, best_error_pct_enc2);
        is_better = current_max_pct < best_max_pct;
    end
    
    if is_better
        best_params = current_params;
        best_error_mse = current_error_mse;
        best_error_pct_enc1 = current_pct_enc1;
        best_error_pct_enc2 = current_pct_enc2;
        fprintf('New best parameters found!\n');
        fprintf('Encoder 1 error: %.4f%%\n', best_error_pct_enc1);
        if isnan(best_error_pct_enc2)
            fprintf('Encoder 2 error: NaN%%\n');
        else
            fprintf('Encoder 2 error: %.4f%%\n', best_error_pct_enc2);
        end
        
        % Save best parameters found so far for recovery
        save('best_params_backup.mat', 'best_params', 'best_error_mse', 'best_error_pct_enc1', 'best_error_pct_enc2');
    end
    
    % Plot current result
    [sim_time_current, sim_enc1_current, sim_enc2_current] = simulate_model(current_params, step_size, sim_dwell_time, encoderout, DOFs, signofsystem);
    
    figure('Name', sprintf('Optimization - Attempt %d', attempt), 'Position', [200, 200, 900, 600]);
    
    % Encoder 1 comparison
    subplot(2,1,1);
    plot(exp_time_plot, exp_enc1_plot, 'b-', 'LineWidth', 1.5);
    hold on;
    plot(sim_time_current, sim_enc1_current, 'r--', 'LineWidth', 1.5);
    title(sprintf('Encoder 1: Attempt %d - Error: %.4f%%', attempt, current_pct_enc1));
    xlabel('Time (s)');
    ylabel('Encoder 1 Reading (counts)');
    legend('Experimental (Trimmed)', 'Optimized Model');
    xlim([0, max_plot_time]);
    grid on;
    
    % Encoder 2 comparison
    subplot(2,1,2);
    plot(exp_time_plot, exp_enc2_plot, 'r-', 'LineWidth', 1.5);
    hold on;
    plot(sim_time_current, sim_enc2_current, 'r--', 'LineWidth', 1.5);
    if isnan(current_pct_enc2)
        title(sprintf('Encoder 2: Attempt %d - Error: NaN%%', attempt));
    else
        title(sprintf('Encoder 2: Attempt %d - Error: %.4f%%', attempt, current_pct_enc2));
    end
    xlabel('Time (s)');
    ylabel('Encoder 2 Reading (counts)');
    legend('Experimental (Trimmed)', 'Optimized Model');
    xlim([0, max_plot_time]);
    grid on;
    
    % Check if we've reached our target error
    if best_error_pct_enc1 < opt_data.target_pct_error
        if isnan(best_error_pct_enc2)
            fprintf('\nTarget error of <%.2f%% achieved for Encoder 1! Ending optimization early.\n', opt_data.target_pct_error);
            fprintf('(Unable to optimize for Encoder 2 due to data issues)\n');
        elseif best_error_pct_enc2 < opt_data.target_pct_error
            fprintf('\nTarget error of <%.2f%% achieved for both encoders! Ending optimization early.\n', opt_data.target_pct_error);
        end
        break;
    end
end

% Multi-point optimization - Use best candidates from different attempts
if save_intermediates && ~(best_error_pct_enc1 < opt_data.target_pct_error)
    fprintf('\n--- Performing Multi-Point Refinement ---\n');
    
    % Get top 3 parameter sets based on encoder 1 error
    [~, idx] = sort(intermediate_errors(:, 2));
    top_idx = idx(1:min(3, length(idx)));
    top_params = cell2mat(intermediate_params(top_idx)');
    
    % Average the parameters with weights based on their errors
    weights = 1 ./ intermediate_errors(top_idx, 2);
    weights = weights / sum(weights);
    avg_params = zeros(size(initial_params));
    
    for i = 1:length(top_idx)
        avg_params = avg_params + weights(i) * intermediate_params{top_idx(i)};
    end
    
    % Final refinement with fminsearch around the weighted average
    fprintf('Refining based on weighted average of top parameter sets...\n');
    lb_fine = avg_params * 0.95;
    ub_fine = avg_params * 1.05;
    
    wrapped_cost_fn = @(params) cost_function_with_bounds(params, robust_cost_function, lb_fine, ub_fine);
    options = optimset('Display', 'iter', 'MaxIter', 2000, 'TolFun', 1e-8, 'TolX', 1e-8);
    
    try
        [refined_params, ~] = fminsearch(wrapped_cost_fn, avg_params, options);
        refined_params = max(min(refined_params, ub_fine), lb_fine);
        [refined_error_mse, refined_pct_enc1, refined_pct_enc2] = evaluate_params(refined_params, opt_data);
        
        % Check if refined parameters are better
        if refined_pct_enc1 < best_error_pct_enc1
            best_params = refined_params;
            best_error_mse = refined_error_mse;
            best_error_pct_enc1 = refined_pct_enc1;
            best_error_pct_enc2 = refined_pct_enc2;
            fprintf('Multi-point refinement successful!\n');
            fprintf('Encoder 1 error improved to: %.4f%%\n', best_error_pct_enc1);
        else
            fprintf('Multi-point refinement did not improve results.\n');
        end
    catch
        fprintf('Multi-point refinement failed, keeping previous best parameters.\n');
    end
end

% Assign the best parameters found as our optimized parameters
optimized_params = best_params;

%% Display final optimization results
fprintf('\n=== Final Optimization Results ===\n');
fprintf('Initial error metrics:\n');
fprintf('  Encoder 1: %.4f%%\n', initial_enc1_pct);
if isnan(initial_enc2_pct)
    fprintf('  Encoder 2: NaN%% (unable to calculate)\n');
else
    fprintf('  Encoder 2: %.4f%%\n', initial_enc2_pct);
end

fprintf('Final error metrics:\n');
fprintf('  Encoder 1: %.4f%%\n', best_error_pct_enc1);
if isnan(best_error_pct_enc2)
    fprintf('  Encoder 2: NaN%% (unable to calculate)\n');
    fprintf('  Note: Optimization focused on Encoder 1 only due to issues with Encoder 2 data\n');
else
    fprintf('  Encoder 2: %.4f%%\n', best_error_pct_enc2);
end

fprintf('Improvement:\n');
fprintf('  Encoder 1: %.2f%%\n', (1 - best_error_pct_enc1/initial_enc1_pct)*100);
if ~isnan(initial_enc2_pct) && ~isnan(best_error_pct_enc2)
    fprintf('  Encoder 2: %.2f%%\n', (1 - best_error_pct_enc2/initial_enc2_pct)*100);
end

% Check if we met our target
if best_error_pct_enc1 <= opt_data.target_pct_error 
    if isnan(best_error_pct_enc2)
        fprintf('\n✓ PARTIAL SUCCESS: Achieved <%.2f%% error for Encoder 1!\n', opt_data.target_pct_error);
        fprintf('  (Unable to optimize for Encoder 2 due to data issues)\n');
    elseif best_error_pct_enc2 <= opt_data.target_pct_error
        fprintf('\n✓ SUCCESS: Achieved <%.2f%% error for both encoders!\n', opt_data.target_pct_error);
    else
        fprintf('\n✓ PARTIAL SUCCESS: Achieved <%.2f%% error for Encoder 1, but not for Encoder 2.\n', opt_data.target_pct_error);
        fprintf('  Encoder 2 error: %.4f%%\n', best_error_pct_enc2);
    end
else
    fprintf('\n! TARGET NOT MET: Could not achieve <%.2f%% error target.\n', opt_data.target_pct_error);
    fprintf('  Best results: Encoder 1: %.4f%%', best_error_pct_enc1);
    if ~isnan(best_error_pct_enc2)
        fprintf(', Encoder 2: %.4f%%\n', best_error_pct_enc2);
    else
        fprintf('\n  (Encoder 2 optimization not possible due to data issues)\n');
    end
end

% Print parameter comparison
param_names = {'m1', 'd1', 'k1', 'm2', 'd2', 'k2'};
fprintf('\nParameter   | Initial Value         | Optimized Value      | Change (%%) \n');
fprintf('------------------------------------------------------------------------\n');
for i = 1:length(param_names)
    pct_change = (optimized_params(i) - initial_params(i)) / initial_params(i) * 100;
    fprintf('%-11s | %-21.6e | %-21.6e | %+7.2f\n', ...
        param_names{i}, initial_params(i), optimized_params(i), pct_change);
end

%% Simulate final optimized model
fprintf('\nSimulating final optimized model...\n');

% Simulate with optimized parameters
[sim_time_opt, sim_enc1_opt, sim_enc2_opt] = simulate_model(optimized_params, step_size, sim_dwell_time, encoderout, DOFs, signofsystem);

% Plot comparison with optimized parameters
figure('Name', 'Final Optimized Model vs Experimental Data', 'Position', [300, 300, 900, 800]);

% Encoder 1 comparison with error display
subplot(2,2,1);
plot(exp_time_plot, exp_enc1_plot, 'b-', 'LineWidth', 1.5);
hold on;
plot(sim_time_opt, sim_enc1_opt, 'r--', 'LineWidth', 1.5);
title(sprintf('Encoder 1: Final Model - Error: %.4f%%', best_error_pct_enc1));
xlabel('Time (s)');
ylabel('Encoder 1 Reading (counts)');
legend('Experimental', 'Optimized Model');
xlim([0, max_plot_time]);
grid on;

% Encoder 2 comparison with error display
subplot(2,2,2);
plot(exp_time_plot, exp_enc2_plot, 'r-', 'LineWidth', 1.5);
hold on;
plot(sim_time_opt, sim_enc2_opt, 'r--', 'LineWidth', 1.5);
if isnan(best_error_pct_enc2)
    title('Encoder 2: Final Model - Error: NaN%');
else
    title(sprintf('Encoder 2: Final Model - Error: %.4f%%', best_error_pct_enc2));
end
xlabel('Time (s)');
ylabel('Encoder 2 Reading (counts)');
legend('Experimental', 'Optimized Model');
xlim([0, max_plot_time]);
grid on;

% Error plots for detailed analysis
sim_enc1_interp = interp1(sim_time_opt, sim_enc1_opt, exp_time_plot, 'linear', 'extrap');
sim_enc2_interp = interp1(sim_time_opt, sim_enc2_opt, exp_time_plot, 'linear', 'extrap');

% Calculate point-by-point percentage errors with safety checks
point_pct_err1 = zeros(size(exp_time_plot));
for i = 1:length(exp_time_plot)
    if abs(exp_enc1_plot(i)) > 1e-10
        point_pct_err1(i) = abs((sim_enc1_interp(i) - exp_enc1_plot(i)) / exp_enc1_plot(i)) * 100;
    else
        point_pct_err1(i) = 0;
    end
end

point_pct_err2 = zeros(size(exp_time_plot));
for i = 1:length(exp_time_plot)
    if abs(exp_enc2_plot(i)) > 1e-10
        point_pct_err2(i) = abs((sim_enc2_interp(i) - exp_enc2_plot(i)) / exp_enc2_plot(i)) * 100;
    else
        point_pct_err2(i) = 0;
    end
end

% Plot percentage errors over time
subplot(2,2,3);
plot(exp_time_plot, point_pct_err1, 'b-', 'LineWidth', 1.5);
title('Encoder 1: Percentage Error Over Time');
xlabel('Time (s)');
ylabel('Error (%)');
yline(0.5, 'r--', '0.5% Target', 'LineWidth', 1.5);
xlim([0, max_plot_time]);
grid on;

subplot(2,2,4);
plot(exp_time_plot, point_pct_err2, 'r-', 'LineWidth', 1.5);
title('Encoder 2: Percentage Error Over Time');
xlabel('Time (s)');
ylabel('Error (%)');
yline(0.5, 'r--', '0.5% Target', 'LineWidth', 1.5);
xlim([0, max_plot_time]);
grid on;

%% Final comparison: Initial vs Optimized vs Experimental with Percentage Error Target
figure('Name', 'Complete Comparison', 'Position', [350, 350, 1000, 800]);

% Encoder 1 comparison
subplot(2,2,1);
plot(exp_time_plot, exp_enc1_plot, 'b-', 'LineWidth', 1.5);
hold on;
plot(sim_time_init, sim_enc1_init, 'g--', 'LineWidth', 1.5);
plot(sim_time_opt, sim_enc1_opt, 'r--', 'LineWidth', 1.5);
title('Encoder 1: Complete Comparison (0-4 sec)');
xlabel('Time (s)');
ylabel('Encoder 1 Reading (counts)');
legend('Experimental (Trimmed)', 'Initial Model', 'Optimized Model');
xlim([0, max_plot_time]);
grid on;

% Encoder 2 comparison
subplot(2,2,2);
plot(exp_time_plot, exp_enc2_plot, 'r-', 'LineWidth', 1.5);
hold on;
plot(sim_time_init, sim_enc2_init, 'g--', 'LineWidth', 1.5);
plot(sim_time_opt, sim_enc2_opt, 'r--', 'LineWidth', 1.5);
title('Encoder 2: Complete Comparison (0-4 sec)');
xlabel('Time (s)');
ylabel('Encoder 2 Reading (counts)');
legend('Experimental (Trimmed)', 'Initial Model', 'Optimized Model');
xlim([0, max_plot_time]);
grid on;

% Add error comparison plots to evaluate against the 0.5% target
initial_enc1_interp = interp1(sim_time_init, sim_enc1_init, exp_time_plot, 'linear', 'extrap');
initial_enc2_interp = interp1(sim_time_init, sim_enc2_init, exp_time_plot, 'linear', 'extrap');

% Calculate point-by-point percentage errors for both initial and optimized models
% with safety checks
initial_pct_err1 = zeros(size(exp_time_plot));
for i = 1:length(exp_time_plot)
    if abs(exp_enc1_plot(i)) > 1e-10
        initial_pct_err1(i) = abs((initial_enc1_interp(i) - exp_enc1_plot(i)) / exp_enc1_plot(i)) * 100;
    else
        initial_pct_err1(i) = 0;
    end
end

initial_pct_err2 = zeros(size(exp_time_plot));
for i = 1:length(exp_time_plot)
    if abs(exp_enc2_plot(i)) > 1e-10
        initial_pct_err2(i) = abs((initial_enc2_interp(i) - exp_enc2_plot(i)) / exp_enc2_plot(i)) * 100;
    else
        initial_pct_err2(i) = 0;
    end
end

% Plot percentage errors for Encoder 1
subplot(2,2,3);
plot(exp_time_plot, initial_pct_err1, 'g-', 'LineWidth', 1.5);
hold on;
plot(exp_time_plot, point_pct_err1, 'r-', 'LineWidth', 1.5);
title('Encoder 1: Percentage Error Comparison');
xlabel('Time (s)');
ylabel('Error (%)');
yline(0.5, 'k--', '0.5% Target', 'LineWidth', 1.5);
legend('Initial Model Error', 'Optimized Model Error', 'Target Threshold');
xlim([0, max_plot_time]);
grid on;

% Plot percentage errors for Encoder 2
subplot(2,2,4);
plot(exp_time_plot, initial_pct_err2, 'g-', 'LineWidth', 1.5);
hold on;
plot(exp_time_plot, point_pct_err2, 'r-', 'LineWidth', 1.5);
title('Encoder 2: Percentage Error Comparison');
xlabel('Time (s)');
ylabel('Error (%)');
yline(0.5, 'k--', '0.5% Target', 'LineWidth', 1.5);
legend('Initial Model Error', 'Optimized Model Error', 'Target Threshold');
xlim([0, max_plot_time]);
grid on;

%% Create parameters.m file
fprintf('\nGenerating parameters.m file...\n');

fid = fopen('parameters.m', 'w');
fprintf(fid, '%% Parameters for 2DOF system optimized to match encoder data with <0.5%% error\n');
fprintf(fid, '%% Generated automatically on %s\n\n', datestr(now));
fprintf(fid, '%% Achieved accuracy:\n');
fprintf(fid, '%% - Encoder 1: %.4f%% average error\n', best_error_pct_enc1);
if isnan(best_error_pct_enc2)
    fprintf(fid, '%% - Encoder 2: NaN%% (unable to calculate)\n\n');
else
    fprintf(fid, '%% - Encoder 2: %.4f%% average error\n\n', best_error_pct_enc2);
end
fprintf(fid, 'm1 = %.14e; %% Mass m1\n', optimized_params(1));
fprintf(fid, 'd1 = %.14e; %% Damping d1\n', optimized_params(2));
fprintf(fid, 'k1 = %.14e; %% Spring constant k1\n', optimized_params(3));
fprintf(fid, 'm2 = %.14e; %% Mass m2\n', optimized_params(4));
fprintf(fid, 'd2 = %.14e; %% Damping d2\n', optimized_params(5));
fprintf(fid, 'k2 = %.14e; %% Spring constant k2\n', optimized_params(6));
fclose(fid);

fprintf('Parameters file "parameters.m" has been created and is ready to use with MAELAB.\n');
fprintf('\nTo use with MAELAB:\n');
fprintf('1. Run maelab.m\n');
fprintf('2. Choose encoder output: %d\n', encoderout);
fprintf('3. Choose parameter file: parameters\n');
fprintf('4. Choose degrees of freedom: %d\n', DOFs);
fprintf('5. Choose sign of encoder: %d\n', signofsystem);
fprintf('6. Select option 2 to simulate/compare open loop step response\n');
fprintf('7. Enter step size: %.1f Volts\n', step_size);
fprintf('8. Enter dwell time: %d msec\n', dwell_time);
fprintf('9. For open-loop step experiment file, enter: x2.%d\n', trial_num);

fprintf('\nOptimization complete!\n');

%% Helper Functions
function [sim_time, sim_enc1, sim_enc2] = simulate_model(params, step_amplitude, dwell_time, encoderout, DOFs, signofsystem)
    % Simplified simulation of 2DOF model with better sampling rate
    
    % Extract parameters
    m1 = params(1);
    d1 = params(2);
    k1 = params(3);
    m2 = params(4);
    d2 = params(5);
    k2 = params(6);
    
    % Build 2DOF model - exactly as in MAELAB
    if DOFs == 2
        % Full 2DOF model
        deng = [m1*m2, (m1*d2+m2*d1), (k2*m1+(k1+k2)*m2+d1*d2), ((k1+k2)*d2+k2*d1), k1*k2];
        numg1 = [m2, d2, k2];  % Numerator for encoder 1
        numg2 = [k2];          % Numerator for encoder 2
    else
        % 1DOF model
        if encoderout == 1
            deng = [m1, d1, k1];
            numg1 = [1];
            numg2 = [0];  % Not used in 1DOF with encoder 1
        else
            deng = [m2, d2, k2];
            numg1 = [0];  % Not used in 1DOF with encoder 2
            numg2 = [1];
        end
    end
    
    % Apply sign of system
    numg1 = signofsystem * numg1;
    numg2 = signofsystem * numg2;
    
    % Store original transfer function (without motor dynamics)
    numg1_0 = numg1;
    numg2_0 = numg2;
    
    % Enhanced motor dynamics model 
    motor_time_constant = 1/209;
    
    % Add DC motor dynamics (actuator)
    deng = conv(deng, [motor_time_constant, 1]);
    
    % Create transfer functions
    G1 = tf(numg1_0, deng);  % Transfer function for encoder 1
    G2 = tf(numg2_0, deng);  % Transfer function for encoder 2
    
    % Use many more points to avoid undersampling warnings
    % Calculate appropriate time step based on system dynamics
    % Get the fastest time constant of the system
    [~, p] = tfdata(G1, 'v');
    fastest_pole = max(abs(roots(p)));
    if fastest_pole > 0
        % If system has unstable poles, use a conservative sampling rate
        dt = 0.0001; 
    else
        % Otherwise use a reasonable fraction of the fastest time constant
        dt = min(0.01, 0.1/fastest_pole);
    end
    
    % Create time vector with appropriate sampling
    sim_time = 0:dt:dwell_time;
    
    % Step input
    u = step_amplitude * ones(size(sim_time));
    
    % Simulate step response for both encoders
    [sim_enc1, ~] = lsim(G1, u, sim_time);
    [sim_enc2, ~] = lsim(G2, u, sim_time);
end

function [error_mse, pct_error_enc1, pct_error_enc2] = evaluate_params(params, data)
    % Evaluates parameters and returns both MSE and percentage errors
    
    % First check if params are positive
    if any(params <= 0)
        error_mse = 1e10;
        pct_error_enc1 = 100;
        pct_error_enc2 = 100;
        return;
    end
    
    % Wrap simulation in try-catch to handle numerical issues
    try
        % Simulate model with current parameters
        [sim_time, sim_enc1, sim_enc2] = simulate_model(params, data.step_size, ...
            data.dwell_time, data.encoderout, data.DOFs, data.signofsystem);
        
        % Interpolate simulation results to match experimental time points
        sim_enc1_interp = interp1(sim_time, sim_enc1, data.exp_time_plot, 'linear', 'extrap');
        sim_enc2_interp = interp1(sim_time, sim_enc2, data.exp_time_plot, 'linear', 'extrap');
        
        % Handle invalid interpolation
        if any(isnan(sim_enc1_interp)) || any(isnan(sim_enc2_interp)) || ...
           any(isinf(sim_enc1_interp)) || any(isinf(sim_enc2_interp))
            error_mse = 1e10;
            pct_error_enc1 = 100;
            pct_error_enc2 = 100;
            return;
        end
        
        % Calculate MSE
        enc1_mse = mean((sim_enc1_interp - data.exp_enc1_plot).^2);
        enc2_mse = mean((sim_enc2_interp - data.exp_enc2_plot).^2);
        error_mse = enc1_mse + 0.5 * enc2_mse;
        
        % Protect against division by zero and NaN in percentage calculation
        safe_exp_enc1 = data.exp_enc1_plot;
        safe_exp_enc2 = data.exp_enc2_plot;
        
        % Replace zeros with small values to avoid division by zero
        zero_indices_1 = abs(safe_exp_enc1) < 1e-10;
        zero_indices_2 = abs(safe_exp_enc2) < 1e-10;
        
        % If there are zeros in the data, use MSE for those points instead of percentage
        pct_errors_enc1 = zeros(size(safe_exp_enc1));
        pct_errors_enc2 = zeros(size(safe_exp_enc2));
        
        % Calculate percentage errors for non-zero values
        pct_errors_enc1(~zero_indices_1) = abs((sim_enc1_interp(~zero_indices_1) - safe_exp_enc1(~zero_indices_1)) ./ safe_exp_enc1(~zero_indices_1)) * 100;
        pct_errors_enc2(~zero_indices_2) = abs((sim_enc2_interp(~zero_indices_2) - safe_exp_enc2(~zero_indices_2)) ./ safe_exp_enc2(~zero_indices_2)) * 100;
        
        % For zero values, use a scaled absolute error instead
        if any(zero_indices_1)
            mean_exp_enc1 = mean(abs(safe_exp_enc1(~zero_indices_1)));
            pct_errors_enc1(zero_indices_1) = abs(sim_enc1_interp(zero_indices_1) - safe_exp_enc1(zero_indices_1)) / (mean_exp_enc1 + eps) * 100;
        end
        
        if any(zero_indices_2)
            mean_exp_enc2 = mean(abs(safe_exp_enc2(~zero_indices_2)));
            pct_errors_enc2(zero_indices_2) = abs(sim_enc2_interp(zero_indices_2) - safe_exp_enc2(zero_indices_2)) / (mean_exp_enc2 + eps) * 100;
        end
        
        % Average percentage errors (handle NaN values)
        pct_error_enc1 = mean(pct_errors_enc1(~isnan(pct_errors_enc1)));
        pct_error_enc2 = mean(pct_errors_enc2(~isnan(pct_errors_enc2)));
        
        % If no valid points remain, return high error
        if isnan(pct_error_enc1) || isnan(pct_error_enc2)
            pct_error_enc1 = 100;
            pct_error_enc2 = 100;
        end
        
    catch ME
        % If simulation fails, return high error values
        warning('Simulation failed with error: %s', ME.message);
        error_mse = 1e10;
        pct_error_enc1 = 100;
        pct_error_enc2 = 100;
    end
end

function error = safe_cost_function(params, data)
    % Robust wrapper for the cost function
    % Ensures a scalar output even if the underlying calculation fails
    
    try
        error = calculate_percentage_error(params, data);
        
        % Ensure the result is a scalar
        if ~isscalar(error) || isnan(error) || isinf(error)
            error = 1e10;
        end
    catch ME
        % If any error occurs, return a large error value
        warning('Cost function calculation failed: %s', ME.message);
        error = 1e10;
    end
end

function error = calculate_percentage_error(params, data)
    % Error function focused on percentage error for high precision matching
    
    % Ensure parameters are positive
    if any(params <= 0)
        error = 1e10;
        return;
    end
    
    % Get percentage errors
    [error_mse, pct_error_enc1, pct_error_enc2] = evaluate_params(params, data);
    
    % If we have NaN or inf values, return a large penalty
    if isnan(error_mse) || isinf(error_mse) || isnan(pct_error_enc1) || isnan(pct_error_enc2)
        error = 1e10;
        return;
    end
    
    % If encoder 2 is giving NaN errors (which seems to be the case from your output),
    % focus optimization on encoder 1 only
    if isnan(pct_error_enc2) || pct_error_enc2 > 1000
        error = pct_error_enc1^2;
    else
        % Focus on the maximum percentage error between the two encoders
        % This ensures we're optimizing to get both encoders below the target
        max_pct_error = max(pct_error_enc1, pct_error_enc2);
        
        % Apply progressive penalty as we get closer to the target
        if max_pct_error > data.target_pct_error
            % More aggressive penalty for being above target
            error = max_pct_error^2;
        else
            % Lighter penalty once we're below target, to further refine
            error = max_pct_error;
        end
        
        % Add small weight to favor encoder 1 slightly (often more important)
        error = error + 0.1 * pct_error_enc1;
    end
    
    % Add a small MSE component to maintain good overall fit
    error = error + 1e-6 * error_mse;
    
    % Final error value with minimal regularization
    error = error + 1e-12 * sum(params.^2);
end

function error = calculate_global_error(params, data)
    % Error function for the global search stage
    % Balances between MSE and percentage error
    
    % Ensure parameters are positive
    if any(params <= 0)
        error = 1e10;
        return;
    end
    
    % Get MSE and percentage errors
    [error_mse, pct_error_enc1, pct_error_enc2] = evaluate_params(params, data);
    
    % If we have NaN or inf values, return a large penalty
    if isnan(error_mse) || isinf(error_mse) || isnan(pct_error_enc1)
        error = 1e10;
        return;
    end
    
    % If encoder 2 is giving NaN errors (which seems to be the case from your output),
    % focus optimization on encoder 1 and MSE
    if isnan(pct_error_enc2) || pct_error_enc2 > 1000
        error = error_mse * (1 + pct_error_enc1/100);
    else
        % Combine MSE and percentage error approaches
        % This helps the global search find good starting points
        error = error_mse * (1 + max(pct_error_enc1, pct_error_enc2)/100);
    end
    
    % Add regularization
    regularization = 1e-9 * sum(params.^2);
    
    % Final error value
    error = error + regularization;
end

function cost = cost_function_with_bounds(params, orig_fn, lb, ub)
    % Apply penalty if parameters are outside bounds
    penalty = 0;
    for i = 1:length(params)
        if params(i) < lb(i)
            penalty = penalty + 1e10 * (lb(i) - params(i))^2;
        elseif params(i) > ub(i)
            penalty = penalty + 1e10 * (params(i) - ub(i))^2;
        end
    end
    
    % Only evaluate the original function if we're in bounds
    if penalty > 0
        cost = penalty;
    else
        cost = orig_fn(params);
    end
end

function [x, fval] = robust_patternsearch(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options)
    % Robust wrapper for patternsearch that catches and handles errors
    
    try
        [x, fval] = patternsearch(fun, x0, A, b, Aeq, beq, lb, ub, nonlcon, options);
    catch ME
        warning('Patternsearch failed: %s\nUsing fallback optimization...', ME.message);
        
        % Try particle swarm as a fallback
        try
            ps_options = optimoptions('particleswarm', 'Display', 'iter', ...
                'MaxIterations', 3000, 'SwarmSize', 20, ...
                'FunctionTolerance', 1e-3);
            [x, fval] = particleswarm(fun, length(x0), lb, ub, ps_options);
        catch ME2
            warning('Fallback particleswarm also failed: %s\nUsing fminsearch...', ME2.message);
            
            % Last resort: Try fminsearch with bounds
            try
                bounded_fun = @(p) cost_function_with_bounds(p, fun, lb, ub);
                ms_options = optimset('Display', 'iter', 'MaxIter', 3000, ...
                    'TolFun', 1e-4, 'TolX', 1e-4);
                [unbounded_x, fval] = fminsearch(bounded_fun, x0, ms_options);
                
                % Project back to bounds
                x = min(max(unbounded_x, lb), ub);
            catch ME3
                warning('All optimization methods failed. Using starting point.');
                x = x0;
                fval = fun(x0);
            end
        end
    end
end