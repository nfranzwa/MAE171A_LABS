%% PID Design for Minimized Settling Time - Iterative Refinement
% This script refines PID controller parameters starting from provided values
% to minimize settling time while meeting specified constraints, focusing on Kp

clear all; close all; clc;

%% Load the transfer function
load('G.mat'); % Load the open loop transfer function G

% Display information about the transfer function
disp('Transfer Function Information:');
G

%% Analyze the uncompensated system
poles = pole(G);
zeros = zero(G);

disp('Poles of the open loop transfer function:');
disp(poles);
disp('Zeros of the open loop transfer function:');
disp(zeros);

% Plot the original root locus
figure(1);
rlocus(G);
title('Root Locus of Uncompensated System');
grid on;
sgrid(0.6, 0); % Draw constant damping ratio line (ζ = 0.6)
xlabel('Real Axis');
ylabel('Imaginary Axis');

% Step response of uncompensated system with longer simulation time
figure(2);
opt = stepDataOptions('StepAmplitude', 1000);
step(G, 5, opt);
title('Step Response of Uncompensated System');
grid on;

%% Design Constraints:
% - 0 < kp < 1
% - 0 < kd < 0.02
% - 0 < ki < 1
% - Overshoot < 25%
% - Steady-state error < 2%
% - Reference input: 1000 encoder counts
% - PRIORITY: Minimize settling time while meeting all constraints

disp('==== PID Controller Design with Settling Time Minimization (Exclusively Prioritizing Kp Changes) ====');

% Define simulation options with Final Value specified to avoid warnings
sim_time = 5; % 5 seconds simulation time
ref_input = 1000; % Reference input in encoder counts

%% Iterative Parameter Refinement
% Starting point (provided values)
start_Kp = 0.13467;
start_Ki = 0.94435;
start_Kd = 0.02;

% Define parameter sampling and search ranges
num_iterations = 3;
num_Kp_samples = 30; % Even more samples for Kp
num_Ki_samples = 3;  % Minimal samples for Ki
num_Kd_samples = 3;  % Minimal samples for Kd

% Define search ranges - much wider for Kp, extremely narrow for Ki and Kd
Kp_range = [max(0.001, start_Kp * 0.3), min(1.0, start_Kp * 3.0)];    % 30-300% variation
Ki_range = [max(0.001, start_Ki * 0.99), min(1.0, start_Ki * 1.01)];  % 1% variation
Kd_range = [max(0.001, start_Kd * 0.99), min(0.02, start_Kd * 1.01)]; % 1% variation

% Initialize best parameters with starting point
best_Kp = start_Kp;
best_Ki = start_Ki;
best_Kd = start_Kd;
best_settling = Inf;
best_overshoot = NaN;
best_sse = NaN;

% Evaluate starting point
% Create PID controller
start_PID = tf([start_Kd, start_Kp, start_Ki], [1, 0]);
% Create closed-loop system
start_CL = feedback(start_PID*G, 1);

% Check if system is stable
if isstable(start_CL)
    % Simulate step response with reference input
    opt = stepDataOptions('StepAmplitude', ref_input);
    [y, t] = step(start_CL, sim_time, opt);
    
    % Final value approximation
    final_value = mean(y(max(1, end-10):end));
    
    % Calculate steady-state error
    start_sse = abs(ref_input - final_value) / ref_input * 100;
    
    % Find overshoot
    max_value = max(y);
    start_overshoot = max(0, (max_value - final_value) / final_value * 100);
    
    % Calculate settling time (2% criterion)
    settle_band = 0.02 * final_value;
    settled_indices = find(abs(y - final_value) <= settle_band);
    
    start_settling = Inf;
    if ~isempty(settled_indices)
        % Find the first index where the response stays within the band
        for idx = 1:length(settled_indices)-1
            if all(diff(settled_indices(idx:end)) == 1)
                start_settling = t(settled_indices(idx));
                break;
            end
        end
    end
    
    % Check if constraints are met
    start_valid = (start_overshoot < 25) && (start_sse < 2) && isfinite(start_settling);
    
    % Display starting point evaluation
    disp('Starting Point Evaluation:');
    disp(['Kp = ', num2str(start_Kp), ', Ki = ', num2str(start_Ki), ', Kd = ', num2str(start_Kd)]);
    disp(['Settling Time = ', num2str(start_settling), 's, Overshoot = ', num2str(start_overshoot), '%, SSE = ', num2str(start_sse), '%']);
    disp(['Valid = ', num2str(start_valid)]);
    
    % Initialize best params if starting point is valid
    if start_valid
        best_settling = start_settling;
        best_overshoot = start_overshoot;
        best_sse = start_sse;
    end
else
    disp('Starting point produces an unstable system.');
    start_valid = false;
    start_settling = Inf;
    start_overshoot = Inf;
    start_sse = Inf;
end

% Storage for all valid results
all_results = [];
if start_valid
    all_results = [all_results; start_Kp, start_Ki, start_Kd, start_settling, start_overshoot, start_sse];
end

% Begin iterative refinement
disp('Starting iterative parameter refinement (focusing almost exclusively on Kp optimization)...');
for iter = 1:num_iterations
    disp(['==== Iteration ', num2str(iter), ' of ', num2str(num_iterations), ' ====']);
    
    % Create parameter samples - more for Kp, fewer for Ki and Kd
    Kp_values = linspace(Kp_range(1), Kp_range(2), num_Kp_samples);
    Ki_values = linspace(Ki_range(1), Ki_range(2), num_Ki_samples);
    Kd_values = linspace(Kd_range(1), Kd_range(2), num_Kd_samples);
    
    % Evaluate each parameter combination
    improved = false;
    total_combinations = length(Kp_values) * length(Ki_values) * length(Kd_values);
    iteration_count = 0;
    valid_count = 0;
    
    tic; % Start timing
    
    % Display progress update
    fprintf('Testing %d parameter combinations...\n', total_combinations);
    
    iteration_results = [];
    
    for i = 1:length(Kp_values)
        for j = 1:length(Ki_values)
            for k = 1:length(Kd_values)
                iteration_count = iteration_count + 1;
                
                % Show progress every 10%
                if mod(iteration_count, round(total_combinations/10)) == 0
                    fprintf('Progress: %d%% (%d/%d) - Time elapsed: %.1f seconds\n', ...
                        round(100*iteration_count/total_combinations), iteration_count, total_combinations, toc);
                end
                
                % Get current parameter set
                Kp = Kp_values(i);
                Ki = Ki_values(j);
                Kd = Kd_values(k);
                
                % Skip if identical to best parameters (to save time)
                if Kp == best_Kp && Ki == best_Ki && Kd == best_Kd
                    continue;
                end
                
                % Create PID controller
                PID = tf([Kd, Kp, Ki], [1, 0]);
                
                % Create closed-loop system
                CL = feedback(PID*G, 1);
                
                % Check if system is stable
                if isstable(CL)
                    % Simulate step response with reference input
                    try
                        opt = stepDataOptions('StepAmplitude', ref_input);
                        [y, t] = step(CL, sim_time, opt);
                        
                        % Final value approximation
                        final_value = mean(y(max(1, end-10):end));
                        
                        % Calculate steady-state error
                        sse = abs(ref_input - final_value) / ref_input * 100;
                        
                        % Find overshoot
                        max_value = max(y);
                        overshoot = max(0, (max_value - final_value) / final_value * 100);
                        
                        % Calculate settling time (2% criterion)
                        settle_band = 0.02 * final_value;
                        settled_indices = find(abs(y - final_value) <= settle_band);
                        
                        settling_time = Inf;
                        if ~isempty(settled_indices)
                            % Find the first index where the response stays within the band
                            for idx = 1:length(settled_indices)-1
                                if all(diff(settled_indices(idx:end)) == 1)
                                    settling_time = t(settled_indices(idx));
                                    break;
                                end
                            end
                        end
                        
                        % Check if constraints are met
                        if overshoot < 25 && sse < 2 && isfinite(settling_time)
                            valid_count = valid_count + 1;
                            iteration_results = [iteration_results; Kp, Ki, Kd, settling_time, overshoot, sse];
                        end
                    catch
                        % Skip if simulation fails
                    end
                end
            end
        end
    end
    
    % Add penalty for Ki and Kd changes when selecting best parameters
    if iter > 0 && ~isempty(iteration_results)
        % Apply penalty to settling time based on Ki/Kd deviation from starting values
        penalized_results = iteration_results;
        for r = 1:size(penalized_results, 1)
            Ki_deviation = abs(penalized_results(r,2) - start_Ki) / start_Ki;
            Kd_deviation = abs(penalized_results(r,3) - start_Kd) / start_Kd;
            
            % Only apply penalties if the deviations are significant
            if Ki_deviation > 0.005 || Kd_deviation > 0.005
                % Apply a penalty factor to the settling time (makes it appear worse)
                penalty = 1 + 10 * (Ki_deviation + Kd_deviation); % 10x weight to penalize Ki/Kd changes
                penalized_settling = penalized_results(r,4) * penalty;
                penalized_results(r,4) = penalized_settling;
            end
        end
        
        % Find best parameters considering penalties
        [min_settling, min_idx] = min(penalized_results(:,4));
        
        % Only update best parameters if there's actual improvement
        if min_settling < best_settling * 1.001 % Allow small tolerance
            best_Kp = penalized_results(min_idx,1);
            best_Ki = penalized_results(min_idx,2);
            best_Kd = penalized_results(min_idx,3);
            best_settling = iteration_results(min_idx,4); % Use actual (not penalized) settling time
            best_overshoot = iteration_results(min_idx,5);
            best_sse = iteration_results(min_idx,6);
            improved = true;
        end
    elseif ~isempty(iteration_results)
        % For first iteration, use normal approach
        for r = 1:size(iteration_results, 1)
            if iteration_results(r,4) < best_settling
                best_Kp = iteration_results(r,1);
                best_Ki = iteration_results(r,2);
                best_Kd = iteration_results(r,3);
                best_settling = iteration_results(r,4);
                best_overshoot = iteration_results(r,5);
                best_sse = iteration_results(r,6);
                improved = true;
            end
        end
    end
    
    % Combine with all results
    all_results = [all_results; iteration_results];
    
    % Report iteration results
    disp(['Iteration ', num2str(iter), ' completed in ', num2str(toc), ' seconds.']);
    disp(['Valid parameter combinations found: ', num2str(valid_count)]);
    
    if improved
        disp('Improvement found!');
        disp(['New best parameters: Kp = ', num2str(best_Kp), ', Ki = ', num2str(best_Ki), ', Kd = ', num2str(best_Kd)]);
        disp(['Settling Time = ', num2str(best_settling), 's, Overshoot = ', num2str(best_overshoot), '%, SSE = ', num2str(best_sse), '%']);
        
        % Narrow search range around best parameters for next iteration
        % Keep a much wider range for Kp, extremely narrow for Ki and Kd
        Kp_range = [max(0.001, best_Kp * 0.7), min(1.0, best_Kp * 1.3)];   % 30% variation for Kp
        Ki_range = [max(0.001, best_Ki * 0.995), min(1.0, best_Ki * 1.005)]; % 0.5% variation for Ki
        Kd_range = [max(0.001, best_Kd * 0.995), min(0.02, best_Kd * 1.005)]; % 0.5% variation for Kd
    else
        disp('No improvement found in this iteration.');
        
        % Widen search range for next iteration, but almost exclusively for Kp
        Kp_range = [max(0.001, Kp_range(1) * 0.7), min(1.0, Kp_range(2) * 1.3)];  % 30% wider for Kp
        Ki_range = [max(0.001, Ki_range(1) * 0.998), min(1.0, Ki_range(2) * 1.002)]; % 0.2% wider for Ki
        Kd_range = [max(0.001, Kd_range(1) * 0.998), min(0.02, Kd_range(2) * 1.002)]; % 0.2% wider for Kd
    end
end

% Sort all results by settling time
if ~isempty(all_results)
    all_results = sortrows(all_results, 4);
    
    % Display results of valid combinations
    disp(['Found total of ', num2str(size(all_results, 1)), ' valid parameter combinations.']);
    disp('Top 5 parameter sets (sorted by settling time):');
    disp('   Kp      Ki      Kd    Settling Time  Overshoot   SSE(%)');
    disp('--------------------------------------------------------------');
    
    % Show top 5 or fewer results
    num_to_show = min(5, size(all_results, 1));
    for i = 1:num_to_show
        fprintf(' %6.5f  %6.5f  %6.5f      %6.5f      %6.2f%%   %6.2f%%\n', ...
            all_results(i,1), all_results(i,2), all_results(i,3), all_results(i,4), ...
            all_results(i,5), all_results(i,6));
    end
    
    % Confirm the best parameters
    best_Kp = all_results(1,1);
    best_Ki = all_results(1,2);
    best_Kd = all_results(1,3);
    best_settling = all_results(1,4);
    best_overshoot = all_results(1,5);
    best_sse = all_results(1,6);
    
    disp('=== Final Optimized PID Controller ===');
    disp(['Kp = ', num2str(best_Kp)]);
    disp(['Ki = ', num2str(best_Ki)]);
    disp(['Kd = ', num2str(best_Kd)]);
    disp(['Settling time: ', num2str(best_settling), 's']);
    disp(['Overshoot: ', num2str(best_overshoot), '%']);
    disp(['Steady-state error: ', num2str(best_sse), '%']);
    
    % Visualize parameter influence on settling time
    if size(all_results, 1) > 5
        figure(3);
        scatter3(all_results(:,1), all_results(:,2), all_results(:,3), 30, all_results(:,4), 'filled');
        xlabel('Kp');
        ylabel('Ki');
        zlabel('Kd');
        title('Parameter Space: Color = Settling Time');
        colormap(jet);
        colorbar;
        
        % Highlight the best point
        hold on;
        scatter3(best_Kp, best_Ki, best_Kd, 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
        hold off;
    end
else
    disp('No valid parameter combinations found!');
    % Revert to starting points
    best_Kp = start_Kp;
    best_Ki = start_Ki;
    best_Kd = start_Kd;
end

%% Root Locus Analysis of Final Design
% Create the optimal PID controller
PID_optimal = tf([best_Kd, best_Kp, best_Ki], [1, 0]);

% Create closed-loop transfer function
CL_PID = feedback(PID_optimal*G, 1);

% Compute all closed-loop poles 
cl_poles = pole(CL_PID);
disp('Closed-Loop Poles:');
disp(cl_poles);

% Find dominant poles (those closest to imaginary axis)
[~, idx] = sort(abs(real(cl_poles)));
dominant_poles = cl_poles(idx(1:min(2, length(cl_poles)))); 

% Plot pole-zero map
figure(4);
pzmap(CL_PID);
grid on;
title('Pole-Zero Map of Optimized Closed-Loop System');

% Add text only if we have valid dominant poles
if ~isempty(dominant_poles)
    text(real(dominant_poles(1)), imag(dominant_poles(1)), '  Dominant Pole', 'FontSize', 10);
    
    % Estimate performance from dominant poles
    dominant_real = real(dominant_poles(1));
    dominant_imag = imag(dominant_poles(1));
    damping_ratio = -dominant_real / sqrt(dominant_real^2 + dominant_imag^2);
    natural_freq = sqrt(dominant_real^2 + dominant_imag^2);
    est_settling_time = 4 / abs(dominant_real);
    est_overshoot = 100 * exp(-pi * damping_ratio / sqrt(1 - damping_ratio^2));
    
    disp('Dominant Closed-Loop Poles:');
    disp(dominant_poles);
    disp('Estimated Performance from Dominant Poles:');
    disp(['Damping Ratio = ', num2str(damping_ratio)]);
    disp(['Natural Frequency = ', num2str(natural_freq), ' rad/s']);
    disp(['Estimated Settling Time = ', num2str(est_settling_time), ' seconds']);
    disp(['Estimated Overshoot = ', num2str(est_overshoot), '%']);
end

%% Stability Analysis
% Plot root locus of the compensated system
figure(5);
rlocus(PID_optimal*G);
title('Root Locus of PID-Compensated System');
sgrid(0.6, 0);
grid on;

% Analyze the phase margin and gain margin for robustness
figure(6);
margin(PID_optimal*G);
grid on;
title('Bode Plot with Stability Margins');

% Get stability margins
[Gm, Pm, Wcg, Wcp] = margin(PID_optimal*G);
Gm_dB = 20*log10(Gm);

disp('Stability Margins:');
disp(['Gain Margin = ', num2str(Gm_dB), ' dB']);
disp(['Phase Margin = ', num2str(Pm), ' degrees']);

%% Verify Final Performance
% Simulate step response with reference input
figure(7);
opt = stepDataOptions('StepAmplitude', ref_input);
[y, t] = step(CL_PID, sim_time, opt);
plot(t, y, 'LineWidth', 2);
hold on;
plot([0 max(t)], [ref_input ref_input], 'r--');

% Add 2% error bounds
plot([0 max(t)], [ref_input*1.02 ref_input*1.02], 'g:', 'LineWidth', 1);
plot([0 max(t)], [ref_input*0.98 ref_input*0.98], 'g:', 'LineWidth', 1);

grid on;
title(sprintf('Optimized Step Response (Kp=%.5f, Ki=%.5f, Kd=%.5f)', best_Kp, best_Ki, best_Kd));
xlabel('Time (seconds)');
ylabel('Amplitude (encoder counts)');

% Calculate final performance metrics
final_value = mean(y(max(1, end-10):end));
steady_state_error = abs(ref_input - final_value) / ref_input * 100;
max_value = max(y);
overshoot = max(0, (max_value - final_value) / final_value * 100);

% Calculate settling time (2% criterion)
settle_band = 0.02 * final_value;
settled_indices = find(abs(y - final_value) <= settle_band);

if ~isempty(settled_indices)
    settling_time = inf;
    for idx = 1:length(settled_indices)-1
        if all(diff(settled_indices(idx:end)) == 1)
            settling_time = t(settled_indices(idx));
            break;
        end
    end
else
    settling_time = inf;
end

% Calculate rise time (10% to 90%)
rise_indices_10 = find(y >= 0.1*final_value, 1);
rise_indices_90 = find(y >= 0.9*final_value, 1);

if ~isempty(rise_indices_10) && ~isempty(rise_indices_90)
    rise_time = t(rise_indices_90) - t(rise_indices_10);
else
    rise_time = NaN;
end

disp('Actual Performance from Simulation:');
disp(['Settling Time = ', num2str(settling_time), ' seconds']);
disp(['Overshoot = ', num2str(overshoot), '%']);
disp(['Rise Time = ', num2str(rise_time), ' seconds']);
disp(['Steady-State Error = ', num2str(steady_state_error), '%']);

% Check if all constraints are met
constraints_met = (best_Kp <= 1) && (best_Kp > 0) && ...
                  (best_Kd <= 0.02) && (best_Kd > 0) && ...
                  (best_Ki <= 1) && (best_Ki > 0) && ...
                  (overshoot < 25) && (steady_state_error < 2);
                 
if constraints_met
    constraint_message = 'All constraints are met!';
else
    constraint_message = 'Warning: Not all constraints are met!';
end

disp(constraint_message);

% Add vertical line at settling time
if isfinite(settling_time)
    settled_idx = find(t >= settling_time, 1);
    if ~isempty(settled_idx)
        settled_value = y(settled_idx);
        plot([settling_time settling_time], [0 settled_value], 'm-.', 'LineWidth', 1.5);
        text(settling_time, settled_value/2, sprintf('  Ts = %.3fs', settling_time), 'Color', 'm');
    end
end

% Mark peak point for overshoot
if overshoot > 0
    [peak_value, peak_idx] = max(y);
    peak_time = t(peak_idx);
    plot(peak_time, peak_value, 'ro', 'MarkerSize', 6);
    text(peak_time, peak_value*1.05, sprintf('  OS = %.2f%%', overshoot), 'Color', 'r');
end

% Annotate the plot with performance metrics
text(0.6*max(t), 0.4*ref_input, sprintf('Settling Time: %.3f s\nOvershoot: %.2f%%\nSteady-State Error: %.2f%%\n%s', ...
    settling_time, overshoot, steady_state_error, constraint_message), ...
    'FontSize', 10, 'BackgroundColor', 'w');

legend('System Response', 'Reference', 'Upper 2% Bound', 'Lower 2% Bound', 'Settling Time', 'Peak');

%% Compare with Starting Parameters
% Create PID controller with starting values
Start_PID = tf([start_Kd, start_Kp, start_Ki], [1, 0]);
Start_CL = feedback(Start_PID*G, 1);

% Compare step responses
figure(8);
[y_opt, t_opt] = step(CL_PID, sim_time, opt);
[y_start, t_start] = step(Start_CL, sim_time, opt);

plot(t_opt, y_opt, 'b-', 'LineWidth', 2);
hold on;
plot(t_start, y_start, 'r--', 'LineWidth', 2);
plot([0 max([max(t_opt); max(t_start)])], [ref_input ref_input], 'k:', 'LineWidth', 1);
grid on;
title('Comparison: Optimized vs. Starting PID');
xlabel('Time (seconds)');
ylabel('Amplitude (encoder counts)');
legend('Optimized PID', 'Starting PID', 'Reference');

%% Final Summary
% Create final PID controller in standard form
final_PID = pid(best_Kp, best_Ki, best_Kd);

% Display the final optimized PID controller
disp('==== Final Optimized PID Controller ====');
disp(final_PID);

% Final constraint check
disp('Constraint Check:');
disp(['Kp = ', num2str(best_Kp), ' (constraint: 0 < Kp ≤ 1) => ', check_constraint(best_Kp > 0 && best_Kp <= 1)]);
disp(['Ki = ', num2str(best_Ki), ' (constraint: 0 < Ki ≤ 1) => ', check_constraint(best_Ki > 0 && best_Ki <= 1)]);
disp(['Kd = ', num2str(best_Kd), ' (constraint: 0 < Kd ≤ 0.02) => ', check_constraint(best_Kd > 0 && best_Kd <= 0.02)]);
disp(['Overshoot = ', num2str(overshoot), '% (constraint: < 25%) => ', check_constraint(overshoot < 25)]);
disp(['SSE = ', num2str(steady_state_error), '% (constraint: < 2%) => ', check_constraint(steady_state_error < 2)]);
disp(['SUMMARY: This controller is ', check_constraint(constraints_met, 'VALID - All constraints satisfied', 'INVALID - Constraints not met')]);

% Display improvement from starting point and parameter changes
if isfinite(settling_time) && isfinite(start_settling)
    improvement = (start_settling - settling_time) / start_settling * 100;
    disp(['Settling Time Improvement: ', num2str(improvement), '%']);
    
    % Calculate parameter changes
    Kp_change_pct = abs(best_Kp - start_Kp) / start_Kp * 100;
    Ki_change_pct = abs(best_Ki - start_Ki) / start_Ki * 100;
    Kd_change_pct = abs(best_Kd - start_Kd) / start_Kd * 100;
    
    disp('Parameter Changes from Starting Point:');
    disp(['Kp change: ', num2str(Kp_change_pct), '% (', num2str(start_Kp), ' -> ', num2str(best_Kp), ')']);
    disp(['Ki change: ', num2str(Ki_change_pct), '% (', num2str(start_Ki), ' -> ', num2str(best_Ki), ')']);
    disp(['Kd change: ', num2str(Kd_change_pct), '% (', num2str(start_Kd), ' -> ', num2str(best_Kd), ')']);
end

% Save the optimized parameters to a file for future use
save('optimized_pid_params.mat', 'best_Kp', 'best_Ki', 'best_Kd', 'settling_time', 'overshoot', 'steady_state_error');

% Helper function for constraint checking
function result = check_constraint(condition, true_result, false_result)
    if nargin < 3
        true_result = 'OK';
        false_result = 'VIOLATED';
    end
    
    if condition
        result = true_result;
    else
        result = false_result;
    end
end