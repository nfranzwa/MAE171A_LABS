% Clear workspace, figures, and command window
clear all;
close all;
clc;

% Load the transfer function from G.mat
try
    load('G.mat');
    disp('Transfer Function:');
    G
catch
    error('Error: G.mat file must be present with the transfer function.')
end

% System Analysis: Extract key information from the transfer function
[num, den] = tfdata(G, 'v');
disp('System analysis:');

% Calculate DC gain - important for steady-state error
dcGain = dcgain(G);
disp(['DC gain (K): ', num2str(dcGain)]);

% Extract poles and zeros
zeros = roots(num);
poles = roots(den);
disp('System zeros:');
disp(zeros);
disp('System poles:');
disp(poles);

% Check system stability
if any(real(poles) >= 0)
    disp('WARNING: Open-loop system is unstable or marginally stable');
else
    disp('Open-loop system is stable');
end

% Set system time constant (tau)
% Using the specified settling/reaction time
tau = 2;  % Time constant = 2 seconds
disp(['Time constant (τ): ', num2str(tau), ' seconds']);

% Step 1: Define performance requirements
maxOvershoot = 25;      % Maximum overshoot <= 25%
settlingTime = 2;       % Settling time = 2 seconds

% Step 2: Calculate damping ratio and natural frequency from requirements
% Calculate damping ratio from overshoot specification
% Maximum overshoot formula: Mp = exp(-pi*zeta/sqrt(1-zeta^2))
% Solved for zeta: zeta = -ln(Mp/100)/sqrt(pi^2 + ln(Mp/100)^2)
Mp = maxOvershoot/100;
lnMp = log(Mp);
damping = -lnMp/sqrt(pi^2 + lnMp^2);
disp(['Calculated damping ratio (ζ): ', num2str(damping)]);

% Calculate natural frequency from settling time specification
% For 2% settling time: ts ≈ 4/(ζ*ωn)
naturalFreq = 4/(damping * settlingTime);
disp(['Calculated natural frequency (ωn): ', num2str(naturalFreq), ' rad/s (', num2str(naturalFreq/(2*pi)), ' Hz)']);

% Step 3: Derive PID parameters using pole placement approach based on PDF article
disp('--------------------------------------------------------');
disp('PID Parameter Derivation (First-Order System Method):');

% Using the formulas from the PDF:
% Ki = ωn²τ/K
% Kp = (2ζωnτ - 1)/K

% Step 3a: Derive Integral Gain (Ki)
Ki_theo = (naturalFreq^2 * tau) / dcGain;
disp(['Theoretical Ki (unconstrained): ', num2str(Ki_theo)]);

% Apply constraint: 0 < Ki < 1
Ki = min(Ki_theo, 1);
if Ki ~= Ki_theo
    disp(['Ki constrained to: ', num2str(Ki), ' (original: ', num2str(Ki_theo), ')']);
else
    disp(['Final Ki: ', num2str(Ki)]);
end

% Step 3b: Derive Proportional Gain (Kp)
Kp_theo = (2 * damping * naturalFreq * tau - 1) / dcGain;
disp(['Theoretical Kp (unconstrained): ', num2str(Kp_theo)]);

% Apply constraint: 0 < Kp < 1
Kp = min(max(0, Kp_theo), 1);
if Kp ~= Kp_theo
    disp(['Kp constrained to: ', num2str(Kp), ' (original: ', num2str(Kp_theo), ')']);
else
    disp(['Final Kp: ', num2str(Kp)]);
end

% Step 3c: Derive Derivative Gain (Kd)
% Kd primarily affects damping ratio
% Based on the relationship: 2*ζ*ωn ≈ Kd*B (B is system gain parameter)
% For standard 2nd order systems: Kd ≈ 2*ζ/ωn
Kd_theo = 2*damping/naturalFreq;
disp(['Theoretical Kd (unconstrained): ', num2str(Kd_theo)]);

% Apply constraint: 0 < Kd < 0.02
Kd = min(Kd_theo, 0.02);
if Kd ~= Kd_theo
    disp(['Kd constrained to: ', num2str(Kd), ' (original: ', num2str(Kd_theo), ')']);
else
    disp(['Final Kd: ', num2str(Kd)]);
end

% Step 4: Calculate final theoretically-derived PID parameters
disp('--------------------------------------------------------');
disp('Final Theoretically-Derived PID Parameters:');
disp(['Kp = ', num2str(Kp)]);
disp(['Ki = ', num2str(Ki)]);
disp(['Kd = ', num2str(Kd)]);

% Create PID controller with theoretical parameters
C_theoretical = pid(Kp, Ki, Kd);

% Analyze closed-loop system with theoretical PID values
CL_theoretical = feedback(C_theoretical*G, 1);

% Plot step response
figure;
step(CL_theoretical);
title('Step Response with Theoretically-Derived PID Parameters');
grid on;

% Calculate performance metrics
info = stepinfo(CL_theoretical);
disp('Predicted Performance:');
disp(['Overshoot: ', num2str(info.Overshoot), '%']);
disp(['Settling Time: ', num2str(info.SettlingTime), ' seconds']);
disp(['Rise Time: ', num2str(info.RiseTime), ' seconds']);

% Calculate steady-state error
[y, t] = step(CL_theoretical);
steadyStateValue = y(end);
steadyStateErrorPct = abs(1 - steadyStateValue) * 100;
disp(['Steady-state Error: ', num2str(steadyStateErrorPct), '%']);

% Just analyze the single set of parameters
disp('--------------------------------------------------------');
disp('System Analysis with Selected Parameters:');

% Create a figure for the step response
figure;

% Plot step response of the closed-loop system
step(CL_theoretical, t);
title(['Step Response with ωn = ', num2str(naturalFreq), ' rad/s, ζ = ', num2str(damping)]);
grid on;

% Print a summary of the selected parameters and performance
disp(['Selected Parameters:']);
disp(['  ωn = ', num2str(naturalFreq), ' rad/s (', num2str(naturalFreq/(2*pi)), ' Hz)']);
disp(['  ζ = ', num2str(damping)]);
disp(['  Kp = ', num2str(Kp)]);
disp(['  Ki = ', num2str(Ki)]);
disp(['  Kd = ', num2str(Kd)]);
disp(['Performance Metrics:']);
disp(['  Overshoot: ', num2str(info.Overshoot), '%']);
disp(['  Settling Time: ', num2str(info.SettlingTime), ' seconds']);
disp(['  Rise Time: ', num2str(info.RiseTime), ' seconds']);
disp(['  Steady-state Error: ', num2str(steadyStateErrorPct), '%']);

% Print recommendations for iterative tuning
disp('--------------------------------------------------------');
disp('Recommendations for Iterative Tuning:');
disp('1. Start with these theoretical parameters as initial values');
disp('2. If overshoot is too high: decrease Kp, increase ζ, or try a higher Kd');
disp('3. If settling time is too long: increase ωn carefully');
disp('4. If steady-state error is too high: increase Ki');
disp('5. Remember that changes to Kp and Ki will affect both damping and frequency response');
disp('6. The most sensitive parameter is likely Kd given its tight constraint (< 0.02)');
disp('--------------------------------------------------------');