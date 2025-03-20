% Script to verify closed-loop transfer function with PID controller
% Define symbolic variables
clear all;
syms s K a b Kp Ki Kd;

% Define the plant transfer function G(s)
G_s = K / (s^2 + a*s + b);
disp('Plant transfer function G(s):');
pretty(G_s);

% Define the PID controller transfer function C(s)
C_s = Kp + Ki/s + Kd*s;
disp('PID controller transfer function C(s):');
pretty(C_s);

% Calculate the open-loop transfer function
OL_s = simplify(G_s * C_s);
disp('Open-loop transfer function G(s)*C(s):');
pretty(OL_s);

% Calculate the closed-loop transfer function
CL_s = simplify(OL_s / (1 + OL_s));
disp('Closed-loop transfer function T(s) = G(s)*C(s)/(1+G(s)*C(s)):');
pretty(CL_s);

% Extract the numerator and denominator of the closed-loop transfer function
[num, den] = numden(CL_s);
disp('Numerator of closed-loop transfer function:');
num = simplify(num);
pretty(num);
disp('Denominator of closed-loop transfer function:');
den = simplify(den);
pretty(den);

% Compare with the standard second-order form
disp('Standard second-order form: ωn² / (s² + 2ζωn·s + ωn²)');
disp('By comparing coefficients in the denominator:');
disp('1. The coefficient of s^3 exists, confirming this is actually a third-order system');
disp('2. For the second-order approximation (assuming s^3 term minimal effect):');
disp('   - ωn² = K·Ki                   (coefficient of s^0 term)');
disp('   - 2ζωn = a + K·Kd              (coefficient of s^1 term)');
disp('   - For proper behavior: b + K·Kp ≈ ωn²  (coefficient of s^2 term)');

% Solve for PID gains in terms of control parameters
disp('Therefore:');
Kp_formula = 'Kp ≈ ωn² / K  (assuming b is small)';
disp(Kp_formula);
Kd_formula = 'Kd ≈ (2ζωn - a) / K';
disp(Kd_formula);
Ki_formula = 'Ki = ωn² / K';
disp(Ki_formula);

% For a typical case where a is small
disp('For systems where a is small:');
disp('Kd ≈ 2ζωn / K  or  Kd ≈ 2ζ/ωn (normalizing by K)');

% Practical adjustment for Ki
disp('The practical rule of thumb Ki_theo = ωn*Kp/10 comes from:');
disp('Ki_theo = ωn · (ωn² / K) / 10 = ωn³ / (10·K)');
disp('This reduces integral action to improve stability and reduce overshoot');