%% MAE 171A Combined Stress-Strain Plot
% Script to create a single combined plot of all stress-strain data
% Created on Mar 03, 2025

% This code assumes you've already run the data loading and processing part
% of your original script or that those variables are available in your workspace

%% Create combined plot with all three tests
figure('Position', [100, 100, 800, 600])

% Plot all three datasets
plot(strain1_1, stress1_1, 'b-', 'LineWidth', 2);
hold on;
plot(strain1_2, stress1_2, 'g-', 'LineWidth', 2);
plot(strain2, stress2, 'r-', 'LineWidth', 2);
hold off;

% Add labels and title with LaTeX formatting
xlabel('$\mathrm{Strain}~(\epsilon)$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{[MPa]}$', 'Interpreter', 'latex', 'FontSize', 14);
title('$\mathrm{PMMA~Mechanical~Tests:~Stress~vs.~Strain}$', 'Interpreter', 'latex', 'FontSize', 16);

% Add legend
legend('Test 1 (File 1)', 'Test 1 (File 2)', 'Test 1 (File 3)', 'Location', 'best', 'Interpreter', 'latex');

% Customize grid and appearance
grid on;
set(gca, 'FontSize', 12);
box on;

% Optional: Set axis limits if you want to focus on a specific range
% xlim([0 0.2]);  % Limit x-axis from 0 to 0.2 strain
 ylim([0 2.7]);  % Limit y-axis from 0 to 100 MPa

% Optional: Add annotations for key points
% text(0.01, 60, 'Elastic Region', 'FontSize', 10);
% text(0.05, 80, 'Ultimate Strength', 'FontSize', 10);

% Save figure
saveas(gcf, 'all_tests_combined_stress_strain.png');
saveas(gcf, 'all_tests_combined_stress_strain.fig');

% Display material properties in the command window for reference
fprintf('\nMaterial Properties Summary:\n');
fprintf('---------------------------\n');
fprintf('Test 1 (File 1): E = %.2f GPa, UTS = %.2f MPa, Max Strain = %.2f%%\n', ...
    E1_1/1000, max_stress1_1, max_strain1_1*100);
fprintf('Test 1 (File 2): E = %.2f GPa, UTS = %.2f MPa, Max Strain = %.2f%%\n', ...
    E1_2/1000, max_stress1_2, max_strain1_2*100);
fprintf('Test 2: E = %.2f GPa, UTS = %.2f MPa, Max Strain = %.2f%%\n', ...
    E2/1000, max_stress2, max_strain2*100);