%% MAE 171A Test Data Analysis and Plotting with Slack Correction
% Script to calculate and plot stress-strain curves from test data
% This version includes corrections for slack in the tensile tester jaws
% Created on Feb 27, 2025

clear all; close all; clc;

% Set default text interpreter to LaTeX for all plots
set(0, 'defaultTextInterpreter', 'latex');
set(0, 'defaultAxesTickLabelInterpreter', 'latex');
set(0, 'defaultLegendInterpreter', 'latex');
set(0, 'defaultColorbarTickLabelInterpreter', 'latex');

%% Load and process data from all test files
% File paths
test1_file1 = 'test 1 W25 MAE 171A_20250207_132831_1.csv';
test1_file2 = 'test 1 W25 MAE 171A_20250207_131243_1.csv';
test2_file = 'test 2 W25 MAE 171A_20250207_134054_1_1.csv';

% ASTM standard dogbone dimensions (from lab handout)
width = 13;          % mm - width of narrow section
thickness = 3.2;     % mm - thickness
gaugeLength = 50;    % mm - gauge length
crossSectionArea = width * thickness;  % mm²

% Function to read and process CSV files with custom formatting
function [time, displacement, force] = readTestData(filename)
    % Read the file as text
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file: %s', filename);
    end
    
    % Read all lines
    allLines = cell(1000, 1); % Pre-allocate
    lineCount = 0;
    
    while ~feof(fid)
        lineCount = lineCount + 1;
        allLines{lineCount} = fgetl(fid);
    end
    
    allLines = allLines(1:lineCount); % Trim to actual size
    fclose(fid);
    
    % Find the header line (contains Time, Displacement, Force)
    headerLineIdx = 0;
    for i = 1:lineCount
        if contains(allLines{i}, 'Time') && contains(allLines{i}, 'Displacement') && contains(allLines{i}, 'Force')
            headerLineIdx = i;
            break;
        end
    end
    
    if headerLineIdx == 0
        error('Could not find header line in file: %s', filename);
    end
    
    % Find where the data starts (skip units line)
    dataStartIdx = headerLineIdx + 2;
    
    % Extract data
    time = [];
    displacement = [];
    force = [];
    
    for i = dataStartIdx:lineCount
        line = allLines{i};
        if isempty(line) || all(isspace(line))
            continue;
        end
        
        % Replace quotes and split by comma
        line = strrep(line, '"', '');
        parts = strsplit(line, ',');
        
        % Skip empty parts
        nonEmptyParts = parts(~cellfun(@isempty, strtrim(parts)));
        
        if length(nonEmptyParts) >= 3
            % Handle different file formats - some have an extra empty column at start
            if length(nonEmptyParts) >= 4 && isempty(strtrim(nonEmptyParts{1}))
                time(end+1) = str2double(nonEmptyParts{2});
                displacement(end+1) = str2double(nonEmptyParts{3});
                force(end+1) = str2double(nonEmptyParts{4});
            else
                time(end+1) = str2double(nonEmptyParts{1});
                displacement(end+1) = str2double(nonEmptyParts{2});
                force(end+1) = str2double(nonEmptyParts{3});
            end
        end
    end
    
    % Convert to column vectors
    time = time(:);
    displacement = displacement(:);
    force = force(:);
    
    % Check for kN in Force units and convert to N if needed
    unitsLine = allLines{headerLineIdx + 1};
    if contains(unitsLine, 'kN')
        fprintf('Converting force from kN to N for file: %s\n', filename);
        force = force * 1000; % Convert kN to N
    end
end

% Read all test data
fprintf('Reading Test 1 File 1: %s\n', test1_file1);
[time1_1, displacement1_1, force1_1] = readTestData(test1_file1);

fprintf('Reading Test 1 File 2: %s\n', test1_file2);
[time1_2, displacement1_2, force1_2] = readTestData(test1_file2);

fprintf('Reading Test 2: %s\n', test2_file);
[time2, displacement2, force2] = readTestData(test2_file);

%% Calculate stress and strain
% For Test 1 File 1
stress1_1 = force1_1 / crossSectionArea;  % Stress in MPa (N/mm²)
strain1_1 = displacement1_1 / gaugeLength;  % Engineering strain (mm/mm)

% For Test 1 File 2
stress1_2 = force1_2 / crossSectionArea;  % Stress in MPa (N/mm²)
strain1_2 = displacement1_2 / gaugeLength;  % Engineering strain (mm/mm)

% For Test 2
stress2 = force2 / crossSectionArea;  % Stress in MPa (N/mm²)
strain2 = displacement2 / gaugeLength;  % Engineering strain (mm/mm)

%% Function to find the end of the plateau (slack) region
function plateauEndIndex = findPlateauEnd(strain, stress)
    % Calculate derivatives (stress-strain slope)
    derivatives = diff(stress) ./ diff(strain);
    
    % Calculate threshold based on initial derivatives
    % For a plateau, the initial derivatives should be very low
    avgInitialDerivative = mean(derivatives(1:min(20, length(derivatives))));
    
    % Find where derivative exceeds threshold (transition from slack to actual loading)
    % Adjust this multiplier based on your data characteristics
    threshold = avgInitialDerivative * 10;
    
    % Find first point where derivative exceeds threshold
    indices = find(derivatives > threshold);
    
    if ~isempty(indices)
        plateauEndIndex = indices(1) + 1; % +1 because derivatives array is one shorter
    else
        % Default if no clear transition is found
        plateauEndIndex = floor(length(strain) * 0.1); % Default to 10% of the data
    end
end

%% Function to shift strain data to account for slack
function shiftedStrain = shiftStrainData(strain, stress, targetStrain)
    % Find plateau end
    plateauEndIndex = findPlateauEnd(strain, stress);
    
    % Get strain at plateau end
    plateauEndStrain = strain(plateauEndIndex);
    
    % Calculate shift amount
    shiftAmount = targetStrain - plateauEndStrain;
    
    % Apply shift
    shiftedStrain = strain + shiftAmount;
    
    % Report on what was done
    fprintf('Plateau detected at index %d (strain = %.6f)\n', plateauEndIndex, plateauEndStrain);
    fprintf('Applied strain shift of %.6f to align with target strain %.6f\n', shiftAmount, targetStrain);
end

%% Apply slack correction to shift strain data
% Target strain is the requested value of 0.006632 for the transition point
targetStrain = 0.006632;

% Shift all strain datasets
fprintf('\nApplying slack correction for Test 1 File 1:\n');
strain1_1_shifted = shiftStrainData(strain1_1, stress1_1, targetStrain);

fprintf('\nApplying slack correction for Test 1 File 2:\n');
strain1_2_shifted = shiftStrainData(strain1_2, stress1_2, targetStrain);

fprintf('\nApplying slack correction for Test 2:\n');
strain2_shifted = shiftStrainData(strain2, stress2, targetStrain);

%% Create Stress-Strain plot for both Test 1 files on the same plot with corrected data
figure('Position', [100, 100, 800, 600])
plot(strain1_1_shifted, stress1_1, 'b-', 'LineWidth', 2);
hold on;
plot(strain1_2_shifted, stress1_2, 'g-', 'LineWidth', 2);
hold off;

% Add labels and title with LaTeX formatting
xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 14);
ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 14);
title('$\mathrm{Test~1:~Corrected~Stress-Strain~Curve~Comparison}$', 'FontSize', 16);

% Add legend
legend('Test 1 (File 1)', 'Test 1 (File 2)', 'Location', 'northwest');

% Customize grid
grid on;
set(gca, 'FontSize', 12);

% Save figure
saveas(gcf, 'test1_combined_stress_strain_corrected.png');
saveas(gcf, 'test1_combined_stress_strain_corrected.fig');

%% Create Stress-Strain plot for Test 2 with corrected data
figure('Position', [100, 100, 800, 600])
plot(strain2_shifted, stress2, 'r-', 'LineWidth', 2);

% Add labels and title with LaTeX formatting
xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 14);
ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 14);
title('$\mathrm{Test~2:~Corrected~Stress-Strain~Curve}$', 'FontSize', 16);

% Customize grid
grid on;
set(gca, 'FontSize', 12);

% Save figure
saveas(gcf, 'test2_stress_strain_corrected.png');
saveas(gcf, 'test2_stress_strain_corrected.fig');

%% Calculate and display material properties with corrected strain data
% Function to calculate elastic modulus from stress-strain data
function [E, yieldStress, ultimateStress] = calculateProperties(stress, strain)
    % Skip the initial error blip by finding where the stress exceeds a minimum threshold
    minStressThreshold = 5; % MPa - skip the initial error blip
    startIndex = find(stress > minStressThreshold, 1, 'first');
    
    if isempty(startIndex) || startIndex < 2
        startIndex = 2; % Default to second point if can't find a clear start
    end
    
    % Find a suitable end point for the elastic region
    % For PMMA, the linear elastic region typically extends to about 25-30 MPa
    endElasticThreshold = 25; % MPa
    endIndex = find(stress > endElasticThreshold, 1, 'first');
    
    if isempty(endIndex) || endIndex <= startIndex + 5
        endIndex = startIndex + floor((length(strain) - startIndex) * 0.2); % Use 20% of remaining data
    end
    
    % Make sure endIndex is within bounds
    endIndex = min(endIndex, length(strain));
    
    % Calculate the slope of the linear portion (elastic modulus)
    elasticStrain = strain(startIndex:endIndex);
    elasticStress = stress(startIndex:endIndex);
    p = polyfit(elasticStrain, elasticStress, 1);
    E = p(1);  % Elastic modulus in MPa
    
    % 0.2% offset yield strength calculation
    offset = 0.002;  % 0.2% strain offset
    yieldLine = @(x) E * (x - offset);  % Offset line parallel to elastic region
    
    % Find where the offset line intersects the stress-strain curve
    diff = abs(stress - yieldLine(strain));
    [~, yieldIndex] = min(diff);
    yieldStress = stress(yieldIndex);
    
    % Ultimate strength is the maximum stress
    ultimateStress = max(stress);
    
    % Create a diagnostic plot to visualize the elastic region selection
    figure('Position', [100, 100, 600, 400]);
    plot(strain, stress, 'b-', 'LineWidth', 1);
    hold on;
    
    % Plot the selected elastic region
    plot(elasticStrain, elasticStress, 'r-', 'LineWidth', 2);
    
    % Plot the fitted line
    elasticLine = @(x) p(1) * x + p(2);
    x_range = linspace(min(strain), max(strain), 100);
    plot(x_range, elasticLine(x_range), 'g--', 'LineWidth', 1.5);
    
    % Plot the offset line for yield strength calculation
    plot(x_range, yieldLine(x_range), 'k--', 'LineWidth', 1.5);
    
    % Mark the yield point
    plot(strain(yieldIndex), stress(yieldIndex), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    
    % Add labels and title
    xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 12);
    ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 12);
    title('$\mathrm{Elastic~Modulus~and~Yield~Strength~Calculation}$', 'FontSize', 14);
    
    % Add legend
    legend('Full Curve', 'Selected Elastic Region', 'Elastic Fit Line', '0.2\% Offset Line', 'Yield Point');
    
    % Customize grid
    grid on;
end

% Calculate properties for Test 1 File 1 with corrected strain data
[E1_1, yieldStress1_1, ultimateStress1_1] = calculateProperties(stress1_1, strain1_1_shifted);
fprintf('\nTest 1 (File 1) Material Properties (Corrected Data):\n');
fprintf('Elastic Modulus: %.2f MPa\n', E1_1);
fprintf('0.2%% Offset Yield Strength: %.2f MPa\n', yieldStress1_1);
fprintf('Ultimate Strength: %.2f MPa\n', ultimateStress1_1);

% Calculate properties for Test 1 File 2 with corrected strain data
[E1_2, yieldStress1_2, ultimateStress1_2] = calculateProperties(stress1_2, strain1_2_shifted);
fprintf('\nTest 1 (File 2) Material Properties (Corrected Data):\n');
fprintf('Elastic Modulus: %.2f MPa\n', E1_2);
fprintf('0.2%% Offset Yield Strength: %.2f MPa\n', yieldStress1_2);
fprintf('Ultimate Strength: %.2f MPa\n', ultimateStress1_2);

% Calculate properties for Test 2 with corrected strain data
[E2, yieldStress2, ultimateStress2] = calculateProperties(stress2, strain2_shifted);
fprintf('\nTest 2 Material Properties (Corrected Data):\n');
fprintf('Elastic Modulus: %.2f MPa\n', E2);
fprintf('0.2%% Offset Yield Strength: %.2f MPa\n', yieldStress2);
fprintf('Ultimate Strength: %.2f MPa\n', ultimateStress2);

% Calculate average properties and compare with expected values
avgE = (E1_1 + E1_2 + E2) / 3;
avgYield = (yieldStress1_1 + yieldStress1_2 + yieldStress2) / 3;
avgUltimate = (ultimateStress1_1 + ultimateStress1_2 + ultimateStress2) / 3;

fprintf('\nAverage Material Properties (Corrected Data):\n');
fprintf('Average Elastic Modulus: %.2f MPa\n', avgE);
fprintf('Average Yield Strength: %.2f MPa\n', avgYield);
fprintf('Average Ultimate Strength: %.2f MPa\n', avgUltimate);

% Comparison with expected values from the lab handout
fprintf('\nComparison with expected values from lab handout:\n');
fprintf('Expected Elastic Modulus: 2.56 GPa (2560 MPa)\n');
fprintf('Expected Yield Strength: 68.9 MPa\n');
fprintf('Expected Ultimate Strength: 77.7 MPa\n');

% Calculate percentage differences
fprintf('\nPercentage Differences from Expected Values (Corrected Data):\n');
fprintf('Elastic Modulus: %.2f%%\n', (avgE - 2560)/2560 * 100);
fprintf('Yield Strength: %.2f%%\n', (avgYield - 68.9)/68.9 * 100);
fprintf('Ultimate Strength: %.2f%%\n', (avgUltimate - 77.7)/77.7 * 100);

%% Create comparison plots showing before and after slack correction
% Test 1 File 1 Comparison
figure('Position', [100, 100, 800, 600]);
plot(strain1_1, stress1_1, 'b--', 'LineWidth', 1.5);
hold on;
plot(strain1_1_shifted, stress1_1, 'b-', 'LineWidth', 2);
hold off;
xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 14);
ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 14);
title('$\mathrm{Test~1~File~1:~Before~and~After~Slack~Correction}$', 'FontSize', 16);
legend('Original Data', 'Corrected Data', 'Location', 'northwest');
grid on;
saveas(gcf, 'test1_file1_correction_comparison.png');

% Test 1 File 2 Comparison
figure('Position', [100, 100, 800, 600]);
plot(strain1_2, stress1_2, 'g--', 'LineWidth', 1.5);
hold on;
plot(strain1_2_shifted, stress1_2, 'g-', 'LineWidth', 2);
hold off;
xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 14);
ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 14);
title('$\mathrm{Test~1~File~2:~Before~and~After~Slack~Correction}$', 'FontSize', 16);
legend('Original Data', 'Corrected Data', 'Location', 'northwest');
grid on;
saveas(gcf, 'test1_file2_correction_comparison.png');

% Test 2 Comparison
figure('Position', [100, 100, 800, 600]);
plot(strain2, stress2, 'r--', 'LineWidth', 1.5);
hold on;
plot(strain2_shifted, stress2, 'r-', 'LineWidth', 2);
hold off;
xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 14);
ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 14);
title('$\mathrm{Test~2:~Before~and~After~Slack~Correction}$', 'FontSize', 16);
legend('Original Data', 'Corrected Data', 'Location', 'northwest');
grid on;
saveas(gcf, 'test2_correction_comparison.png');

%% Add code to calculate Poisson's ratio if DIC data is available
% This section would process DIC data to extract transverse and longitudinal strains
% and calculate Poisson's ratio

% Example placeholder for Poisson's ratio calculation (uncomment and modify when DIC data is available):
%
% % After processing DIC data to get transverse and longitudinal strains
% % Extract data for the elastic region only
% transverseStrain = ...; % From DIC analysis
% longitudinalStrain = ...; % From DIC analysis
%
% % Linear fit to find Poisson's ratio
% p = polyfit(longitudinalStrain, transverseStrain, 1);
% poissonsRatio = -p(1);
%
% fprintf('Poisson''s ratio: %.4f\n', poissonsRatio);
% fprintf('Expected Poisson''s ratio (from lab handout): 0.370-0.430 (avg: 0.402)\n');
%
% % Plot for visualization
% figure('Position', [100, 100, 600, 400]);
% plot(longitudinalStrain, transverseStrain, 'bo');
% hold on;
% x_range = linspace(min(longitudinalStrain), max(longitudinalStrain), 100);
% plot(x_range, p(1)*x_range + p(2), 'r-', 'LineWidth', 2);
% xlabel('$\mathrm{Longitudinal~Strain}~(\varepsilon_y)$', 'FontSize', 14);
% ylabel('$\mathrm{Transverse~Strain}~(\varepsilon_x)$', 'FontSize', 14);
% title(sprintf('$\\mathrm{Poisson''s~Ratio~Determination:}~\\nu = %.4f$', poissonsRatio), 'FontSize', 16);
% grid on;
% saveas(gcf, 'poissons_ratio_calculation.png');

%% Calculate Strain at Ultimate Tensile Strength (UTS)
% Function to find the strain at UTS
function strainAtUTS = findStrainAtUTS(stress, strain)
    % Find the maximum stress value (UTS)
    [ultimateStress, utsIndex] = max(stress);
    
    % Get the corresponding strain value
    strainAtUTS = strain(utsIndex);
    
    % Create a diagnostic plot
    figure('Position', [100, 100, 600, 400]);
    plot(strain, stress, 'b-', 'LineWidth', 1.5);
    hold on;
    
    % Mark the UTS point
    plot(strainAtUTS, ultimateStress, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    
    % Add labels and title
    xlabel('$\mathrm{Strain}~(\varepsilon)$', 'FontSize', 12);
    ylabel('$\mathrm{Stress}~(\sigma)~\mathrm{MPa}$', 'FontSize', 12);
    title('$\mathrm{Strain~at~Ultimate~Tensile~Strength~(UTS)}$', 'FontSize', 14);
    text(strainAtUTS*1.05, ultimateStress*0.95, ...
        ['$\varepsilon_{UTS} = ', num2str(strainAtUTS, '%.4f'), '$'], ...
        'FontSize', 12);
    
    % Customize grid
    grid on;
    
    % Save figure
    saveas(gcf, ['strain_at_uts_plot_', num2str(ultimateStress, '%.0f'), 'MPa.png']);
end

% Calculate strain at UTS for Test 1 File 1
strainAtUTS1_1 = findStrainAtUTS(stress1_1, strain1_1_shifted);
fprintf('\nTest 1 (File 1) Strain at UTS:\n');
fprintf('Ultimate Strength: %.2f MPa\n', ultimateStress1_1);
fprintf('Strain at UTS: %.4f\n', strainAtUTS1_1);

% Calculate strain at UTS for Test 1 File 2
strainAtUTS1_2 = findStrainAtUTS(stress1_2, strain1_2_shifted);
fprintf('\nTest 1 (File 2) Strain at UTS:\n');
fprintf('Ultimate Strength: %.2f MPa\n', ultimateStress1_2);
fprintf('Strain at UTS: %.4f\n', strainAtUTS1_2);

% Calculate strain at UTS for Test 2
strainAtUTS2 = findStrainAtUTS(stress2, strain2_shifted);
fprintf('\nTest 2 Strain at UTS:\n');
fprintf('Ultimate Strength: %.2f MPa\n', ultimateStress2);
fprintf('Strain at UTS: %.4f\n', strainAtUTS2);

% Calculate average strain at UTS
avgStrainAtUTS = (strainAtUTS1_1 + strainAtUTS1_2 + strainAtUTS2) / 3;
fprintf('\nAverage Strain at UTS: %.4f\n', avgStrainAtUTS);