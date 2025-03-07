%% PMMA Poisson's Ratio Calculation from DICe Data
% This script loads DICe solution files and calculates Poisson's ratio
% for PMMA dogbone specimens

clear all; close all; clc;

%% Parameters
numImages = 11; % DICe solution files from 00 to 17
filePrefix = 'DICe_solution_';
fileExt = '.txt';

% Create arrays to store results across all images
poissonsRatios = zeros(numImages, 1);
maxStrainYY = zeros(numImages, 1);
rSquared = zeros(numImages, 1);

% Specimen geometry parameters (based on ASTM D638)
gaugeWidth = 13; % mm
gaugeThickness = 3.2; % mm
gaugeLength = 50; % mm

% Create a figure to visualize the strain relationship
figure('Position', [100, 100, 800, 600]);
hold on;

%% Process all DICe solution files
for imgIndex = 0:numImages-1
    % Construct filename with correct formatting
    if imgIndex < 10
        filename = [filePrefix, '0', num2str(imgIndex), fileExt];
    else
        filename = [filePrefix, num2str(imgIndex), fileExt];
    end
    
    % Check if file exists
    if exist(filename, 'file') ~= 2
        warning('File %s does not exist, skipping.', filename);
        continue;
    end
    
    % Read DICe solution file
    try
        % Read the data
        data = readtable(filename, 'Delimiter', ',');
        fprintf('Processing %s: %d data points\n', filename, height(data));
        
        % Filter out invalid points
        validData = data(data.STATUS_FLAG == 4, :);
        
        % Extract coordinates and strains
        x = validData.COORDINATE_X;
        y = validData.COORDINATE_Y;
        strainXX = validData.VSG_STRAIN_XX;
        strainYY = validData.VSG_STRAIN_YY;
        strainXY = validData.VSG_STRAIN_XY;
        
        % Calculate the bounds of the specimen
        yMin = min(y);
        yMax = max(y);
        centerY = (yMin + yMax) / 2;
        
        % Define the gauge section (focused on middle portion of specimen)
        % Pixel to mm conversion factor would typically come from camera calibration
        pixelToMm = 10; % Estimate for a typical DIC setup
        gaugeHalfLength = gaugeLength / 2;
        gaugePixels = gaugeHalfLength * pixelToMm;
        
        gaugeYMin = centerY - gaugePixels;
        gaugeYMax = centerY + gaugePixels;
        
        % Filter points in the gauge section
        gaugeIndices = (y >= gaugeYMin) & (y <= gaugeYMax);
        
        gaugeX = x(gaugeIndices);
        gaugeY = y(gaugeIndices);
        gaugeStrainXX = strainXX(gaugeIndices);
        gaugeStrainYY = strainYY(gaugeIndices);
        
        % Remove outliers using IQR method
        [filteredStrainXX, xxOutlierIndices] = removeOutliers(gaugeStrainXX);
        [filteredStrainYY, yyOutlierIndices] = removeOutliers(gaugeStrainYY);
        
        % Create indices of non-outlier points
        xxValidIndices = true(size(gaugeStrainXX));
        xxValidIndices(xxOutlierIndices) = false;
        
        yyValidIndices = true(size(gaugeStrainYY));
        yyValidIndices(yyOutlierIndices) = false;
        
        % Keep only points that are valid for both XX and YY
        validIndices = xxValidIndices & yyValidIndices;
        
        % Extract final filtered strains
        finalStrainXX = gaugeStrainXX(validIndices);
        finalStrainYY = gaugeStrainYY(validIndices);
        
        % Calculate average strains
        avgStrainXX = mean(finalStrainXX);
        avgStrainYY = mean(finalStrainYY);
        
        % Calculate Poisson's ratio from averages
        poissonRatioAvg = -avgStrainXX / avgStrainYY;
        
        % Linear regression for more robust calculation
        if ~isempty(finalStrainXX) && ~isempty(finalStrainYY)
            % Linear regression from YY to XX (εxx = α + β·εyy)
            [p, S] = polyfit(finalStrainYY, finalStrainXX, 1);
            slope = p(1);
            intercept = p(2);
            
            % Calculate R-squared
            yMean = mean(finalStrainXX);
            SStot = sum((finalStrainXX - yMean).^2);
            SSres = S.normr^2;
            R2 = 1 - SSres/SStot;
            
            % Store results
            poissonRatio = -slope;
            poissonsRatios(imgIndex+1) = poissonRatio;
            maxStrainYY(imgIndex+1) = max(abs(finalStrainYY));
            rSquared(imgIndex+1) = R2;
            
            % Plot the strain relationship for this image
            scatter(finalStrainYY, finalStrainXX, 25, 'filled', 'MarkerFaceAlpha', 0.5);
            
            % Display results for this image
            fprintf('Image %d: Poisson ratio = %.4f, R² = %.4f, Max strain = %.6f\n', ...
                imgIndex, poissonRatio, R2, max(abs(finalStrainYY)));
        else
            fprintf('Image %d: Not enough valid data points\n', imgIndex);
        end
    catch e
        warning('Error processing %s: %s', filename, e.message);
    end
end

%% Plot and Calculate Final Results

% Filter out zero values (files that weren't processed)
validResults = poissonsRatios ~= 0;
poissonsRatios = poissonsRatios(validResults);
maxStrainYY = maxStrainYY(validResults);
rSquared = rSquared(validResults);

% Apply weights based on R-squared and strain magnitude
weights = rSquared .* maxStrainYY;
weights = weights / sum(weights); % Normalize weights

% Calculate weighted average of Poisson's ratio
finalPoissonRatio = sum(poissonsRatios .* weights);

% Add linear regression line to plot
[sortedStrain, sortIndex] = sort(finalStrainYY);
sortedXX = finalStrainXX(sortIndex);
plot(sortedStrain, sortedXX, 'k-', 'LineWidth', 2);

% Add reference line showing Poisson's ratio = 0.35 (literature value)
refSlope = -0.35;
refIntercept = mean(finalStrainXX) - refSlope * mean(finalStrainYY);
refX = [min(finalStrainYY), max(finalStrainYY)];
refY = refSlope * refX + refIntercept;
plot(refX, refY, 'r--', 'LineWidth', 2);

% Add labels and legend
xlabel('Axial Strain (\epsilon_{yy})', 'FontSize', 14);
ylabel('Transverse Strain (\epsilon_{xx})', 'FontSize', 14);
title('Strain Relationship for PMMA Specimen', 'FontSize', 16);
legend('Strain Data', 'Linear Fit', 'Literature Value (v=0.35)', 'Location', 'best');
grid on;

% Display final results
fprintf('\n=== Final Results ===\n');
fprintf('Number of images processed: %d\n', sum(validResults));
fprintf('Weighted Average Poisson''s Ratio: %.4f\n', finalPoissonRatio);
fprintf('Literature value for PMMA: 0.35-0.40\n');

% Create a table with results for each image
resultTable = table((0:numImages-1)', poissonsRatios, rSquared, maxStrainYY, ...
    'VariableNames', {'ImageIndex', 'PoissonsRatio', 'R_Squared', 'MaxStrain'});
disp(resultTable);

% Create a figure showing Poisson's ratio vs strain level
figure('Position', [100, 400, 800, 400]);
plot(maxStrainYY, poissonsRatios, 'o-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'auto');
xlabel('Maximum Axial Strain', 'FontSize', 14);
ylabel('Poisson''s Ratio', 'FontSize', 14);
title('Poisson''s Ratio vs. Strain Level', 'FontSize', 16);
grid on;
yline(0.35, 'r--', 'Literature Value', 'LineWidth', 2);

%% Helper Functions

function [filteredData, outlierIndices] = removeOutliers(data)
    % Remove outliers using the IQR method
    
    % Sort the data
    sortedData = sort(data);
    
    % Calculate quartiles
    q1Idx = floor(length(sortedData) * 0.25);
    q3Idx = floor(length(sortedData) * 0.75);
    
    q1 = sortedData(q1Idx);
    q3 = sortedData(q3Idx);
    
    % Calculate IQR and bounds
    iqr = q3 - q1;
    lowerBound = q1 - 1.5 * iqr;
    upperBound = q3 + 1.5 * iqr;
    
    % Find outlier indices
    outlierIndices = find(data < lowerBound | data > upperBound);
    
    % Filter the data
    filteredData = data;
    filteredData(outlierIndices) = [];
end