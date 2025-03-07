% DICe result reader for multiple time points
% Clear workspace and figures
clc;
clear all;
close all;

% Define the files to read
files = {'DICe_solution_00.txt', 'DICe_solution_03.txt', 'DICe_solution_09.txt'};
titles = {'t = 0', 't = 9', 't = 16'};

% Create a figure
figure('Position', [100, 100, 1200, 400]);

% Initialize variable to track global min/max for consistent colorbar
global_min = inf;
global_max = -inf;

% First pass to find global min/max
for i = 1:length(files)
    mydata = readmatrix(files{i});
    strain_yy = mydata(:,12);
    global_min = min(global_min, min(strain_yy));
    global_max = max(global_max, max(strain_yy));
end

% Second pass to create plots
for i = 1:length(files)
    % Read data
    mydata = readmatrix(files{i});
    strain_yy = mydata(:,12);
    xpos = mydata(:,2);
    ypos = mydata(:,3);
    
    % Create subplot
    subplot(1,3,i)
    scatter(xpos, ypos, [], strain_yy, 'filled')
    
    % Set consistent colormap limits
    caxis([global_min global_max])
    
    % Add labels and title
    xlabel('x')
    ylabel('y')
    title(titles{i})
    
    % Set font size and line width
    set(gca, 'fontsize', 14, 'linewidth', 1.5)
    
    % Add colorbar to each subplot
    colorbar
end

% Add overall title
sgtitle('Strain YY Comparison', 'FontSize', 16)

% Use the same colormap for all plots
colormap('jet')

% Make the figure look nice
set(gcf, 'Color', 'white')