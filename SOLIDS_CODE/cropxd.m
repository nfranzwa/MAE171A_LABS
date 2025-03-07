% Clear workspace and figures
clc; clear all; close all;

% Get current directory and set folder path
current_dir = pwd;
folder_path = fullfile(current_dir, '7131');
cd(folder_path);

% Test on one file first to check crop coordinates
test_file = 'Screenshot 2025-02-23 205414.png';  % Use one of your actual filenames
test_img = imread(test_file);
imshow(test_img);

% You'll need to adjust these crop coordinates based on your images
% Format is [x_start y_start width height]
cropimg = imcrop(test_img,[840 200 250 550]); % We'll adjust these values after testing
figure;
imshow(cropimg);

% Once you're happy with the crop coordinates, process all files
% Create the cropped folder if it doesn't exist
if ~exist('cropped', 'dir')
    mkdir('cropped');
end

% Get all screenshot files
myfiles = dir('Screenshot*.png');  % This will match your screenshot files

for i = 1:length(myfiles)
    myimg = imread(myfiles(i).name);
    cropimg = imcrop(myimg,[800 175 400 600]); % Use same coordinates as above
    imwrite(cropimg,[pwd '/cropped/crop_' myfiles(i).name]);
end