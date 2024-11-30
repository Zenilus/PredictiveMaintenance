% test_data_prep.m
% Script to test the data preparation

% Clear workspace and command window
clear; clc;

% Add preprocessing folder to path
addpath('src/preprocessing');

% Define path to your data
data_path = fullfile('data', 'raw', 'predictive_maintenance.csv');

% Prepare the data
try
    [X_train, X_val, X_test, Y_train, Y_val, Y_test] = prepare_features(data_path);
    fprintf('Data preparation successful!\n');
    
    % Save prepared data
    save(fullfile('data', 'processed', 'prepared_data.mat'), ...
         'X_train', 'X_val', 'X_test', 'Y_train', 'Y_val', 'Y_test');
    fprintf('Prepared data saved successfully!\n');
catch ME
    fprintf('Error in data preparation: %s\n', ME.message);
end