% test_data_prep.m
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
    
    % Display size information
    fprintf('\nDataset sizes:\n');
    fprintf('Training set: %d samples\n', size(X_train, 2));
    fprintf('Validation set: %d samples\n', size(X_val, 2));
    fprintf('Test set: %d samples\n', size(X_test, 2));
catch ME
    fprintf('Error in data preparation: %s\n', ME.message);
end