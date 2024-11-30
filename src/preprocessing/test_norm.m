% test_normalization.m
clear; clc;

% Add preprocessing folder to path
addpath('src/preprocessing');

% Load your data
data = readtable(fullfile('data', 'raw', 'predictive_maintenance.csv'));

% Extract numerical features
numericalFeatures = table2array(data(:, {'AirTemperature_K_', ...
                                        'ProcessTemperature_K_', ...
                                        'RotationalSpeed_rpm_', ...
                                        'Torque_Nm_', ...
                                        'ToolWear_min_'}));

% Normalize the data
[normalized_data, norm_params] = normalize_data(numericalFeatures);

% Display statistics before and after normalization
fprintf('\nBefore normalization:\n');
fprintf('Feature\t\t\tMean\t\tStd\n');
fprintf('----------------------------------------\n');
for i = 1:size(numericalFeatures, 2)
    fprintf('%s\t%.4f\t%.4f\n', ...
        data.Properties.VariableNames{i+3}, ... % +3 to skip UDI, ProductID, Type
        mean(numericalFeatures(:,i)), ...
        std(numericalFeatures(:,i)));
end

fprintf('\nAfter normalization:\n');
fprintf('Feature\t\t\tMean\t\tStd\n');
fprintf('----------------------------------------\n');
for i = 1:size(normalized_data, 2)
    fprintf('%s\t%.4f\t%.4f\n', ...
        data.Properties.VariableNames{i+3}, ...
        mean(normalized_data(:,i)), ...
        std(normalized_data(:,i)));
end

% Verify normalization parameters are stored correctly
fprintf('\nStored normalization parameters:\n');
disp(norm_params);

% Test normalization with stored parameters (simulation of validation/test data)
[normalized_test, ~] = normalize_data(numericalFeatures, norm_params);

% Verify both normalizations give same result
max_diff = max(abs(normalized_data(:) - normalized_test(:)));
fprintf('\nMaximum difference between normalizations: %.10f\n', max_diff);