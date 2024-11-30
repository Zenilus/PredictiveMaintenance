% test_network.m
clear; clc;

% Add network folder to path
addpath('src/network');

% Load your prepared data to get dimensions
load('data/processed/prepared_data.mat');

% Get input size and number of classes
inputSize = size(X_train, 1);
numClasses = size(Y_train, 1);

% Create the network
net = create_network(inputSize, numClasses);

% Display network information
fprintf('\nNetwork Structure Summary:\n');
fprintf('Input size: %d\n', inputSize);
fprintf('Number of classes: %d\n', numClasses);
fprintf('Number of layers: %d\n', numel(net.Layers));

% Display layer information
fprintf('\nDetailed Layer Information:\n');
fprintf('----------------------------------------\n');
for i = 1:numel(net.Layers)
    layer = net.Layers(i);
    fprintf('Layer %d: %s\n', i, class(layer));
    if isprop(layer, 'OutputSize')
        fprintf('Output size: %d\n', layer.OutputSize);
    end
    fprintf('----------------------------------------\n');
end