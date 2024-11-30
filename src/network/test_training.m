% test_training.m
clear; clc;

% Add paths
addpath('src/network');
addpath('src/preprocessing');

try
    % Load prepared data
    load('data/processed/prepared_data.mat');
    
    % Display training data information
    fprintf('Training Data Information:\n');
    fprintf('Training samples: %d\n', size(X_train, 2));
    fprintf('Validation samples: %d\n', size(X_val, 2));
    fprintf('Number of features: %d\n', size(X_train, 1));
    fprintf('Number of classes: %d\n', size(Y_train, 1));
    
    % Create directory for models if it doesn't exist
    if ~exist('models', 'dir')
        mkdir('models');
    end
    
    % Train the network
    fprintf('\nStarting network training...\n');
    [trainedNet, trainInfo] = train_network(X_train, Y_train, X_val, Y_val);
    
    % Display training results
    fprintf('\nTraining completed!\n');
    fprintf('Final validation accuracy: %.2f%%\n', trainInfo.ValidationAccuracy(end));
    fprintf('Best validation accuracy: %.2f%%\n', max(trainInfo.ValidationAccuracy));
    fprintf('Total epochs trained: %d\n', numel(trainInfo.TrainingAccuracy));
    
    % Plot training progress
    figure('Name', 'Training Progress');
    subplot(2,1,1);
    plot(trainInfo.TrainingAccuracy, 'b-');
    hold on;
    plot(trainInfo.ValidationAccuracy, 'r-');
    xlabel('Epoch');
    ylabel('Accuracy (%)');
    title('Training and Validation Accuracy');
    legend('Training', 'Validation');
    grid on;
    
    subplot(2,1,2);
    plot(trainInfo.TrainingLoss, 'b-');
    hold on;
    plot(trainInfo.ValidationLoss, 'r-');
    xlabel('Epoch');
    ylabel('Loss');
    title('Training and Validation Loss');
    legend('Training', 'Validation');
    grid on;
    
catch ME
    fprintf('Error: %s\n', ME.message);
end