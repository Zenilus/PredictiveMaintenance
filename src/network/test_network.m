% test_network.m
clear; clc;

try
    % Set up parallel computing
    if isempty(gcp('nocreate'))
        % Get number of physical cores
        numCores = feature('numcores');
        fprintf('Number of CPU cores detected: %d\n', numCores);
        
        % Use all cores except one for system processes
        numWorkersToUse = max(1, numCores - 1);
        fprintf('Setting up parallel pool with %d workers...\n', numWorkersToUse);
        
        pool = parpool('local', numWorkersToUse);
        fprintf('Parallel pool created successfully.\n');
    else
        pool = gcp;
        fprintf('Using existing parallel pool with %d workers.\n', pool.NumWorkers);
    end

    % Add required paths
    addpath('src/network');
    addpath('src/preprocessing');

    % Load prepared data
    fprintf('Loading prepared data...\n');
    load('data/processed/prepared_data.mat');
    
    % Display data information
    fprintf('\nData Information:\n');
    fprintf('Input features: %d\n', size(X_train, 1));
    
    % Check unique classes
    [~, Y_train_idx] = max(Y_train, [], 1);
    uniqueClasses = unique(Y_train_idx);
    fprintf('Number of unique classes: %d\n', numel(uniqueClasses));
    
    % Create network with correct number of classes
    inputSize = size(X_train, 1);
    numClasses = numel(uniqueClasses);
    fprintf('\nCreating network with:\n');
    fprintf('Input size: %d\n', inputSize);
    fprintf('Output classes: %d\n', numClasses);
    
    % Create the network
    net = create_network(inputSize, numClasses);
    
     % Setup optimized training options for CPU-only
    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 20, ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 256, ...
        'ExecutionEnvironment', 'cpu', ...
        'ValidationFrequency', 50, ...
        'ValidationPatience', 8, ...
        'Verbose', true, ...
        'VerboseFrequency', 20, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch');


    % Prepare data format
    fprintf('Preparing data for training...\n');
    X_train = X_train';
    Y_train = Y_train';
    X_val = X_val';
    Y_val = Y_val';
    
    % Convert Y to categorical
    [~, Y_train_idx] = max(Y_train, [], 2);
    [~, Y_val_idx] = max(Y_val, [], 2);
    Y_train_cat = categorical(Y_train_idx);
    Y_val_cat = categorical(Y_val_idx);
    
    % Set validation data
    validationData = {X_val, Y_val_cat};
    options.ValidationData = validationData;
    
    % Train the network
    fprintf('\nStarting network training...\n');
    tic; % Start timing
    [trainedNet, trainInfo] = trainNetwork(X_train, Y_train_cat, net, options);
    trainingTime = toc; % End timing
    
    % Create models directory if it doesn't exist
    if ~exist('models', 'dir')
        mkdir('models');
    end
    
    % Save the trained model
    save('models/trained_model.mat', 'trainedNet', 'trainInfo');
    
    % Display results
    fprintf('\nTraining completed!\n');
    fprintf('Training time: %.2f minutes\n', trainingTime/60);
    fprintf('Final validation accuracy: %.2f%%\n', trainInfo.ValidationAccuracy(end));
    fprintf('Best validation accuracy: %.2f%%\n', max(trainInfo.ValidationAccuracy));
    
    % Clean up parallel pool
    delete(gcp('nocreate'));
    fprintf('Parallel pool cleaned up.\n');
    
catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Error location: %s\n', ME.stack(1).name);
    % Clean up parallel pool in case of error
    delete(gcp('nocreate'));
end