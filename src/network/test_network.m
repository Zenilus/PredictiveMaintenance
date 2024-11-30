% test_network.m
clear; clc;

try
    % Record start time and user info
    startTime = datetime('now', 'TimeZone', 'UTC');
    currentUser = 'Zenilus';
    fprintf('Starting training session at %s UTC\n', char(startTime));
    fprintf('User: %s\n\n', currentUser);

    % Set up parallel computing
    if isempty(gcp('nocreate'))
        numCores = feature('numcores');
        fprintf('Number of CPU cores detected: %d\n', numCores);
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

    % Load and preprocess data
    fprintf('Loading prepared data...\n');
    load('data/processed/prepared_data.mat');
    
    % Display original data information
    fprintf('\nOriginal Data Information:\n');
    fprintf('Input features: %d\n', size(X_train, 1));
    [~, Y_train_idx] = max(Y_train, [], 1);
    uniqueClasses = unique(Y_train_idx);
    fprintf('Number of unique classes: %d\n', numel(uniqueClasses));
    
    % Enhanced Feature Preprocessing
    fprintf('\nApplying advanced feature preprocessing...\n');
    
    % Z-score normalization
    X_mean = mean(X_train, 2);
    X_std = std(X_train, 0, 2);
    X_std(X_std == 0) = 1;
    X_train = (X_train - X_mean) ./ X_std;
    X_val = (X_val - X_mean) ./ X_std;
    
    % Add polynomial features
    fprintf('Adding polynomial features...\n');
    X_train_poly = X_train;
    X_val_poly = X_val;
    
    for i = 1:size(X_train, 1)
        for j = i:size(X_train, 1)
            new_feature = X_train(i,:) .* X_train(j,:);
            X_train_poly = [X_train_poly; new_feature];
            
            new_feature_val = X_val(i,:) .* X_val(j,:);
            X_val_poly = [X_val_poly; new_feature_val];
        end
    end
    
    X_train = X_train_poly;
    X_val = X_val_poly;
    
    % Balance with more samples
    fprintf('\nBalancing training data...\n');
    [X_train, Y_train] = balance_data(X_train, Y_train, 2000);
    
    % Create network architecture
    inputSize = size(X_train, 1);
    numClasses = numel(uniqueClasses);
    
    layers = [
        featureInputLayer(inputSize, 'Name', 'input')
        
        fullyConnectedLayer(512, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.4, 'Name', 'drop1')
        
        fullyConnectedLayer(1024, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.4, 'Name', 'drop2')
        
        fullyConnectedLayer(512, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        dropoutLayer(0.4, 'Name', 'drop3')
        
        fullyConnectedLayer(256, 'Name', 'fc4')
        batchNormalizationLayer('Name', 'bn4')
        reluLayer('Name', 'relu4')
        dropoutLayer(0.3, 'Name', 'drop4')
        
        fullyConnectedLayer(128, 'Name', 'fc5')
        batchNormalizationLayer('Name', 'bn5')
        reluLayer('Name', 'relu5')
        
        fullyConnectedLayer(numClasses, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classification')
    ];
    
    % Enhanced training options
    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 50, ...
        'MaxEpochs', 400, ...
        'MiniBatchSize', 32, ...
        'L2Regularization', 0.00001, ...
        'ExecutionEnvironment', 'cpu', ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 25, ...
        'Verbose', true, ...
        'VerboseFrequency', 20, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch', ...
        'GradientDecayFactor', 0.9, ...
        'SquaredGradientDecayFactor', 0.999);

    % Prepare data format
    fprintf('Preparing data for training...\n');
    X_train = X_train';
    Y_train = Y_train';
    X_val = X_val';
    Y_val = Y_val';
    
    % Convert labels
    [~, Y_train_idx] = max(Y_train, [], 2);
    [~, Y_val_idx] = max(Y_val, [], 2);
    Y_train_cat = categorical(Y_train_idx);
    Y_val_cat = categorical(Y_val_idx);
    
    % Train ensemble models
    fprintf('\nStarting ensemble training...\n');
    tic;
    numEnsemble = 3;
    ensembleNets = cell(numEnsemble, 1);
    ensembleAccuracies = zeros(numEnsemble, 1);
    
    for e = 1:numEnsemble
        fprintf('\nTraining ensemble model %d/%d\n', e, numEnsemble);
        
        % Bootstrap sampling
        n = size(X_train, 1);
        bootIdx = randi(n, n, 1);
        X_boot = X_train(bootIdx, :);
        Y_boot = Y_train_cat(bootIdx);
        
        % Split data for validation
        validationRatio = 0.2;
        numValidation = round(size(X_boot, 1) * validationRatio);
        
        % Random permutation for splitting
        idx = randperm(size(X_boot, 1));
        validIdx = idx(1:numValidation);
        trainIdx = idx(numValidation+1:end);
        
        X_train_split = X_boot(trainIdx, :);
        Y_train_split = Y_boot(trainIdx);
        X_val_split = X_boot(validIdx, :);
        Y_val_split = Y_boot(validIdx);
        
        % Update validation data
        options.ValidationData = {X_val_split, Y_val_split};
        
        % Train network
        [trainedNet, ~] = trainNetwork(X_train_split, Y_train_split, layers, options);
        
        % Evaluate on validation set
        YPred = classify(trainedNet, X_val_split);
        accuracy = sum(YPred == Y_val_split) / numel(Y_val_split);
        
        ensembleNets{e} = trainedNet;
        ensembleAccuracies(e) = accuracy;
    end
    
    trainingTime = toc;
    
    % Save the ensemble model
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    modelFileName = sprintf('models/ensemble_model_%s.mat', timestamp);
    
    if ~exist('models', 'dir')
        mkdir('models');
    end
    
    modelInfo.timestamp = timestamp;
    modelInfo.trainingTime = trainingTime;
    modelInfo.ensembleAccuracies = ensembleAccuracies;
    modelInfo.meanAccuracy = mean(ensembleAccuracies);
    modelInfo.stdAccuracy = std(ensembleAccuracies);
    modelInfo.bestAccuracy = max(ensembleAccuracies);
    modelInfo.normalization.mean = X_mean;
    modelInfo.normalization.std = X_std;
    modelInfo.trainingOptions = options;
    modelInfo.startTime = startTime;
    modelInfo.user = currentUser;
    
    save(modelFileName, 'ensembleNets', 'modelInfo');
    
    % Evaluate ensemble on validation set
    fprintf('\nEvaluating ensemble on validation set...\n');
    ensemble_predictions = zeros(size(X_val, 1), numClasses);
    
    for e = 1:numEnsemble
        net = ensembleNets{e};
        pred = predict(net, X_val);
        ensemble_predictions = ensemble_predictions + pred;
    end
    
    [~, ensemble_pred] = max(ensemble_predictions, [], 2);
    ensemble_accuracy = sum(ensemble_pred == Y_val_idx) / numel(Y_val_idx);
    
    % Display results
    fprintf('\nTraining completed!\n');
    fprintf('Training time: %.2f minutes\n', trainingTime/60);
    fprintf('Individual model accuracies:\n');
    for e = 1:numEnsemble
        fprintf('Model %d: %.2f%%\n', e, ensembleAccuracies(e)*100);
    end
    fprintf('Ensemble validation accuracy: %.2f%%\n', ensemble_accuracy*100);
    fprintf('Model saved as: %s\n', modelFileName);
    
    % Clean up
    delete(gcp('nocreate'));
    fprintf('Parallel pool cleaned up.\n');
    
catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Error location: %s\n', ME.stack(1).name);
    delete(gcp('nocreate'));
    rethrow(ME);
end