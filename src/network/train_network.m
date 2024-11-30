% train_network.m
function [trainedNet, trainInfo] = train_network(X_train, Y_train, X_val, Y_val)
    % Train the neural network for predictive maintenance
    
    % Create models directory at the project root
    modelDir = fullfile(pwd, 'models');
    if ~exist(modelDir, 'dir')
        mkdir(modelDir);
        fprintf('Created models directory at: %s\n', modelDir);
    end
    
    % Get input size and number of classes
    inputSize = size(X_train, 1);
    numClasses = size(Y_train, 1);
    
    % Create the network
    net = create_network(inputSize, numClasses);
    
    % Setup training options with validation data
    options = setup_training(true);
    
    % Prepare data for training
    % Transpose X to have samples as rows
    X_train = X_train';
    X_val = X_val';
    
    % Convert Y to categorical
    [~, Y_train_idx] = max(Y_train, [], 1);
    [~, Y_val_idx] = max(Y_val, [], 1);
    Y_train_cat = categorical(Y_train_idx');
    Y_val_cat = categorical(Y_val_idx');
    
    % Set validation data
    options.ValidationData = {X_val, Y_val_cat};
    
    % Train the network
    try
        % Initialize best validation accuracy tracker
        bestValAcc = 0;
        bestNet = [];
        
        [trainedNet, trainInfo] = trainNetwork(X_train, Y_train_cat, net, options);
        
        % Save best model based on validation accuracy
        valAcc = max(trainInfo.ValidationAccuracy);
        if valAcc > bestValAcc
            bestValAcc = valAcc;
            bestNet = trainedNet;
        end
        
        % Save both best and final models
        modelDir = fullfile(pwd, 'models');
        save(fullfile(modelDir, 'best_model.mat'), 'bestNet', 'bestValAcc');
        save(fullfile(modelDir, 'final_model.mat'), 'trainedNet', 'trainInfo');
        
        fprintf('Best validation accuracy achieved: %.2f%%\n', bestValAcc);
        
    catch ME
        fprintf('Error during training: %s\n', ME.message);
        rethrow(ME);
    end
end