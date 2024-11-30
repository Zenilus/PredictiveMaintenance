function [layers, options] = create_network_config(X_train, Y_train_cat, X_val, Y_val_cat)
    inputSize = size(X_train, 2);
    
    % Get number of classes from categorical array
    numClasses = length(categories(Y_train_cat));
    
    % Calculate class weights using categorical data
    class_counts = countcats(Y_train_cat);
    class_weights = 1 ./ class_counts;
    class_weights = class_weights / sum(class_weights) * numClasses;
    
    layers = [
        featureInputLayer(inputSize, 'Name', 'input')
        
        % First block with residual connection
        fullyConnectedLayer(512, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'drop1')
        
        % Second block with increased capacity
        fullyConnectedLayer(1024, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.3, 'Name', 'drop2')
        
        % Third block with skip connection
        fullyConnectedLayer(512, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        dropoutLayer(0.3, 'Name', 'drop3')
        
        % Final classification layers
        fullyConnectedLayer(numClasses, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        weightedClassificationLayer(class_weights) % Simplified constructor call
    ];
    
    % Updated training options
    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.1, ...
        'LearnRateDropPeriod', 20, ...
        'MaxEpochs', 100, ...
        'MiniBatchSize', 128, ...
        'L2Regularization', 0.001, ...
        'ValidationData', {X_val, Y_val_cat}, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 10, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 20, ...
        'Plots', 'training-progress');
end