function [layers, options] = create_network_config(X_train, Y_train, X_val, Y_val)
    % Get dimensions
    inputSize = size(X_train, 2);    % should be 8
    numClasses = 6;                  % We now know we have 6 classes
    
    fprintf('Network configuration:\n');
    fprintf('Input size: %d\n', inputSize);
    fprintf('Number of classes: %d\n', numClasses);
    
    % Convert labels to categorical
    Y_train_cat = categorical(Y_train);
    Y_val_cat = categorical(Y_val);
    
    layers = [
        featureInputLayer(inputSize, 'Name', 'input')
        
        % Hidden layers
        fullyConnectedLayer(512, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'drop1')
        
        fullyConnectedLayer(256, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.3, 'Name', 'drop2')
        
        % Output layer
        fullyConnectedLayer(numClasses, 'Name', 'fc_out')  % Changed to 6 classes
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classification')
    ];
    
    options = trainingOptions('adam', ...
        'InitialLearnRate', 0.001, ...
        'MaxEpochs', 50, ...
        'MiniBatchSize', 128, ...
        'ValidationData', {X_val, Y_val_cat}, ...
        'ValidationFrequency', 30, ...
        'ValidationPatience', 5, ...
        'Shuffle', 'every-epoch', ...
        'Verbose', true, ...
        'VerboseFrequency', 20, ...
        'Plots', 'training-progress');
end