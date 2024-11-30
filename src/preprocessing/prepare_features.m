% prepare_features.m
function [X_train, X_val, X_test, Y_train, Y_val, Y_test] = prepare_features(data_path)
    % Load the data
    data = readtable(data_path);
    
    % 1. Extract features
    % Remove UID as it's just an identifier
    data = removevars(data, {'UDI'});
    
    % 2. Handle Product Type
    % Convert Type to categorical
    productTypes = categorical(data.Type);
    typeMatrix = dummyvar(productTypes);  % One-hot encoding
    
    % 3. Extract numerical features
    numericalFeatures = table2array(data(:, {'AirTemperature_K_', ...
                                            'ProcessTemperature_K_', ...
                                            'RotationalSpeed_rpm_', ...
                                            'Torque_Nm_', ...
                                            'ToolWear_min_'}));
    
    % 4. Normalize numerical features
    normalizedFeatures = normalize(numericalFeatures);
    
    % 5. Combine features
    X = [typeMatrix normalizedFeatures];
    
    % 6. Prepare labels - Modified to include non-failure cases
    % Create a categorical array for all cases
    failureTypes = categorical(data.FailureType);
    % Replace empty failure types with 'No_Failure'
    failureTypes(failureTypes == '') = 'No_Failure';
    
    % Convert labels to one-hot encoding for neural network
    Y = dummyvar(failureTypes)';
    X = X';  % Transpose for neural network (samples should be in columns)
    
    % 7. Split data (60% train, 20% validation, 20% test)
    numSamples = size(X, 2);
    
    % Create random indices for splitting
    rng(42); % For reproducibility
    idx = randperm(numSamples);
    
    trainSize = floor(0.6 * numSamples);
    valSize = floor(0.2 * numSamples);
    
    % Split indices
    trainIdx = idx(1:trainSize);
    valIdx = idx(trainSize+1:trainSize+valSize);
    testIdx = idx(trainSize+valSize+1:end);
    
    % Create final datasets
    X_train = X(:, trainIdx);
    X_val = X(:, valIdx);
    X_test = X(:, testIdx);
    
    Y_train = Y(:, trainIdx);
    Y_val = Y(:, valIdx);
    Y_test = Y(:, testIdx);
    
    % Display information about the prepared dataset
    fprintf('\nDataset preparation completed:\n');
    fprintf('Input features: %d\n', size(X, 1));
    fprintf('Training samples: %d\n', size(X_train, 2));
    fprintf('Validation samples: %d\n', size(X_val, 2));
    fprintf('Test samples: %d\n', size(X_test, 2));
    fprintf('Number of classes: %d\n', size(Y, 1));
    
    % Display class distribution
    fprintf('\nClass distribution in training set:\n');
    [~, trainLabels] = max(Y_train);
    tabulate(trainLabels);
end