function [trainedNet, validationAccuracy] = train_single_model(modelIndex, X_train, Y_train, X_val, Y_val, layers, options)
    % Debug output for input dimensions
    fprintf('Initial dimensions in train_single_model (Model %d):\n', modelIndex);
    fprintf('X_train: %dx%d\n', size(X_train));
    fprintf('Y_train: %dx%d\n', size(Y_train));
    
    % Convert Y_train to categorical first
    Y_train_cat = categorical(Y_train);
    Y_val_cat = categorical(Y_val);
    
    % Get unique classes
    uniqueClasses = categories(Y_train_cat);
    fprintf('Unique classes in original data: %d\n', length(uniqueClasses));
    
    % Initialize arrays for stratified bootstrap
    samplesPerClass = 1000;  % Fixed number of samples per class for balance
    bootIdx = [];
    
    % Perform stratified bootstrap sampling
    for classIdx = 1:length(uniqueClasses)
        % Find samples for current class
        classIndices = find(Y_train_cat == uniqueClasses{classIdx});
        
        % Sample with replacement from current class
        classSamples = randsample(classIndices, samplesPerClass, true);
        bootIdx = [bootIdx; classSamples];
    end
    
    % Shuffle the combined indices
    bootIdx = bootIdx(randperm(length(bootIdx)));
    
    % Create bootstrap samples
    X_boot = X_train(bootIdx, :);
    Y_boot = Y_train_cat(bootIdx);
    
    % Debug output for bootstrap dimensions and classes
    fprintf('After bootstrap (Model %d):\n', modelIndex);
    fprintf('X_boot: %dx%d\n', size(X_boot));
    fprintf('Number of classes in Y_boot: %d\n', length(categories(Y_boot)));
    
    % Show class distribution
    tbl = tabulate(Y_boot);
    disp(tbl);
    
    % Verify dimensions and classes
    assert(size(X_boot, 1) == length(Y_boot), 'Dimension mismatch between X_boot and Y_boot');
    assert(length(categories(Y_boot)) == 6, 'Expected 6 classes in Y_boot');  % Changed to 6
    
    % Train network
    trainedNet = trainNetwork(X_boot, Y_boot, layers, options);
    
    % Calculate validation accuracy
    Y_pred = classify(trainedNet, X_val);
    validationAccuracy = sum(Y_pred == Y_val_cat) / numel(Y_val_cat);
    
    fprintf('Model %d Validation Accuracy: %.2f%%\n', modelIndex, validationAccuracy * 100);
end