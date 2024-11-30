function [trainedNet, accuracy] = train_single_model(modelNum, X_train, Y_train, X_val, Y_val, layers, options)
    % Train a single model in the ensemble
    
    fprintf('\nTraining ensemble model %d\n', modelNum);
    
    % Bootstrap sampling
    n = size(X_train, 1);
    bootIdx = randi(n, n, 1);
    X_boot = X_train(bootIdx, :);
    Y_boot = Y_train(bootIdx);
    
    % Train network
    [trainedNet, ~] = trainNetwork(X_boot, Y_boot, layers, options);
    
    % Evaluate
    YPred = classify(trainedNet, X_val);
    accuracy = sum(YPred == Y_val) / numel(Y_val);
    
    fprintf('Model %d accuracy: %.2f%%\n', modelNum, accuracy * 100);
end