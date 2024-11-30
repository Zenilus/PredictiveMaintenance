function [X_train, Y_train, X_val, Y_val, Y_train_cat, Y_val_cat] = prepare_data_format(X_train, Y_train, X_val, Y_val)
    % Convert data to correct format for training
    
    % Transpose matrices if needed
    X_train = X_train';
    Y_train = Y_train';
    X_val = X_val';
    Y_val = Y_val';
    
    % Convert labels to categorical format
    [~, Y_train_idx] = max(Y_train, [], 2);
    [~, Y_val_idx] = max(Y_val, [], 2);
    Y_train_cat = categorical(Y_train_idx);
    Y_val_cat = categorical(Y_val_idx);
    
    % Print data format information
    fprintf('Data format prepared:\n');
    fprintf('Training data size: %dx%d\n', size(X_train));
    fprintf('Validation data size: %dx%d\n', size(X_val));
    fprintf('Number of classes: %d\n', numel(unique(Y_train_idx)));
end