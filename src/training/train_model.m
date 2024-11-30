function model = train_model(session)
    % Load and preprocess data
    [X_train, Y_train, X_val, Y_val, preprocessing_params] = load_and_preprocess_data(session);
    
    % Print dimensions of data before training
    fprintf('Data dimensions in train_model:\n');
    fprintf('X_train: %dx%d\n', size(X_train));
    fprintf('Y_train: %dx%d\n', size(Y_train));
    fprintf('X_val: %dx%d\n', size(X_val));
    fprintf('Y_val: %dx%d\n', size(Y_val));
    
    % Ensure Y_train and Y_val are column vectors
    Y_train = Y_train(:);  % Convert to column vector
    Y_val = Y_val(:);      % Convert to column vector
    
    % Ensure X_train and X_val have samples as rows
    if size(X_train, 2) > size(X_train, 1)
        X_train = X_train';
        X_val = X_val';
        fprintf('Transposed X matrices to have samples as rows\n');
    end
    
    % Verify dimensions match
    assert(size(X_train, 1) == length(Y_train), ...
        sprintf('Dimension mismatch: X_train has %d samples but Y_train has %d labels', ...
        size(X_train, 1), length(Y_train)));
    
    assert(size(X_val, 1) == length(Y_val), ...
        sprintf('Dimension mismatch: X_val has %d samples but Y_val has %d labels', ...
        size(X_val, 1), length(Y_val)));
    
    % Train the ensemble model
    model = train_ensemble(X_train, Y_train, X_val, Y_val, preprocessing_params, session);
end