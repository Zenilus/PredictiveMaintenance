function model = train_model(session)
    fprintf('Loading and preparing data...\n');
    load('data/processed/prepared_data.mat');
    
    % Display data information
    fprintf('\nOriginal Data Information:\n');
    fprintf('Input features: %d\n', size(X_train, 1));
    [~, Y_train_idx] = max(Y_train, [], 1);
    uniqueClasses = unique(Y_train_idx);
    fprintf('Number of unique classes: %d\n', numel(uniqueClasses));
    
    % Feature preprocessing
    fprintf('\nApplying feature preprocessing...\n');
    [X_train, X_val, preprocessing_params] = preprocess_features(X_train, X_val);
    
    % Balance data
    fprintf('\nBalancing training data...\n');
    [X_train, Y_train] = balance_data_smote(X_train, Y_train, 2000);
    
    % Create and train ensemble
    model = train_ensemble(X_train, Y_train, X_val, Y_val, preprocessing_params, session);
    
    % Save model
    save_trained_model(model, session);
end