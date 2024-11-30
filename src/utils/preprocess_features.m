function [X_train_processed, X_val_processed, params] = preprocess_features(X_train, X_val)
    % Initialize preprocessing parameters
    params = struct();
    
    % 1. Z-score normalization
    params.mean = mean(X_train, 2);
    params.std = std(X_train, 0, 2);
    params.std(params.std == 0) = 1; % Prevent division by zero
    
    % Apply normalization
    X_train_norm = (X_train - params.mean) ./ params.std;
    X_val_norm = (X_val - params.mean) ./ params.std;
    
    % 2. Add polynomial features
    fprintf('Adding polynomial features...\n');
    X_train_poly = X_train_norm;
    X_val_poly = X_val_norm;
    
    n_features = size(X_train_norm, 1);
    feature_pairs = [];
    
    % Store feature pair indices for later use
    for i = 1:n_features
        for j = i:n_features
            feature_pairs = [feature_pairs; i j];
            new_feature_train = X_train_norm(i,:) .* X_train_norm(j,:);
            X_train_poly = [X_train_poly; new_feature_train];
            
            new_feature_val = X_val_norm(i,:) .* X_val_norm(j,:);
            X_val_poly = [X_val_poly; new_feature_val];
        end
    end
    
    % Store polynomial feature information
    params.original_features = n_features;
    params.feature_pairs = feature_pairs;
    params.total_features = size(X_train_poly, 1);
    
    % Return processed features
    X_train_processed = X_train_poly;
    X_val_processed = X_val_poly;
    
    % Print feature information
    fprintf('Original features: %d\n', n_features);
    fprintf('Polynomial features added: %d\n', size(feature_pairs, 1));
    fprintf('Total features after preprocessing: %d\n', params.total_features);
end