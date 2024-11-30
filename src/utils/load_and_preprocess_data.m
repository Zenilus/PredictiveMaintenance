function [X_train, Y_train, X_val, Y_val, preprocessing_params] = load_and_preprocess_data(session)
    try
        % Load the dataset
        data = load('prepared_data.mat');
        
        % Extract variables
        X_train = data.X_train;  % 8x6000
        Y_train = data.Y_train;  % 7x6000
        X_val = data.X_val;      % 8x2000
        Y_val = data.Y_val;      % 7x2000
        
        % Transpose X to have samples as rows
        X_train = X_train';  % Now 6000x8
        X_val = X_val';      % Now 2000x8
        
        % Standardize features
        feature_mean = mean(X_train);
        feature_std = std(X_train);
        
        % Avoid division by zero
        feature_std(feature_std == 0) = 1;
        
        % Apply standardization
        X_train = (X_train - feature_mean) ./ feature_std;
        X_val = (X_val - feature_mean) ./ feature_std;
        
        % Convert Y from one-hot encoding to class indices
        [~, Y_train] = max(Y_train);  % Convert to 1x6000
        [~, Y_val] = max(Y_val);      % Convert to 1x2000
        
        % Ensure Y is a column vector
        Y_train = Y_train(:);    % Convert to 6000x1
        Y_val = Y_val(:);        % Convert to 2000x1
        
        % Store preprocessing parameters
        preprocessing_params = struct();
        preprocessing_params.input_size = size(X_train, 2);    % 8 features
        preprocessing_params.num_classes = 6;                   % 6 classes
        preprocessing_params.class_labels = 1:6;               % Store class labels
        preprocessing_params.feature_mean = feature_mean;      % Store mean
        preprocessing_params.feature_std = feature_std;        % Store std
        
        % Verify all 6 classes are present
        assert(length(unique(Y_train)) == 6, 'Training data missing some classes');
        assert(length(unique(Y_val)) == 6, 'Validation data missing some classes');
        
    catch ME
        fprintf('Error in data processing:\n');
        fprintf('Error message: %s\n', ME.message);
        fprintf('Error location: %s\n', ME.stack(1).name);
        rethrow(ME);
    end
end