function model = train_ensemble(X_train, Y_train, X_val, Y_val, preprocessing_params, session)
    % Debug: Print dimensions at entry point
    fprintf('Dimensions in train_ensemble:\n');
    fprintf('X_train: %dx%d\n', size(X_train));
    fprintf('Y_train: %dx%d\n', size(Y_train));
    fprintf('X_val: %dx%d\n', size(X_val));
    fprintf('Y_val: %dx%d\n', size(Y_val));
    
    % Create parallel pool for CPU-based processing
    if isempty(gcp('nocreate'))
        parpool('local', feature('numcores'));
    end
    
    % Verify dimensions match
    assert(size(X_train, 1) == length(Y_train), ...
        sprintf('Training data mismatch: X_train has %d samples, Y_train has %d labels', ...
        size(X_train, 1), length(Y_train)));
    
    % Get network configuration
    [layers, options] = create_network_config(X_train, Y_train, X_val, Y_val);
    
    % Initialize ensemble
    numModels = 3;
    nets = cell(1, numModels);
    accuracies = zeros(1, numModels);
    
    % Train ensemble models in parallel
    parfor e = 1:numModels
        fprintf('\nTraining ensemble model %d/%d\n', e, numModels);
        [net, accuracy] = train_single_model(e, X_train, Y_train, X_val, Y_val, layers, options);
        nets{e} = net;
        accuracies(e) = accuracy;
    end
    
    % Create model structure
    model = struct();
    model.ensembleNets = nets;
    model.ensembleAccuracies = accuracies;
    model.preprocessing = preprocessing_params;
    
    % Display results
    fprintf('\nEnsemble Training Complete:\n');
    fprintf('Average Validation Accuracy: %.2f%%\n', mean(accuracies) * 100);
    fprintf('Std Dev of Validation Accuracy: %.2f%%\n', std(accuracies) * 100);
end