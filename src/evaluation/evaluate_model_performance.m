function evaluate_model_performance(session, model)
    fprintf('\nEvaluating model performance...\n');
    
    % Load validation data if not in workspace
    if ~exist('X_val', 'var') || ~exist('Y_val', 'var')
        load('data/processed/prepared_data.mat', 'X_val', 'Y_val');
    end
    
    % Convert Y_val to indices if it's in one-hot encoding format
    [~, Y_val_idx] = max(Y_val, [], 1);
    
    % Make predictions
    predictions = predict_with_ensemble(X_val, model.ensembleNets, model.preprocessing);
    
    % Calculate metrics
    metrics = evaluate_model(Y_val_idx, predictions, numel(unique(Y_val_idx)));
    
    % Monitor performance
    status = monitor_performance(predictions, Y_val_idx);
    
    % Save and display results
    save_evaluation_results(metrics, status, session);
    display_results_summary(metrics, status, model);
end