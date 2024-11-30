function save_trained_model(model, session)
    % Create a timestamp for the filename
    timestamp = datestr(session.start_time, 'yyyy-mm-dd_HH-MM-SS');
    
    % Create the model filename
    modelFileName = sprintf('models/ensemble_model_%s.mat', timestamp);
    
    % Save the model structure
    save(modelFileName, '-struct', 'model');
    
    % Print confirmation message
    fprintf('\nModel saved successfully as: %s\n', modelFileName);
    
    % Save additional metadata
    metadata = struct();
    metadata.saved_by = session.user;
    metadata.save_date = datetime('now', 'TimeZone', 'UTC');
    metadata.model_performance = struct(...
        'training_time_minutes', model.trainingTime/60,...
        'ensemble_accuracies', model.ensembleAccuracies,...
        'mean_accuracy', mean(model.ensembleAccuracies)...
    );
    
    % Save metadata alongside model
    metadataFileName = sprintf('models/ensemble_model_%s_metadata.mat', timestamp);
    save(metadataFileName, '-struct', 'metadata');
    
    fprintf('Model metadata saved as: %s\n', metadataFileName);
end