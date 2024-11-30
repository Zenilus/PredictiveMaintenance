function save_evaluation_results(metrics, status, session)
    % Create a timestamp for the filename
    timestamp = datestr(session.start_time, 'yyyy-mm-dd_HH-MM-SS');
    
    % Create the results filename
    resultsFileName = sprintf('results/evaluation_%s.mat', timestamp);
    
    % Prepare results structure
    results = struct(...
        'metrics', metrics,...
        'status', status,...
        'timestamp', session.start_time,...
        'evaluated_by', session.user,...
        'evaluation_date', datetime('now', 'TimeZone', 'UTC')...
    );
    
    % Save the results
    save(resultsFileName, '-struct', 'results');
    
    % Print confirmation message
    fprintf('\nEvaluation results saved as: %s\n', resultsFileName);
end