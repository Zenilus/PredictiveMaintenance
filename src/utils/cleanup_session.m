function cleanup_session(session)
    % Clean up parallel pool
    delete(gcp('nocreate'));
    fprintf('\nParallel pool cleaned up.\n');
    
    % Calculate session duration
    session_duration = datetime('now', 'TimeZone', 'UTC') - session.start_time;
    fprintf('Session completed in %.2f minutes\n', minutes(session_duration));
end