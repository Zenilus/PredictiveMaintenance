function handle_error(ME)
    % Clean up parallel pool
    delete(gcp('nocreate'));
    
    % Log error
    fprintf('\nError occurred:\n');
    fprintf('Message: %s\n', ME.message);
    fprintf('Location: %s\n', ME.stack(1).name);
    
    % Create error log
    logFile = sprintf('logs/error_%s.log', datestr(now, 'yyyy-mm-dd_HH-MM-SS'));
    fid = fopen(logFile, 'w');
    fprintf(fid, 'Error Details:\n');
    fprintf(fid, 'Time: %s\n', datestr(now));
    fprintf(fid, 'Message: %s\n', ME.message);
    fprintf(fid, 'Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf(fid, '%s: line %d\n', ME.stack(i).name, ME.stack(i).line);
    end
    fclose(fid);
    
    % Rethrow error
    rethrow(ME);
end