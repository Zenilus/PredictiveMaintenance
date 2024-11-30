function status = monitor_performance(predictions, actual, threshold)
    if nargin < 3
        threshold = 0.85;
    end
    
    accuracy = sum(predictions == actual) / numel(actual);
    status = struct();
    status.accuracy = accuracy;
    status.below_threshold = accuracy < threshold;
    status.timestamp = datetime('now', 'TimeZone', 'UTC');
    
    if status.below_threshold
        warning('Model accuracy has dropped below threshold: %.2f%%', accuracy*100);
        % Log the event
        log_performance_drop(accuracy, threshold, status.timestamp);
    end
end

function log_performance_drop(accuracy, threshold, timestamp)
    % Create logs directory if it doesn't exist
    if ~exist('logs', 'dir')
        mkdir('logs');
    end
    
    % Log file path
    logFile = 'logs/performance_drops.log';
    
    % Create log message
    logMsg = sprintf('[%s] Accuracy DROP: %.2f%% (Threshold: %.2f%%)\n', ...
        char(timestamp), accuracy*100, threshold*100);
    
    % Append to log file
    fid = fopen(logFile, 'a');
    fprintf(fid, logMsg);
    fclose(fid);
end