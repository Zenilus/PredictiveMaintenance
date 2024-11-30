function session = init_session()
    session = struct();
    
    % Session info
    session.start_time = datetime('now', 'TimeZone', 'UTC');
    session.user = 'Zenilus';
    fprintf('Starting session at %s UTC\n', char(session.start_time));
    fprintf('User: %s\n\n', session.user);
    
    % Add paths
    addpath(genpath('src'));
    
    % Initialize parallel pool
    if isempty(gcp('nocreate'))
        numCores = feature('numcores');
        fprintf('Number of CPU cores detected: %d\n', numCores);
        numWorkersToUse = max(1, numCores - 1);
        fprintf('Setting up parallel pool with %d workers...\n', numWorkersToUse);
        session.pool = parpool('local', numWorkersToUse);
        fprintf('Parallel pool created successfully.\n\n');
    else
        session.pool = gcp;
        fprintf('Using existing parallel pool with %d workers.\n\n', session.pool.NumWorkers);
    end
    
    % Set flags for pipeline stages
    session.should_train = true;      % Set to false if you only want to evaluate
    session.should_evaluate = true;
    
    % Create necessary directories
    dirs = {'models', 'logs', 'results'};
    for dir = dirs
        if ~exist(dir{1}, 'dir')
            mkdir(dir{1});
        end
    end
end