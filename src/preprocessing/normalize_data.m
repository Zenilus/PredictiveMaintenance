% normalize_data.m
function [X_norm, params] = normalize_data(X, params)
    % Normalize data using z-score normalization (mean=0, std=1)
    % Input:
    %   X: Input data matrix (features in columns, samples in rows)
    %   params: (optional) Struct containing mean and std from training data
    % Output:
    %   X_norm: Normalized data
    %   params: Struct containing mean and std used for normalization
    
    if nargin < 2 || isempty(params)
        % Calculate parameters from the data (training mode)
        params.mean = mean(X, 1);
        params.std = std(X, 1);
        
        % Handle constant features (std = 0)
        params.std(params.std == 0) = 1;
        
        fprintf('Normalization parameters calculated from training data\n');
    else
        fprintf('Using provided normalization parameters\n');
    end
    
    % Apply normalization
    X_norm = (X - params.mean) ./ params.std;
    
    % Check for any potential issues
    if any(isnan(X_norm(:)))
        warning('NaN values detected in normalized data');
    end
    if any(isinf(X_norm(:)))
        warning('Inf values detected in normalized data');
    end
end