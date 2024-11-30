function [X, Y] = load_data(session)
    % Load your data here
    % Example:
    % data = load('your_data_file.mat');
    % X = data.features;
    % Y = data.labels;
    
    % Print dimensions of loaded data
    fprintf('Data dimensions in load_data:\n');
    fprintf('X: %dx%d\n', size(X));
    fprintf('Y: %dx%d\n', size(Y));
    
    % Ensure X has samples as rows
    if size(X, 2) > size(X, 1)
        X = X';
        fprintf('Transposed X to have samples as rows\n');
    end
    
    % Ensure Y is a column vector
    Y = Y(:);
end