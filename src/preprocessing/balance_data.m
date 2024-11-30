% balance_data.m
function [X_balanced, Y_balanced] = balance_data(X, Y, target_samples)
    if nargin < 3
        % Increase default target samples
        target_samples = 500; % Much more samples for better learning
    end
    
    % Get class distribution
    [~, Y_idx] = max(Y, [], 1);
    classes = unique(Y_idx);
    
    % Count samples per class
    fprintf('Original class distribution:\n');
    class_counts = zeros(length(classes), 1);
    for i = 1:length(classes)
        class_counts(i) = sum(Y_idx == classes(i));
        fprintf('Class %d: %d samples (%.2f%%)\n', ...
            classes(i), class_counts(i), (class_counts(i)/length(Y_idx))*100);
    end
    
    % Initialize balanced arrays
    X_balanced = [];
    Y_balanced = [];
    
    fprintf('\nBalancing to target %d samples per class using advanced SMOTE\n', target_samples);
    
    % Balance each class
    for i = 1:length(classes)
        class_indices = find(Y_idx == classes(i));
        current_size = length(class_indices);
        X_class = X(:, class_indices);
        Y_class = Y(:, class_indices);
        
        if current_size > target_samples
            % Undersample majority class using random selection
            selected_idx = randsample(current_size, target_samples);
            X_class = X_class(:, selected_idx);
            Y_class = Y_class(:, selected_idx);
        else
            % Oversample minority class using SMOTE with k-nearest neighbors
            while size(X_class, 2) < target_samples
                for idx = 1:current_size
                    if size(X_class, 2) >= target_samples
                        break;
                    end
                    
                    % Find k nearest neighbors
                    k = min(5, current_size - 1);
                    distances = zeros(current_size, 1);
                    for j = 1:current_size
                        if j ~= idx
                            distances(j) = norm(X_class(:,idx) - X_class(:,j));
                        else
                            distances(j) = inf;
                        end
                    end
                    [~, neighbor_indices] = sort(distances);
                    neighbor_indices = neighbor_indices(1:k);
                    
                    % Generate synthetic samples
                    for neighbor_idx = neighbor_indices'
                        if size(X_class, 2) >= target_samples
                            break;
                        end
                        
                        % Create synthetic sample with random interpolation
                        alpha = rand();
                        synthetic_x = X_class(:,idx) + ...
                            alpha * (X_class(:,neighbor_idx) - X_class(:,idx));
                        
                        % Add noise to prevent exact duplicates
                        noise = randn(size(synthetic_x)) * 0.01;
                        synthetic_x = synthetic_x + noise;
                        
                        % Add synthetic sample
                        X_class = [X_class synthetic_x];
                        Y_class = [Y_class Y_class(:,1)];
                    end
                end
            end
        end
        
        % Add balanced class data to final dataset
        X_balanced = [X_balanced X_class];
        Y_balanced = [Y_balanced Y_class];
    end
    
    % Shuffle the balanced dataset
    shuffle_idx = randperm(size(X_balanced, 2));
    X_balanced = X_balanced(:, shuffle_idx);
    Y_balanced = Y_balanced(:, shuffle_idx);
    
    % Print new class distribution
    [~, Y_balanced_idx] = max(Y_balanced, [], 1);
    fprintf('\nNew class distribution:\n');
    for i = 1:length(classes)
        new_count = sum(Y_balanced_idx == classes(i));
        fprintf('Class %d: %d samples (%.2f%%)\n', ...
            classes(i), new_count, (new_count/length(Y_balanced_idx))*100);
    end
end