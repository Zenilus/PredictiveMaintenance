function [X_balanced, Y_balanced] = balance_data_smote(X, Y, target_size_per_class)
    % Get class indices
    [~, Y_idx] = max(Y, [], 1);
    classes = unique(Y_idx);
    num_classes = length(classes);
    
    % Initialize balanced data arrays
    X_balanced = [];
    Y_balanced = [];
    
    for i = 1:num_classes
        % Get samples for current class
        class_indices = find(Y_idx == classes(i));
        X_class = X(:, class_indices);
        Y_class = Y(:, class_indices);
        
        current_size = size(X_class, 2);
        
        if current_size < target_size_per_class
            % Apply SMOTE for minority classes
            [X_synthetic, Y_synthetic] = generate_synthetic_samples(X_class, Y_class, target_size_per_class - current_size);
            X_class = [X_class, X_synthetic];
            Y_class = [Y_class, Y_synthetic];
        elseif current_size > target_size_per_class
            % Undersample majority classes using informed undersampling
            selected_indices = informed_undersampling(X_class, target_size_per_class);
            X_class = X_class(:, selected_indices);
            Y_class = Y_class(:, selected_indices);
        end
        
        % Add balanced class data to final arrays
        X_balanced = [X_balanced, X_class];
        Y_balanced = [Y_balanced, Y_class];
    end
    
    % Shuffle the balanced dataset
    shuffle_idx = randperm(size(X_balanced, 2));
    X_balanced = X_balanced(:, shuffle_idx);
    Y_balanced = Y_balanced(:, shuffle_idx);
end

function [X_synthetic, Y_synthetic] = generate_synthetic_samples(X, Y, n_samples)
    [n_features, n_samples_orig] = size(X);
    X_synthetic = zeros(n_features, n_samples);
    Y_synthetic = zeros(size(Y, 1), n_samples);
    
    % For each new synthetic sample
    for i = 1:n_samples
        % Randomly select a sample
        idx = randi(n_samples_orig);
        base_sample = X(:, idx);
        base_label = Y(:, idx);
        
        % Find k nearest neighbors
        k = min(5, n_samples_orig);
        distances = sum((X - base_sample).^2, 1);
        [~, neighbor_indices] = sort(distances);
        neighbor_idx = neighbor_indices(randi([2, k+1])); % Exclude self
        
        % Generate synthetic sample
        random_weight = rand();
        synthetic_sample = base_sample + random_weight * (X(:, neighbor_idx) - base_sample);
        
        % Store synthetic sample
        X_synthetic(:, i) = synthetic_sample;
        Y_synthetic(:, i) = base_label;
    end
end

function selected_indices = informed_undersampling(X, target_size)
    [~, n_samples] = size(X);
    
    % Compute density-based scores
    scores = compute_sample_importance(X);
    
    % Select samples based on importance scores
    [~, sorted_indices] = sort(scores, 'descend');
    selected_indices = sorted_indices(1:target_size);
end

function importance_scores = compute_sample_importance(X)
    [~, n_samples] = size(X);
    importance_scores = zeros(1, n_samples);
    
    % Compute pairwise distances
    distances = pdist2(X', X');
    
    % Compute density-based importance
    k = min(5, n_samples-1); % number of neighbors to consider
    for i = 1:n_samples
        % Sort distances for current sample
        [sorted_dist, ~] = sort(distances(i,:));
        % Use k nearest neighbors (excluding self)
        knn_dist = sorted_dist(2:k+1);
        % Compute density score (inverse of average distance to neighbors)
        importance_scores(i) = 1 / mean(knn_dist);
    end
end