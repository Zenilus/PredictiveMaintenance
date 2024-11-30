function status = monitor_performance(predictions, true_labels)
    % Initialize status structure
    status = struct();
    
    % Calculate basic statistics
    status.total_samples = length(predictions);
    status.correct_predictions = sum(predictions == true_labels);
    status.accuracy = status.correct_predictions / status.total_samples;
    
    % Get unique classes
    unique_classes = unique(true_labels);
    status.num_classes = length(unique_classes);
    
    % Calculate class distribution
    status.class_distribution = struct();
    for i = 1:status.num_classes
        class_samples = sum(true_labels == unique_classes(i));
        status.class_distribution(i).class = unique_classes(i);
        status.class_distribution(i).count = class_samples;
        status.class_distribution(i).percentage = (class_samples / status.total_samples) * 100;
    end
    
    % Check for significant class imbalance
    class_counts = [status.class_distribution.count];
    status.imbalance_ratio = max(class_counts) / min(class_counts);
    status.is_imbalanced = status.imbalance_ratio > 3; % threshold of 3:1
    
    % Calculate error distribution
    incorrect_indices = find(predictions ~= true_labels);
    status.error_distribution = struct();
    status.error_distribution.total_errors = length(incorrect_indices);
    status.error_distribution.error_rate = length(incorrect_indices) / status.total_samples;
    
    % Print monitoring summary
    fprintf('\nPerformance Monitoring Summary:\n');
    fprintf('Total samples evaluated: %d\n', status.total_samples);
    fprintf('Overall accuracy: %.2f%%\n', status.accuracy * 100);
    fprintf('Class imbalance ratio: %.2f:1\n', status.imbalance_ratio);
    if status.is_imbalanced
        fprintf('Warning: Significant class imbalance detected\n');
    end
end