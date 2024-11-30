function display_results_summary(metrics, status, model)
    fprintf('\n=== Model Evaluation Summary ===\n\n');
    
    % Display overall metrics
    fprintf('Overall Performance:\n');
    fprintf('-------------------\n');
    fprintf('Accuracy: %.2f%%\n', metrics.accuracy * 100);
    fprintf('Macro-F1 Score: %.2f%%\n', metrics.macro_f1 * 100);
    
    % Display class-wise performance
    fprintf('\nPer-Class Performance:\n');
    fprintf('--------------------\n');
    for i = 1:length(metrics.per_class)
        fprintf('Class %d:\n', i);
        fprintf('  Precision: %.2f%%\n', metrics.per_class(i).precision * 100);
        fprintf('  Recall: %.2f%%\n', metrics.per_class(i).recall * 100);
        fprintf('  F1-Score: %.2f%%\n', metrics.per_class(i).f1_score * 100);
    end
    
    % Display confusion matrix
    fprintf('\nConfusion Matrix:\n');
    fprintf('-----------------\n');
    disp(metrics.confusion_matrix);
    
    % Display monitoring status
    fprintf('\nMonitoring Status:\n');
    fprintf('----------------\n');
    fprintf('Total samples: %d\n', status.total_samples);
    fprintf('Class imbalance ratio: %.2f:1\n', status.imbalance_ratio);
    if status.is_imbalanced
        fprintf('Warning: Class imbalance detected!\n');
    end
    
    % Display model information
    fprintf('\nModel Information:\n');
    fprintf('----------------\n');
    fprintf('Number of ensemble models: %d\n', length(model.ensembleNets));
    fprintf('Average ensemble accuracy: %.2f%%\n', mean(model.ensembleAccuracies) * 100);
end