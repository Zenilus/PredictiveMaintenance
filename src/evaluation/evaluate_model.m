function metrics = evaluate_model(Y_true, Y_pred, num_classes)
    % Initialize metrics structure
    metrics = struct();
    
    % Calculate confusion matrix
    conf_matrix = zeros(num_classes);
    for i = 1:length(Y_true)
        conf_matrix(Y_true(i), Y_pred(i)) = conf_matrix(Y_true(i), Y_pred(i)) + 1;
    end
    metrics.confusion_matrix = conf_matrix;
    
    % Calculate accuracy
    metrics.accuracy = sum(Y_true == Y_pred) / length(Y_true);
    
    % Calculate per-class metrics
    metrics.per_class = struct();
    for i = 1:num_classes
        % True Positives, False Positives, False Negatives
        TP = conf_matrix(i,i);
        FP = sum(conf_matrix(:,i)) - TP;
        FN = sum(conf_matrix(i,:)) - TP;
        
        % Precision
        if (TP + FP) == 0
            metrics.per_class(i).precision = 0;
        else
            metrics.per_class(i).precision = TP / (TP + FP);
        end
        
        % Recall
        if (TP + FN) == 0
            metrics.per_class(i).recall = 0;
        else
            metrics.per_class(i).recall = TP / (TP + FN);
        end
        
        % F1 Score
        if (metrics.per_class(i).precision + metrics.per_class(i).recall) == 0
            metrics.per_class(i).f1_score = 0;
        else
            metrics.per_class(i).f1_score = 2 * ...
                (metrics.per_class(i).precision * metrics.per_class(i).recall) / ...
                (metrics.per_class(i).precision + metrics.per_class(i).recall);
        end
    end
    
    % Calculate macro-averaged metrics
    metrics.macro_precision = mean([metrics.per_class.precision]);
    metrics.macro_recall = mean([metrics.per_class.recall]);
    metrics.macro_f1 = mean([metrics.per_class.f1_score]);
    
    % Print summary
    fprintf('\nModel Evaluation Metrics:\n');
    fprintf('Overall Accuracy: %.2f%%\n', metrics.accuracy * 100);
    fprintf('Macro-averaged Precision: %.2f%%\n', metrics.macro_precision * 100);
    fprintf('Macro-averaged Recall: %.2f%%\n', metrics.macro_recall * 100);
    fprintf('Macro-averaged F1-Score: %.2f%%\n', metrics.macro_f1 * 100);
end