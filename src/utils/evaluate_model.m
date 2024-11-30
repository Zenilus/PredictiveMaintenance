function metrics = evaluate_model(Y_true, Y_pred, numClasses)
    % Calculate confusion matrix
    confMat = confusionmat(Y_true, Y_pred);
    
    % Plot confusion matrix
    figure('Name', 'Confusion Matrix');
    confusionchart(confMat, categorical(1:numClasses));
    title('Validation Set Confusion Matrix');
    
    % Calculate metrics
    metrics = struct();
    metrics.per_class = struct('precision', [], 'recall', [], 'f1', []);
    
    for i = 1:numClasses
        precision = confMat(i,i) / sum(confMat(:,i));
        recall = confMat(i,i) / sum(confMat(i,:));
        f1 = 2 * (precision * recall) / (precision + recall);
        
        metrics.per_class.precision(i) = precision;
        metrics.per_class.recall(i) = recall;
        metrics.per_class.f1(i) = f1;
        
        fprintf('Class %d metrics:\n', i);
        fprintf('Precision: %.2f%%\n', precision*100);
        fprintf('Recall: %.2f%%\n', recall*100);
        fprintf('F1-Score: %.2f%%\n\n', f1*100);
    end
    
    % Overall metrics
    metrics.overall_accuracy = sum(diag(confMat)) / sum(confMat(:));
    metrics.confusion_matrix = confMat;
end