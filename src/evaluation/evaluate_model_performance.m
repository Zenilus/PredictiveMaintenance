function metrics = evaluate_model_performance(session, model)
    % Load validation data
    [~, ~, X_val, Y_val, ~] = load_and_preprocess_data(session);
    
    % Convert validation labels to categorical
    Y_val_cat = categorical(Y_val);
    
    % Get predictions from ensemble
    [predictions, probabilities] = predict_with_ensemble(X_val, model.ensembleNets, model.preprocessing);
    
    % Calculate accuracy
    accuracy = sum(predictions == Y_val_cat) / numel(Y_val_cat);
    
    % Calculate confusion matrix
    C = confusionmat(Y_val_cat, predictions);
    
    % Calculate per-class metrics
    num_classes = size(probabilities, 2);
    precision = zeros(num_classes, 1);
    recall = zeros(num_classes, 1);
    f1_score = zeros(num_classes, 1);
    
    for i = 1:num_classes
        tp = C(i,i);
        fp = sum(C(:,i)) - tp;
        fn = sum(C(i,:)) - tp;
        
        precision(i) = tp / (tp + fp);
        recall(i) = tp / (tp + fn);
        f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
    end
    
    % Handle NaN values
    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1_score(isnan(f1_score)) = 0;
    
    % Store metrics
    metrics = struct();
    metrics.accuracy = accuracy;
    metrics.confusion_matrix = C;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.f1_score = f1_score;
    
    % Display results
    fprintf('\nModel Evaluation Results:\n');
    fprintf('Overall Accuracy: %.2f%%\n', accuracy * 100);
    
    fprintf('\nPer-class Performance:\n');
    fprintf('Class\tPrecision\tRecall\t\tF1-Score\n');
    for i = 1:num_classes
        fprintf('%d\t%.2f%%\t\t%.2f%%\t\t%.2f%%\n', ...
            i, precision(i)*100, recall(i)*100, f1_score(i)*100);
    end
    
    % Display confusion matrix
    fprintf('\nConfusion Matrix:\n');
    disp(C);
end