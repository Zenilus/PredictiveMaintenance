% 1. Load the data
data = readtable('maintenance_data.csv');

% 2. Select features using the correct column names
features = data(:, {'AirTemperature_K_', 'ProcessTemperature_K_', 'RotationalSpeed_rpm_', ...
    'Torque_Nm_', 'ToolWear_min_'});

% Convert to array
X = table2array(features);
y = data.Target; % Using 'Target' as our prediction target

% 3. Split data into training and testing sets (80-20 split)
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(cv.training,:);
X_test = X(cv.test,:);
y_train = y(cv.training,:);
y_test = y(cv.test,:);

% 4. Train the model with OOBPredictorImportance enabled
model = TreeBagger(100, X_train, y_train, 'Method', 'classification', ...
    'OOBPredictorImportance', 'on');

% 5. Make predictions on test set
%y_pred = predict(model, X_test);

% 6. Evaluate the model
%confusionchart(y_test, str2double(y_pred));

% Calculate accuracy
%accuracy = sum(str2double(y_pred) == y_test) / length(y_test);
%fprintf('Model Accuracy: %.2f%%\n', accuracy * 100);

% Feature importance visualization
%importance = model.OOBPermutedPredictorDeltaError;
%figure;
%bar(importance);
%xticklabels(features.Properties.VariableNames);
%xtickangle(45);
%ylabel('Predictor Importance');
%title('Feature Importance');

% Function to make predictions
function [prediction, probability] = predictMaintenance(model, newData)
    [prediction, scores] = predict(model, newData);
    probability = max(scores, [], 2); % Get the highest probability
end


% Create a new data point for Prediction
newData = [299, 311, 1000, 2.6, 70];
% [AirTemperature_K_, ProcessTemperature_K_, RotationalSpeed_rpm_, Torque_Nm_, ToolWear_min_]

% Make prediction
[predicted_failure, probability] = predictMaintenance(model, newData);
fprintf('\nPrediction for new data:\n');
fprintf('Failure Prediction: %s\n', predicted_failure{1});
fprintf('Probability: %.2f%%\n', probability * 100);

% Create a function to save the model for future use
save('maintenance_model.mat', 'model');

% Example of how to load and use the saved model later
% load('maintenance_model.mat');
