classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    properties
        ClassWeights
    end
    
    methods
        function layer = weightedClassificationLayer(classWeights)
            % Create a weighted classification layer
            % classWeights: Vector of weights for each class
            
            % Set layer name
            layer.Name = 'weightedClassification';
            
            % Validate and set class weights
            validateattributes(classWeights, {'numeric'}, ...
                {'vector', 'positive', 'finite', 'real'});
            layer.ClassWeights = classWeights(:)'; % Ensure row vector
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Y: Predictions from the network (N x numClasses)
            % T: Target labels in categorical form (N x numClasses)
            
            % Get number of observations
            N = size(Y, 1);
            
            % Convert categorical targets to indices if necessary
            if iscategorical(T)
                T = onehotencode(T, 2);
            end
            
            % Apply class weights to the negative log likelihood loss
            [~, classIdx] = max(T, [], 2);
            weights = layer.ClassWeights(classIdx);
            
            % Compute weighted cross-entropy loss
            loss = -sum(weights .* sum(T .* log(Y + eps), 2)) / N;
        end
    end
end