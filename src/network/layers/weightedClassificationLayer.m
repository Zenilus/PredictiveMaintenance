classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    properties
        ClassWeights
    end
    
    methods
        function layer = weightedClassificationLayer(classWeights)
            % Set layer name
            layer.Name = 'weightedClassification';
            
            % Validate and set class weights
            validateattributes(classWeights, {'numeric'}, ...
                {'vector', 'positive', 'finite', 'real'});
            layer.ClassWeights = classWeights(:)'; % Ensure row vector
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Y: Predictions from the network (N x numClasses)
            % T: Targets in one-hot encoding (N x numClasses)
            
            N = size(Y, 1);
            numClasses = size(Y, 2);
            
            % Ensure class weights match the number of classes
            assert(length(layer.ClassWeights) == numClasses, ...
                'Number of class weights must match number of classes');
            
            % Safe matrix multiplication for weighted loss
            loss = -sum(sum(T .* log(Y + eps) .* layer.ClassWeights, 2)) / N;
        end
    end
end