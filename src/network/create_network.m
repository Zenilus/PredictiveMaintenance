% create_network.m
function [net] = create_network(inputSize, numClasses)
    layers = [
        % Input layer
        featureInputLayer(inputSize, 'Name', 'input')
        
        % First hidden layer
        fullyConnectedLayer(256, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.2, 'Name', 'drop1')
        
        % Second hidden layer
        fullyConnectedLayer(128, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.2, 'Name', 'drop2')
        
        % Third hidden layer
        fullyConnectedLayer(64, 'Name', 'fc3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        dropoutLayer(0.1, 'Name', 'drop3')
        
        % Output layer - ensure this matches the number of classes in your data
        fullyConnectedLayer(numClasses, 'Name', 'fc_out')  % This will now be 6 instead of 7
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    net = layerGraph(layers);
end