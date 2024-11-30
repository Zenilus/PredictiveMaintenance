% create_network.m
function [net] = create_network(inputSize, numClasses)
    % Create a neural network for predictive maintenance
    % inputSize: number of input features
    % numClasses: number of failure types to predict
    
    % Initialize network layers
    layers = [
        % Input layer
        featureInputLayer(inputSize, 'Name', 'input')
        
        % First hidden layer
        fullyConnectedLayer(128, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.3, 'Name', 'drop1')
        
        % Second hidden layer
        fullyConnectedLayer(64, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        dropoutLayer(0.2, 'Name', 'drop2')
        
        % Output layer
        fullyConnectedLayer(numClasses, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'output')
    ];
    
    % Create the network
    net = layerGraph(layers);
    
    % Display network architecture
    figure('Name', 'Network Architecture');
    plot(net);
    title('Predictive Maintenance Neural Network Architecture');
end