function model = train_ensemble(X_train, Y_train, X_val, Y_val, preprocessing_params, session)
    % Prepare data format
    [X_train, Y_train, X_val, Y_val, Y_train_cat, Y_val_cat] = prepare_data_format(X_train, Y_train, X_val, Y_val);
    
    % Initialize ensemble
    numEnsemble = 3;
    model.ensembleNets = cell(numEnsemble, 1);
    model.ensembleAccuracies = zeros(numEnsemble, 1);
    model.preprocessing = preprocessing_params;
    
    % Create network architecture and training options
    [layers, options] = create_network_config(X_train, Y_train_cat, X_val, Y_val_cat);
    
    % Train ensemble models
    fprintf('\nStarting ensemble training...\n');
    tic;
    
    for e = 1:numEnsemble
        [model.ensembleNets{e}, model.ensembleAccuracies(e)] = train_single_model(e, X_train, Y_train_cat, X_val, Y_val_cat, layers, options);
    end
    
    model.trainingTime = toc;
    model.timestamp = session.start_time;
    model.user = session.user;
end