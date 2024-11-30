% main.m
function main()
    try
        % Initialize session
        session = init_session();
        
        % Training phase
        if session.should_train
            model = train_model(session);
        end
        
        % Evaluation phase
        if session.should_evaluate
            evaluate_model_performance(session, model);
        end
        
        % Clean up
        cleanup_session(session);
        
    catch ME
        handle_error(ME);
    end
end