function data = load_data()\n    % Load data from CSV file\n    filename = fullfile('data', 'raw', 'predictive_maintenance.csv');\n    data = readtable(filename);\nend\n