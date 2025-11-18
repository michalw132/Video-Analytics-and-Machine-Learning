clear all
close all

%% Setup parameters
iterations = 5;
sampling = 50;
k = 10;

%% Running the classifier
% Setup results table
results = struct( ...
    'Label', {}, ...
    'trainingTime', {}, ...
    'testingTime', {}, ...
    'accuracy', {}, ...
    'iterations', {}, ...
    'sampling', {}, ...
    'k', {} );

for i = 1:iterations
    r = NNClassifierFunction(sampling, k);
    r.Label = "Run " + i;

    % Leave these empty, we only need to display these values once
    r.iterations = [];
    r.sampling = [];
    r.k = "";

    results(i) = r;
end

% Calculate average values
meanTrainingTime = mean([results.trainingTime]);
meanTestingTime = mean([results.testingTime]);
meanAccuracy = mean([results.accuracy]);

% Summary row
summary.Label = "Mean Values:";
summary.trainingTime = meanTrainingTime;
summary.testingTime = meanTestingTime;
summary.accuracy = meanAccuracy;

% Also include parameters used
summary.iterations = iterations;
summary.sampling = sampling;
summary.k = k;

% Append summary struct to results struct
results(end + 1) = summary;

% Output results as .csv
table = struct2table(results);
writetable(table, 'nnResults.csv');

disp("Classifier completed successfully. csv with results generated at " + fullfile(pwd, 'nnResults.csv'));
disp(table);
