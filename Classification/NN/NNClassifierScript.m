clear all
close all

%% Setup parameters
iterations = 5;
sampling = 5;
k = 10;

%% Running the classifier
% Setup results table
results = struct( ...
    'Label', {}, ...
    'trainingTime', {}, ...
    'testingTime', {}, ...
    'TP', {}, ...
    'TN', {}, ...
    'FP', {}, ...
    'FN', {}, ...
    'accuracy', {}, ...
    'errorRate', {}, ...
    'precision', {}, ...
    'specificity', {}, ...
    'sensitivity', {}, ...
    'fMeasure', {}, ...
    'falseAlarmRate', {}, ...
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

% Summary row comprised of mean values
summary = struct();
summary.Label = "Mean Values:";
summary.Label = "Mean Values:";
summary.trainingTime = mean([results.trainingTime]);
summary.testingTime = mean([results.testingTime]);

summary.TP = mean([results.TP]);
summary.TN = mean([results.TN]);
summary.FP = mean([results.FP]);
summary.FN = mean([results.FN]);

summary.accuracy = mean([results.accuracy]);
summary.errorRate = mean([results.errorRate]);
summary.precision = mean([results.precision]);
summary.specificity = mean([results.specificity]);
summary.sensitivity = mean([results.sensitivity]);
summary.fMeasure = mean([results.fMeasure]);
summary.falseAlarmRate = mean([results.falseAlarmRate]);

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
