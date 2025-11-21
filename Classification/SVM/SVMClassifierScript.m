clear all
close all
 
%% Setup parameters
iterations = 1;
sampling = 50;

% Training + Testing
kernel = 'gaussian';
% Other kernel options:
% gaussian, poly, polyhomog, htrbf, wavelet, frame
% Note: poly and polyhomog are linear kernels

% Training parameters
lambda = 1e-20;
C = Inf;
sigmakernel = 10;

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
    'kernel', {}, ...
    'lambda', {}, ...
    'C', {}, ...
    'sigmakernel', {} );

for i = 1:iterations
    r = SVMClassifierFunction(sampling, kernel, lambda, C, sigmakernel);
    r.Label = "Run " + i;

    % Leave these empty, we only need to display these values once
    r.iterations = [];
    r.sampling = [];
    r.kernel = "";
    r.lambda = [];
    r.C = [];
    r.sigmakernel = [];

    results(i) = r;
end

% Summary row comprised of mean values
summary = struct();
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
summary.kernel = kernel;
summary.lambda = lambda;
summary.C = C;
summary.sigmakernel = sigmakernel;

% Append summary struct to results struct
results(end + 1) = summary;

% Output results as .csv
table = struct2table(results);
writetable(table, 'svmResults.csv');

disp("Classifier completed successfully. csv with results generated at " + fullfile(pwd, 'svmResults.csv'));
disp(table);
