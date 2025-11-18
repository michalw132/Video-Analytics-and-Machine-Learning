clear all
close all

%% Setup parameters
iterations = 5;
sampling = 50;

% Training + Testing
kernel = 'poly';
% other svmkernel kernel options:
% gaussian, poly, polyhomog, htrbf, wavelet, frame
% poly and polyhomog are linear kernels

% Training parameters
lambda = 1e-20;
C = Inf;
sigmakernel = 10;

%will need to save the above valuyes into the csv aswell...

%% Running the classifier
% Setup results table
results = struct( ...
    'Label', {}, ...
    'trainingTime', {}, ...
    'testingTime', {}, ...
    'accuracy', {}, ...
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
    r.iterations  = [];
    r.sampling    = [];
    r.kernel      = "";
    r.lambda      = [];
    r.C           = [];
    r.sigmakernel = [];

    results(i) = r;
end

% Calculate average values
meanTrainingTime = mean([results.trainingTime]);
meanTestingTime  = mean([results.testingTime]);
meanAccuracy     = mean([results.accuracy]);

% Summary row
summary.Label = "Mean Values:";
summary.trainingTime = meanTrainingTime;
summary.testingTime  = meanTestingTime;
summary.accuracy     = meanAccuracy;

% Also include parameters used
summary.iterations  = iterations;
summary.sampling    = sampling;
summary.kernel      = kernel;
summary.lambda      = lambda;
summary.C           = C;
summary.sigmakernel = sigmakernel;

% Append summary struct to results struct
results(end + 1) = summary;

% Output results as .csv
T = struct2table(results);
writetable(T, 'svmResults.csv');

disp("Classifier completed successfully. csv with results generated at " + fullfile(pwd, 'svmResults.csv'));
