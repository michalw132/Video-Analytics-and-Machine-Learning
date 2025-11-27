function [result] = SVMClassifierFunction(sampling, kernel, lambda, C, sigmakernel, k)


%% Initial setup
% Initialise return values
trainingTime = 0;
testingTime = 0;

% Set the folders
posFolder = 'Assets/Images/Pos';
negFolder = 'Assets/Images/Neg';

% Get list of image files
posFiles = dir(fullfile(posFolder, '*.*'));
negFiles = dir(fullfile(negFolder, '*.*'));

% Filter out '.' and '..' entries
posFiles = posFiles(~[posFiles.isdir]);
negFiles = negFiles(~[negFiles.isdir]);

% Initialize variables
allImages = [];
labels = [];

% Load positive images
counter = 0;
for i = 1:sampling:length(posFiles)
    % Used instead of i as you'd have the first be 1 and the second be 1 +
    % sample rate
    counter = counter + 1;

    % Gets the image at i location
    img = imread(fullfile(posFolder, posFiles(i).name));

    % Adds the image to the variable and does some processing to it
    allImages(counter, :) = extractHOGVector(img);

    % As these are all images with pedestrians present, the label is 1 for
    % correct
    labels(counter, 1) = 1;
end

%How many positive images are currently in the allImages variable
posImageTotal = size(allImages, 1);

% Load negative images - only thing different to positive images is the
% global images variable starts where the positive one left off
counter = 0;
for i = 1:sampling:size(negFiles, 1)
    counter = counter +1;
    img = imread(fullfile(negFolder, negFiles(i).name));
    allImages(posImageTotal + counter,:) = extractHOGVector(img);
    labels(posImageTotal + counter,1) = 0;
end


%% Training
% Split data into training and testing sets

% Use cross-validation instead of 50/50 split
cvFolds = splitCrossValidation(allImages, labels, k);

% Initialize accumulators
TPacc = 0; TNacc = 0; FPacc = 0; FNacc = 0;
accuracyAcc = 0; errorRateAcc = 0; precisionAcc = 0;
specificityAcc = 0; sensitivityAcc = 0; fMeasureAcc = 0;
falseAlarmAcc = 0;

for fold = 1:k
    trainImages = cvFolds(fold).trainImages;
    trainLabels = cvFolds(fold).trainLabels;
    testImages  = cvFolds(fold).testImages;
    testLabels  = cvFolds(fold).testLabels;

    % Preprocessing
    % Z-score standardization
    % Prevents features with large numeric scales from dominating
    mu = mean(trainImages, 1);
    sigma = std(trainImages, 0, 1);
    sigma(sigma == 0) = 1;

    trainImages = (trainImages - mu) ./ sigma;
    testImages = (testImages - mu) ./ sigma;

    % L2-normalize each HOG vector
    % Makes HOG descriptors comparable across images
    trainImages = trainImages ./ vecnorm(trainImages, 2, 2);
    testImages  = testImages  ./ vecnorm(testImages, 2, 2);

    % Replace NaNs (if any rows become 0 vector after normalization)
    % Shouldn't happen, but best to stay on the safe side
    trainImages(any(isnan(trainImages),2), :) = 0;
    testImages(any(isnan(testImages),2), :) = 0;

    tic
    % Trains the model
    modelSVM = SVMTraining(trainImages, trainLabels, kernel, lambda, C, sigmakernel);
    trainingTime = trainingTime + toc;

    modelSVM.mu = mu;
    modelSVM.sigma = sigma;
    
    %disp("Training time: " + trainingTime)

    %% Testing
    classificationResult = zeros(size(testImages,1),1);

    for i = 1:size(testImages,1)
        currentTestImage = testImages(i,:);

        tic
        classificationResult(i,1) = SVMTesting(currentTestImage, modelSVM, kernel);
        testingTime = testingTime + toc;
    end

    %disp("Testing time: " + testingTime)

    TPFPresult = TPFP(testImages, testLabels, classificationResult);

    % Accumulate metrics across folds
    TPacc = TPacc + TPFPresult.TP;
    TNacc = TNacc + TPFPresult.TN;
    FPacc = FPacc + TPFPresult.FP;
    FNacc = FNacc + TPFPresult.FN;
    accuracyAcc = accuracyAcc + TPFPresult.accuracy;
    errorRateAcc = errorRateAcc + TPFPresult.errorRate;
    precisionAcc = precisionAcc + TPFPresult.precision;
    specificityAcc = specificityAcc + TPFPresult.specificity;
    sensitivityAcc = sensitivityAcc + TPFPresult.sensitivity;
    fMeasureAcc = fMeasureAcc + TPFPresult.fMeasure;
    falseAlarmAcc = falseAlarmAcc + TPFPresult.falseAlarmRate;

end


%% Evaluation
result.TP = TPacc / k;
result.TN = TNacc / k;
result.FP = FPacc / k;
result.FN = FNacc / k;

result.accuracy = accuracyAcc / k;
result.errorRate = errorRateAcc / k;
result.precision = precisionAcc / k;
result.specificity = specificityAcc / k;
result.sensitivity = sensitivityAcc / k;
result.fMeasure = fMeasureAcc / k;
result.falseAlarmRate = falseAlarmAcc / k;

result.trainingTime = trainingTime / k;
result.testingTime = testingTime / k;

save SVMModel modelSVM;

% % Display correctly classified images
% figure
% sgtitle('Correct Classification (SVM)')
% count = 0;
% i = 1;
% while (count < 25) && (i <= length(comparison))
%     if comparison(i)
%         count = count + 1;
%         subplot(5,5,count)
%         Im = reshape(testImages(i,:),160,96);
%         imshow(Im, [])
%         title(num2str(classificationResult(i)))
%     end
%     i = i + 1;
% end
%
% % Display incorrectly classified images
% figure
% sgtitle('Wrong Classification (SVM)')
% count = 0;
% i = 1;
% while (count < 25) && (i <= length(comparison))
%     if ~comparison(i)
%         count = count + 1;
%         subplot(5,5,count)
%         Im = reshape(testImages(i,:),160,96);
%         imshow(Im, [])
%         title(num2str(classificationResult(i)))
%     end
%     i = i + 1;
% end

end