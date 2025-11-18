%function [outputArg1,outputArg2] = SVMClassifierFunction(sampling)
function result = SVMClassifierFunction(sampling, kernel, lambda, C, sigmakernel)


%% Initial setup and preprocessing
% Initialise return values
trainingTime = 0;
testingTime = 0;
accuracy = 0;

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

% Sets the sample rate, the higher the faster it will run
%sampling = 50;

% Load positive images
counter = 0;
for i = 1:sampling:length(posFiles)
    % Used instead of i as you'd have the first be 1 and the second be 1 +
    % sample rate
    counter = counter + 1;

    % Gets the image at i location
    img = imread(fullfile(posFolder, posFiles(i).name));

    % Adds the image to the variable and does some processing to it
    allImages(counter, :) = extractRawPixels(img);

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
    allImages(posImageTotal + counter,:) = extractRawPixels(img);
    labels(posImageTotal + counter,1) = 0;
end

%% Training
% Split data into training and testing sets
[trainImages, trainLabels, testImages, testLabels] = split5050(allImages, labels);

tic
% Trains the model
modelSVM = SVMTraining(trainImages, trainLabels, kernel, lambda, C, sigmakernel);
trainingTime = trainingTime + toc;
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
%% Evaluation
comparison = (testLabels == classificationResult);
accuracy = sum(comparison) / length(comparison);
%disp(['Accuracy: ', num2str(accuracy)]);

% Set return values
result.trainingTime = trainingTime;
result.testingTime = testingTime;
result.accuracy = accuracy;
end

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

%save SVMModel modelSVM
