function result = NNClassifierFunction(sampling, k)

%% Initial setup and preprocessing

% Check if k has been passed as a parameter. If no, default to NN
if nargin < 2
    k = [];
end

% Initialise return values
trainingTime = 0;
testingTime = 0;

% Set the folders
posFolder = 'Assets/Images/Pos';
negFolder = 'Assets/Images/Neg';

% Get list of image files 
posFiles = dir(fullfile(posFolder, '*.*'));
negFiles = dir(fullfile(negFolder, '*.*'));

% Filter out '.' and '..' entries IDK why but it breaks without it
posFiles = posFiles(~[posFiles.isdir]);
negFiles = negFiles(~[negFiles.isdir]);

allImages = [];
labels = [];


% Load positive images
counter = 0;
for i = 1:sampling:size(posFiles, 1)
    % Used instead of i as you'd have the first be 1 and the second be 1 +
    % sample rate
    counter = counter +1;

    % Gets the image at i location
    img = imread(fullfile(posFolder, posFiles(i).name));

    % Adds the image to the variable and does some processing to it
    allImages(counter,:) = extractRawPixels(img);

    % As these are all images with pedestrians present, the label is 1 for
    % correct
    labels(counter,1) = 1;
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
% Splits the images and labels between testing and training
[trainImages, trainLabels, testImages, testLabels] = split5050(allImages, labels);

% Trains the model
tic
modelNN = NNTraining(trainImages, trainLabels);
trainingTime = trainingTime + toc;
%disp("The training time for NN is: " + trainingTime)

%% Testing

% Pre allocation cause MatLabs keeps complaining about it
classificationResult = zeros(size(testImages,1),1);
for i = 1:size(testImages,1)

    %Gets all the pixels for the current test image
    currentTestImage = testImages(i,:);
    
    tic
    % Classify the current test image using the trained model
    if isempty(k)
        classificationResult(i,1) = NNTesting(currentTestImage, modelNN);
    else
        classificationResult(i,1) = KNNTesting(currentTestImage, modelNN, k);
    end
    testingTime = testingTime + toc;
end

%disp("The testing time for NN is: " + totalTestTime)

%% Evaluation
TPFPresult = Calculate_Metrics(testImages, testLabels, classificationResult);
result.TP = TPFPresult.TP;
result.TN = TPFPresult.TN;
result.FP = TPFPresult.FP;
result.FN = TPFPresult.FN;

result.accuracy = TPFPresult.accuracy;
result.errorRate = TPFPresult.errorRate;
result.precision = TPFPresult.precision;
result.specificity = TPFPresult.specificity;
result.sensitivity = TPFPresult.sensitivity;
result.fMeasure = TPFPresult.fMeasure;
result.falseAlarmRate = TPFPresult.falseAlarmRate;

result.trainingTime = trainingTime;
result.testingTime = testingTime;

% %We display 25 of the correctly classified images
% figure
% sgtitle('Correct Classification')
% count=0;
% i=1;
% while (count<25)&&(i<=length(comparison))
% 
%     if comparison(i)
%         count=count+1;
%         subplot(5,5,count)
%         Im = reshape(testImages(i,:),160,96);
%         imshow(Im, [])
%         title(num2str(classificationResult(i)))
%     end
% 
%     i=i+1;
% 
% end
% 
% %We display 25 of the incorrectly classified images
% figure
% sgtitle('Wrong Classification')
% count=0;
% i=1;
% while (count<25)&&(i<=length(comparison))
% 
%     if ~comparison(i)
%         count=count+1;
%         subplot(5,5,count)
%         Im = reshape(testImages(i,:),160,96);
%         imshow(Im, [])
%         title(num2str(classificationResult(i)))
%     end
% 
%     i=i+1;
% 
% end

end