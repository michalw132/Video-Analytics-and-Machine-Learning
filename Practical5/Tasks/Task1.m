
close all;

% step 1
[images, labels] =  loadPedestrianDatabase('pedestrian_train.cdataset', 1);

path = ('images/neg');

im = imread('images/pos/img00003.jpg');

% step 2
hog = hog_feature_vector(im);
size(im);

% step 3
figure
subplot(1,2,1)
imshow(im)
subplot(1,2,2)
showHog(hog, [160,96]);

% step 4
totalHogs = [];

for i=1:size(images,1)
    im = reshape(images(i,:), [160, 96]);

    thisHog = hog_feature_vector (im);
    totalHogs = [totalHogs; thisHog];
end

% step 5
tic
SVMModel = SVMTraining(totalHogs, labels);
toc


% step 7 (step 6 optional)
[testImages, testLabels] =  loadPedestrianDatabase('pedestrian_test.cdataset', 1);

totalHogs = [];

for i=1:size(testImages,1)
    im = reshape(testImages(i,:), [160, 96]);
    thisHog = hog_feature_vector (im);
    totalHogs = [totalHogs; thisHog];
end

classificationResult = [];

tic
for i=1:size(totalHogs,1)
    classificationResult(end+1,1) = SVMTesting(totalHogs(i,:),SVMModel);
end
toc


% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the esting image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the number of
%correctly classified samples/ the total number of tested samples
Accuracy = (sum(comparison)/length(comparison))

% Compute the weighted average HOG pattern
modelHog = sum((SVMModel.alpha .* labels(SVMModel.pos)) .* SVMModel.xsup, 1);
modelHog = modelHog / norm(modelHog);

figure;
showHog(modelHog, [160, 96]);
title('Learned SVM Model (HOG Visualization)');