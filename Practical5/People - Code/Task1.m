
close all;


[images, labels] =  loadPedestrianDatabase('pedestrian_train.cdataset', 1);

path = ('images/neg');

im = imread('images/pos/img00000.jpg');

hog = hog_feature_vector (im);
size(im);

figure
subplot(1,2,1)
imshow(im)
subplot(1,2,2)
showHog(hog, [160,96]);

totalHogs = [];

for i=1:size(images,1)
    im = reshape(images(i,:), 160, 96);
    thisHog = hog_feature_vector (im);
    totalHogs = [totalHogs; thisHog];
end

tic
SVMModel = SVMtraining(totalHogs, labels);
toc

[testImages, testLabels] =  loadPedestrianDatabase('pedestrian_test.cdataset', 200);

totalHogs = [];

for i=1:size(testImages,1)
    im = reshape(testImages(i,:), 160, 96);
    thisHog = hog_feature_vector (im);
    totalHogs = [totalHogs; thisHog];
end

tic
for i=1:size(testImages,1)
    
    classificationResult(i,1) = SVMTesting(totalHogs,SVMModel);

end
toc


% Finally we compared the predicted classification from our mahcine
% learning algorithm against the real labelling of the esting image
comparison = (testLabels==classificationResult);

%Accuracy is the most common metric. It is defiend as the numebr of
%correctly classified samples/ the total number of tested samples
Accuracy = sum(comparison)/length(comparison)

% Compute the weighted average HOG pattern
modelHog = sum((SVMModel.alpha .* labels(SVMModel.pos)) .* SVMModel.xsup, 1);
modelHog = modelHog / norm(modelHog);

figure;
showHog(modelHog, [160, 96]);
title('Learned SVM Model (HOG Visualization)');