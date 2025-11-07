function [trainImages, trainLabels, testImages, testLabels] = split5050(allImages, labels)

    % Check that the images and labels have the same amount of data
    if size(allImages, 1) ~= size(labels, 1)
        error('Number of images and labels must match.');
    end
    
    
    totNumSamples = size(labels, 1);

    % Gets the midpoint to split
    splitPoint = floor(totNumSamples / 2);

    % radomises the order of the images so that not all of the positive
    % images go into the training set
    randIndices = randperm(totNumSamples);

    % Reorder both images and labels in the same way
    allImages = allImages(randIndices, :);
    labels = labels(randIndices);

    % Divides training and testing images
    trainImages = allImages(1:splitPoint, :);
    trainLabels = labels(1:splitPoint);
    
    testImages = allImages(splitPoint+1:end, :);
    testLabels = labels(splitPoint+1:end);
end
