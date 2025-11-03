function prediction = KNNTesting(testImage, modelNN, K)

for i=1:size(modelNN.neighbours,1)
        distances(i) = EuclideanDistance(testImage, modelNN.neighbours(i,:));
end

    [~, sortedIdx] = sort(distances, 'ascend');
    
    nearestLabels = modelNN.labels(sortedIdx(1:K));
    
    prediction = mode(nearestLabels);

end