function prediction = NNTesting(testImage, modelNN)

    for i=1:size(modelNN.neighbours,1)
        distances(i) = EuclideanDistance(testImage, modelNN.neighbours(i,:));
    end
   
    [~, index] = min(distances);
    
    prediction = modelNN.labels(index);
end