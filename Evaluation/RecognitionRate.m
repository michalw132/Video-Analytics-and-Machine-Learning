function [Accuracy] = RecognitionRate(testLabels,classificationResult)
comparison = (testLabels == classificationResult);
Accuracy = sum(comparison) / length(comparison);
end