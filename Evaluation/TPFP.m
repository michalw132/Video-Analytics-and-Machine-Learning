function [result] = TPFP(testImages, testLabels, classificationResult)

N = size(testImages, 1);
TP = sum((testLabels == 1) & (classificationResult == 1));
TN = sum((testLabels == 0) & (classificationResult == 0));
FP = sum((testLabels == 0) & (classificationResult == 1));
FN = sum((testLabels == 1) & (classificationResult == 0));


result.accuracy = (TN+TP) / N;
result.errorRate = 1 - result.accuracy;
result.precision = TP / (TP+FP);
result.specificity = TN / (TN+FP);
result.sensitivity = TP / (TP+FN);
result.fMeasure = 2 * TP / (2 * TP + FN + FP);
result.falseAlarmRate = 1 - result.specificity;

result.TP = TP;
result.TN = TN;
result.FP = FP;
result.FN = FN;

end