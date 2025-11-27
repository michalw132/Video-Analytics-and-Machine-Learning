function cvFolds = splitCrossValidation(allImages, labels, k)
% stratifiedKFoldCV - Perform stratified k-fold cross-validation
%
% INPUTS:
%   allImages : [N x D] matrix of feature vectors
%   labels    : [N x 1] vector of labels (0 or 1)
%   k         : number of folds (e.g., 5 or 10)
%
% OUTPUT:
%   cvFolds : struct array with fields:
%       trainImages, trainLabels, testImages, testLabels

    % Separate positive and negative indices
    posIdx = find(labels == 1);
    negIdx = find(labels == 0);
    
    % Shuffle the indices
    posIdx = posIdx(randperm(length(posIdx)));
    negIdx = negIdx(randperm(length(negIdx)));
    
    % Split indices into k roughly equal folds
    posFolds = splitIndices(posIdx, k);
    negFolds = splitIndices(negIdx, k);
    
    cvFolds = struct();
    
    for fold = 1:k
        % Test indices for this fold
        testIdx = [posFolds{fold}; negFolds{fold}];
        
        % Train indices are all other indices
        trainIdx = setdiff(1:size(allImages,1), testIdx);
        
        % Assign to struct
        cvFolds(fold).trainImages = allImages(trainIdx,:);
        cvFolds(fold).trainLabels = labels(trainIdx);
        cvFolds(fold).testImages = allImages(testIdx,:);
        cvFolds(fold).testLabels = labels(testIdx);
    end
end

function folds = splitIndices(indices, k)
% Helper function to split indices into k folds
    n = length(indices);
    foldSize = floor(n / k);
    folds = cell(1,k);
    
    for i = 1:k
        startIdx = (i-1)*foldSize + 1;
        if i < k
            endIdx = i*foldSize;
        else
            endIdx = n; % last fold takes remainder
        end
        folds{i} = indices(startIdx:endIdx);
    end
end
