function [TP, FP, FN, matches] = evaluateDetections(predBoxes, gtBoxes, iouThresh)
% predBoxes: [N × 4] detected boxes  (x, y, w, h)
% gtBoxes:   [M × 4] ground truth boxes (x, y, w, h)
% iouThresh: minimum IoU to count as correct (e.g., 0.5)

if isempty(predBoxes)
    TP = 0;
    FP = 0;
    FN = size(gtBoxes,1);
    matches = [];
    return;
end

if isempty(gtBoxes)
    TP = 0;
    FP = size(predBoxes,1);
    FN = 0;
    matches = [];
    return;
end

numPred = size(predBoxes,1);
numGT   = size(gtBoxes,1);

IoUmat = zeros(numPred, numGT);

% compute IoU matrix
for i = 1:numPred
    for j = 1:numGT
        IoUmat(i,j) = computeIoU(predBoxes(i,:), gtBoxes(j,:));
    end
end

matches = zeros(0,2);   % [predIndex, gtIndex]
usedGT = false(numGT,1);

TP = 0;

% greedy match highest IoU predictions first
for i = 1:numPred
    [bestIoU, gtIndex] = max(IoUmat(i,:));

    if bestIoU >= iouThresh && ~usedGT(gtIndex)
        TP = TP + 1;
        usedGT(gtIndex) = true;
        matches(end+1,:) = [i, gtIndex];
    end
end

FP = numPred - TP;
FN = numGT - TP;

end


%% Helper function
function IoU = computeIoU(boxA, boxB)
% [x, y, w, h]

ax1 = boxA(1);
ay1 = boxA(2);
ax2 = ax1 + boxA(3);
ay2 = ay1 + boxA(4);

bx1 = boxB(1);
by1 = boxB(2);
bx2 = bx1 + boxB(3);
by2 = by1 + boxB(4);

% intersection
ix1 = max(ax1, bx1);
iy1 = max(ay1, by1);
ix2 = min(ax2, bx2);
iy2 = min(ay2, by2);

iw = max(0, ix2 - ix1);
ih = max(0, iy2 - iy1);
interArea = iw * ih;

unionArea = boxA(3)*boxA(4) + boxB(3)*boxB(4) - interArea;

IoU = interArea / unionArea;
end
