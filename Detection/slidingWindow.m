clear all;
close all;

% Load classifier
load SVMModel modelSVM;

% Load dataset
pedestrianData = loadDataset('test.dataset', 20);

% Window size (height × width)
winH = 160;
winW = 96;

% Sliding window step size
step = 15;

% Scales to detect closer/further pedestrians
scales = [1, .7, 0.5];

% Storage for detections for each image
totalDetections = cell(1, numel(pedestrianData));

% Loop through test images
for i = 1:2

    if i == 2
        break;
    end
    fprintf("Processing image %d / %d\n", i, numel(pedestrianData));

    % Read image and convert to grayscale
    I = imread(pedestrianData(i).filename);
    I = rgb2gray(I);
    imshow(I); hold on;

    % Initialize detection storage
    detections = [];

    % Loop over each scale
    for s = 1:length(scales)
        scale = scales(s);
        Iresized = imresize(I, scale);
        rMax = size(Iresized,1) - winH + 1;
        cMax = size(Iresized,2) - winW + 1;

        % Sliding window over resized image
        for r = 1:step:rMax
            for c = 1:step:cMax
                patch = Iresized(r:r+winH-1, c:c+winW-1);
                patch = im2double(patch);
                patch = imresize(patch, [winH winW]);  % ensure same size as training

                % Extract HOG features and reshape to row vector
                feat = extractHOGVector(patch);
                feat = feat(:)';

                % Z-score standardization using training mean/std
                feat = (feat - modelSVM.mu) ./ modelSVM.sigma;

                % L2-normalize the feature
                feat = feat / norm(feat, 2);

                % Replace NaNs (just in case)
                if any(isnan(feat))
                    feat(isnan(feat)) = 0;
                end

                % SVM prediction
                prediction = SVMTesting(feat, modelSVM, "gaussian");

                % If positive, map box back to original image coordinates
                if prediction == 1
                    orig_r = round(r / scale);
                    orig_c = round(c / scale);
                    orig_w = round(winW / scale);
                    orig_h = round(winH / scale);
                    detections = [detections; orig_c, orig_r, orig_w, orig_h];
                end
            end
        end
    end

    % Apply Non-Maximum Suppression to remove overlapping boxes
    nmsBoxes = nonMaxSuppression(detections, 0.3);

    % Draw final detections
    for k = 1:size(nmsBoxes,1)
        rectangle('Position', nmsBoxes(k,:), 'EdgeColor', 'g', 'LineWidth', 2);
    end

    % Evaluate detections against ground truth
    gtBoxes = pedestrianData(i).boxes;
    [TP, FP, FN, matches] = evaluateDetections(nmsBoxes, gtBoxes, 0.5);
    fprintf("Image %d Evaluation: TP=%d  FP=%d  FN=%d\n", i, TP, FP, FN);

    totalDetections{i} = nmsBoxes;
end

% % Display the first 25 crops
% figure
% sgtitle('First 25 crops')
% for i=1:25
%     subplot(5,5,i)
%     Im = reshape(pedestrianCrops(i,:),160,96);
%     imshow(Im, [])
%
% end
%
% % Display the next 25 crops
% figure
% sgtitle('Next 25 crops')
% counter = 1;
% for i=25:49
%
%     subplot(5,5,counter)
%     Im = reshape(pedestrianCrops(i,:),160,96);
%     imshow(Im, [])
%     counter = counter + 1;
% end
%
% % Display the next 25 crops
% figure
% sgtitle('Next 25 crops')
% counter = 1;
% for i=50:74
%     subplot(5,5,counter)
%     Im = reshape(pedestrianCrops(i,:),160,96);
%     imshow(Im, [])
%      counter = counter + 1;
%
% end
%
% % Display the next 25 crops
% figure
% sgtitle('Next 25 crops')
% counter = 1;
% for i=75:99
%     subplot(5,5,counter)
%     Im = reshape(pedestrianCrops(i,:),160,96);
%     imshow(Im, [])
%      counter = counter + 1;
%
% end
%
%
% %% Evaluation


