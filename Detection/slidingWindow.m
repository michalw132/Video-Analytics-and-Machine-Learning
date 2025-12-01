clear all;
close all;

% Load classifier
load SVMModel modelSVM;

% Starts the Video Recorder
outputVideo = VideoWriter('pedestrian_detection.mp4','MPEG-4');
outputVideo.FrameRate = .5;  % frames per second
open(outputVideo);

dataset_sampling_rate = 1;

% Load dataset
pedestrianData = loadDataset('test.dataset', dataset_sampling_rate);

% Variables 
svm_threshold = .9;
kernel = "gaussian";
nms_rate = .1;
gt_detection_threshold = .2;
base_step = 15;
scales = [1.8, 1.5, 1.2, 1];
no_of_Images = 5;

% Window size (height × width)
winH = 160;
winW = 96;

% Performance counters
totalTP = 0;
totalFP = 0;
totalFN = 0;

% Loop through test images
for i = 1:no_of_Images

    fprintf("Processing image %d / %d\n", i, no_of_Images);

    % Read image and convert to grayscale
    I = imread(pedestrianData(i).filename);
    I = rgb2gray(I);

    % Initialize detection storage
    detections = [];

    % Loop over each scale
    for s = 1:length(scales)

        % Rescale the image according to the current scale
        scale = scales(s);
        Iresized = imresize(I, scale);

        % Sets the no. of rows and columns in scaled image
        rMax = size(Iresized,1) - winH + 1;
        cMax = size(Iresized,2) - winW + 1;
        
        % So that the crop step is scaled
        stepScaled = max(1, round(base_step * scale));

        % Sliding window over resized image
        for r = 1:stepScaled:rMax
            for c = 1:stepScaled:cMax

                % Gets the current window
                window = Iresized(r:r+winH-1, c:c+winW-1);
                window = im2double(window);

                % Extract HOG features and reshape to row vector
                feat = extractHOGVector(window);
                feat = feat(:)';

                % Z-score standardization using training mean/std
                feat = (feat - modelSVM.mu) ./ modelSVM.sigma;

                % L2-normalize the feature
                feat = feat / norm(feat, 2);

                % Applies PCA 
                feat_centered = feat - modelSVM.PCAMean;
                feat_pca = feat_centered * modelSVM.PCAVectors;

                % Replace NaNs (just in case)
                if any(isnan(feat_pca))
                    feat_pca(isnan(feat_pca)) = 0;
                end

                % SVM prediction
                prediction = SVMTesting(feat_pca, modelSVM, kernel, svm_threshold);

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
    nmsBoxes = nonMaxSuppression(detections, nms_rate);

    gtBoxes = pedestrianData(i).boxes;

    % Create a  figure  to draw boxes
    f = figure(i); 
    imshow(I); 
    hold on;

    % Draw final detections
    for k = 1:size(nmsBoxes,1)
        rectangle('Position', nmsBoxes(k,:), 'EdgeColor', 'g', 'LineWidth', 2);
    end
    
    % Draws Ground Truth
    for k = 1:size(gtBoxes,1)
        rectangle('Position', gtBoxes(k,:), 'EdgeColor', 'r', 'LineWidth', 2);
    end

    % Capture the frame for video
    frame = getframe(f);   % Capture current figure
    writeVideo(outputVideo,frame);

    % Evaluate detections against ground truth
    [TP, FP, FN, matches] = evaluateDetections(nmsBoxes, gtBoxes, gt_detection_threshold);
    fprintf("Image %d Evaluation: TP=%d  FP=%d  FN=%d\n", i, TP, FP, FN);

    % Accumulate totals
    totalTP = totalTP + TP;
    totalFP = totalFP + FP;
    totalFN = totalFN + FN;
end

close(outputVideo);

% --- Final Performance Summary Graph ---
figure;
bar([totalTP, totalFP, totalFN]);
set(gca, 'XTickLabel', {'TP','FP','FN'}, 'FontSize', 14);
ylabel('Count');
title('Overall Detector Performance');
grid on;

% --- Compute Precision, Recall, and F1 ---
precision = totalTP / (totalTP + totalFP);
recall    = totalTP / (totalTP + totalFN);
F1        = 2 * (precision * recall) / (precision + recall);

fprintf("\nFinal Metrics:\n");
fprintf("Precision = %.3f\n", precision);
fprintf("Recall    = %.3f\n", recall);
fprintf("F1 Score  = %.3f\n", F1);
