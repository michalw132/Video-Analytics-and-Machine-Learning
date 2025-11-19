clear all
close all

%We load the classification model of our choice
load SVMModel modelSVM

pedestrianData = loadDataset('test.dataset', 20);

%Open testing image and convert to gray scale
I=imread('./Assets/pedestrian/image_00000011.jpg');
I=rgb2gray(I);


% Rough estimate of how large a pedestrian will be in an image
samplingX = 160;
samplingY = 96;

% we can calculate the number and rows our sampling area gives us
numberRows = round(size(I, 1) / samplingX);
numberColumns = round(size(I, 2) / samplingY);

%This is just for visualisation
pedestrianCrops = [];

%This is what houses each images detections
totalDetections = cell(1, numel(pedestrianData));

% This loops through all of the pedestrian images
for i=1:numel(pedestrianData)

    %Gets the current pedestrian image
    I=imread(pedestrianData(i).filename);
    I=rgb2gray(I);
    
    % Shows it and holds on so it can put a rectangle on it when it detects
    % something, if there's no break point in the detection then this is
    % meaningless
    imshow(I)
    hold on
    
    %for each pedestrian within the image, 
    county = 0;

    detections = [];
    for r = 1:30:size(I,1)      % slide vertically every 5 pixels
        
        %Initialises detections array
        currentDetections = [];
        county = county + 1;
        countx = 0;
        for c = 1:30:size(I,2)  % slide horizontally every 5 pixels    
            
            % if the current box is not outside of the image
            if (c+samplingY-1 <= size(I,2)) && (r+samplingX-1 <= size(I,1))
                
                countx = countx + 1;

                %we crop the area
                pedestrianIm = I(r:r+samplingX-1, c:c+samplingY-1);
                
                % we convert it into doubles from 0 to 1 
                pedestrianIm = im2double(pedestrianIm);
                
                %All training examples were 160x96. To have any chance, we need to
                %resample them into a 160x96 image
                pedestrianIm = imresize(pedestrianIm, [160 96]);

                pedestrianCrops = [pedestrianCrops; pedestrianIm];

                pedestrianIm = extractHOGVector(pedestrianIm); 
                
                tic
                prediction =  SVMTesting(pedestrianIm, modelSVM, "gaussian");
                toc
                
                    if prediction == 1
                        rectangle('Position', [c, r, samplingY, samplingX], ...
                                  'EdgeColor', 'g', 'LineWidth', 2);
                        % store detection box for later evaluation
                        detections = [detections; c, r, samplingY, samplingX];
                    end
         
            end
        end    
    end

    totalDetections{i} = detections;

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


