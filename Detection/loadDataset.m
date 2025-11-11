function pedestrianData = loadDataset(filename, sampling)

if nargin<2
    sampling =1;
end


fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

line1=fgetl(fp);

numberOfImages = fscanf(fp,'%d',1);

pedestrianData = struct();
imageCounter = 0;



for im=1:sampling:numberOfImages
    
    imfile = fscanf(fp,'%s',1);
    numPedestrians = fscanf(fp,'%d',1);   
    
boxes = zeros(numPedestrians, 4);  % [x, y, w, h] for each pedestrian

% Read annotations for this image
    for k = 1:numPedestrians
        x_center = fscanf(fp,'%f',1);
        y_center = fscanf(fp,'%f',1);
        w = fscanf(fp,'%f',1);
        h = fscanf(fp,'%f',1);

        %This isn't needed
        visibility = fscanf(fp,'%d',1);

        x1 = round(x_center - w/2);
        y1 = round(y_center - h/2);
        x1 = max(1,x1); y1 = max(1,y1);
        x2 = min(96, x1 + round(w) - 1);
        y2 = min(160, y1 + round(h) - 1);

        boxes(k,:) = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];  % [x, y, width, height]
    end
    
    % Store in structure
    imageCounter = imageCounter + 1;
    pedestrianData(imageCounter).filename = imfile;
    pedestrianData(imageCounter).boxes = boxes;
    
end

fclose(fp);

end