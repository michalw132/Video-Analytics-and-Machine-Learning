function pedestrianData = loadDataset(filename, sampling)

if nargin<2
    sampling = 1;
end

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

% This just gets the line 'ImageDataset'
line1=fgetl(fp);

% The second line lists the number of images
numberOfImages = fscanf(fp,'%d',1);

pedestrianData = struct();
imageCounter = 0;

for i=1:sampling:numberOfImages

    % Gets the first string in the next line (name of the file)
    imfile = fscanf(fp,'%s',1);

    % Gets the first int in the next line (no. of pedestrians)
    numPedestrians = fscanf(fp,'%d',1);

    boxes = zeros(numPedestrians, 4);  % [x, y, w, h] for each pedestrian

    % For every pedestrian in this image
    for k = 1:numPedestrians

        % Reads in all the numbers
        x_center = fscanf(fp,'%f',1);
        y_center = fscanf(fp,'%f',1);
        w = fscanf(fp,'%f',1);
        h = fscanf(fp,'%f',1);

        %This isn't needed (equates to all of the zeroes in the file)
        visibility = fscanf(fp,'%d',1);
        
        % Top left corner
        x1 = round(x_center - w/2);
        y1 = round(y_center - h/2);
        
        % Ensures that all values are at least 1
        x1 = max(1,x1); y1 = max(1,y1);
        
        % Bottom right corner (ensuring the max is the edge of the image)
        x2 = min(640, x1 + round(w) - 1);
        y2 = min(480, y1 + round(h) - 1);

        boxes(k,:) = [x1, y1, x2 - x1 + 1, y2 - y1 + 1];  % [x, y, width, height]
    end

    % Store in structure
    imageCounter = imageCounter + 1;
    pedestrianData(imageCounter).filename = imfile;
    pedestrianData(imageCounter).boxes = boxes;

end

fclose(fp);

end