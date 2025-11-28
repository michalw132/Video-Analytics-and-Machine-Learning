function finalBoxes = nonMaxSuppression(boxes, overlapThresh)
% boxes: [x, y, width, height]

    if isempty(boxes)
        finalBoxes = boxes;
        return;
    end

    % Top Left Coords
    x1 = boxes(:,1);
    y1 = boxes(:,2);

    % Bottom Right Coords
    x2 = boxes(:,1) + boxes(:,3);
    y2 = boxes(:,2) + boxes(:,4);

    area = boxes(:,3) .* boxes(:,4);

    % Sort by bottom right y 
    [~, idxs] = sort(y2);    
    
    % Stores boxes kept after NMS
    pick = [];

    % While all boxes are considered empty or suppressed
    while ~isempty(idxs)

        % Takes the box with the largest bottom right y
        last = length(idxs);
        i = idxs(last);
        pick = [pick; i];

        suppress = last;

        % Compares every other box with this box
        for pos = 1:last-1
            j = idxs(pos);

            % Top Left Corner of intersection
            xx1 = max(x1(i), x1(j));
            yy1 = max(y1(i), y1(j));

            % Bottom Right Corner of intersection
            xx2 = min(x2(i), x2(j));
            yy2 = min(y2(i), y2(j));
            
            % Width and height of intersection
            w = max(0, xx2 - xx1);
            h = max(0, yy2 - yy1);

            overlap = (w * h) / area(j);

            if overlap > overlapThresh
                % Addes this box to be supressed
                suppress(end+1) = pos;
            end
        end
        
        % Removes all boxes that were picked or suppressed
        idxs(suppress) = [];
    end
    
    % Only 
    finalBoxes = boxes(pick, :);
end
