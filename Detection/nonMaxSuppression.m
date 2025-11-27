function finalBoxes = nonMaxSuppression(boxes, overlapThresh)
% boxes: [x, y, width, height]

    if isempty(boxes)
        finalBoxes = boxes;
        return;
    end

    % Convert boxes into [x1 y1 x2 y2]
    x1 = boxes(:,1);
    y1 = boxes(:,2);
    x2 = boxes(:,1) + boxes(:,3);
    y2 = boxes(:,2) + boxes(:,4);

    area = boxes(:,3) .* boxes(:,4);
    [~, idxs] = sort(y2);    % sort by bottom-right y 

    pick = [];

    while ~isempty(idxs)
        last = length(idxs);
        i = idxs(last);
        pick = [pick; i];

        suppress = last;
        for pos = 1:last-1
            j = idxs(pos);

            % Compute intersection
            xx1 = max(x1(i), x1(j));
            yy1 = max(y1(i), y1(j));
            xx2 = min(x2(i), x2(j));
            yy2 = min(y2(i), y2(j));

            w = max(0, xx2 - xx1);
            h = max(0, yy2 - yy1);

            overlap = (w * h) / area(j);

            if overlap > overlapThresh
                suppress(end+1) = pos;
            end
        end

        idxs(suppress) = [];
    end

    finalBoxes = boxes(pick, :);
end
