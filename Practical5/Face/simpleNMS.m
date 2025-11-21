function Objects = simpleNMS(Objects,threshold)

% Objects = [x, y, width, height, confidence]
% ObjectCorners = [x_min, y_min, x_max, y_max]
% IntersectionPoints = [xi_min, yi_min, xi_max, yi_max]
% IntersectionBounds = [i_width, i_height]
% IntersectionArea = [i_area, confidence]

ObjectCorners = Objects(:, 1:4);
ObjectCorners(:, 3) = Objects(:, 1) + Objects(:, 3); % x + width
ObjectCorners(:, 4) = Objects(:, 2) + Objects(:, 4); % y + height

for i=1:size(Objects, 1) 

    % Declare variables for thisObject
    thisObject = Objects(i, :);

    width = thisObject(3);
    height = thisObject(4);
    confidence = thisObject(5);

    x_min = thisObject(1);
    y_min = thisObject(2);
    x_max = x_min + width;
    y_max = y_min + height;
    
    % Get all the potential points that intersect with thisObject
    IntersectionPoints = [ ...
        max(x_min, ObjectCorners(:, 1)), ... % xi_min
        max(y_min, ObjectCorners(:, 2)), ... % yi_min
        min(x_max, ObjectCorners(:, 3)), ... % xi_max
        min(y_max, ObjectCorners(:, 4)) ...  % yi_max
    ];
    
    % Get bounds potential intersection points (if negative replace with 0)
    IntersectionBounds = [ ...
        max(0, IntersectionPoints(:, 3) - IntersectionPoints(:, 1)), ... % i_width
        max(0, IntersectionPoints(:, 4) - IntersectionPoints(:, 2)) ... % i_height
    ];
    
    % Calculate intersection area (Will be 0 if 1 bound is 0)
    IntersectionArea = [...
        (IntersectionBounds(:, 1) .* IntersectionBounds(:, 2)), ... % i_area
        Objects(:, 5) ... % confidence
    ];

    % Get indices of intersecting objects
    % Threshold is currently 0.3
    % (Threshold is declared on line 115 of 'ObjectDetection')
    indices = find((IntersectionArea(:, 1) ./ (width * height)) > threshold);
    
    % 
    for k = 1:length(indices)
        idx = indices(k);
        % Check if any Object that intersects thisObject has a greater
        % confidence
        if confidence < Objects(idx, 5)
            % Suppress thisObject if intersecting object has a
            % higher confidence
            Objects(i, :) = [0, 0, 0, 0, 0]; 
            ObjectCorners(i, :) = [0, 0, 0, 0];
        end
    end
end

end
