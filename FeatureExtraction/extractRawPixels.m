function imgVector = extractRawPixels(img)

    % Ensure image is RGB
    if size(img,3)>1
        img=rgb2gray(img);
   end
    
    % Squashes the pixels into (1 x no. Of pixels) instead of (width x
    % height)
    vector = reshape(img,1, size(img, 1) * size(img, 2));

    % Changes it from 255 to 0 - 1
    imgVector = double(vector) / 255; 
end
