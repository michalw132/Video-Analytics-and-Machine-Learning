function outIm =preprocessDigit(inputIm)

%any pixel brighter than .5 is turned into white, any less than .5 turned
%into black
binary= inputIm>0.5;

%finds rows and columns of all white pixels
[r c] =find(binary);

%averages all coodinates of white pixels - gets the center of the digit
middlex=round(mean(c));
middley=round(mean(r));

%creates a black image of the same size
outIm =zeros(size(inputIm));

%calculates how far the digit's center is from the image center.
diffx=size(inputIm,2)/2-middlex;
diffy=size(inputIm,1)/2-middley;

%moves the image right or left compared to where the center is. Crops one 
% side and Adds black columns on the other side to fill the image.
if diffx>0

    outIm(:,1:28)=[zeros(size(inputIm,1),abs(diffx)) inputIm(:,1:end-abs(diffx))];
else
    outIm(:,1:28)=[inputIm(:,abs(diffx)+1:end) zeros(size(inputIm,1),abs(diffx)) ];
end

%moves the image up or down compared to where the center is. Crops one side 
% and Adds black columns on the other side to fill the image.
if diffy>0
    outIm(1:28,:)=[zeros(abs(diffy),size(outIm,2));  outIm(1:end-abs(diffy),:)];
else
    outIm(1:28,:)=[outIm(abs(diffy)+1:end,:); zeros(abs(diffy),size(outIm,2)) ];
end

end