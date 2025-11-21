function PixelSum=GetSumRect(IntegralImage,x,y,Width,Height)
%
% PixelSum=GetSumRect(IntegralImage,x,y,Width,Height)
%
 IIWidth=size(IntegralImage,1);
 
D = IntegralImage((x+Width)*IIWidth + y + Height+1);

A = IntegralImage(x*IIWidth+y+1);

B = IntegralImage((x+Width)*IIWidth+y+1);

C = IntegralImage(x*IIWidth+y+Height+1);
          
% Calculate the sum of pixels within the rectangular mask as a combination
% of the elements A, B, C and D on the integral image
%% step 5
PixelSum = D - B - C + A;
end



            
       