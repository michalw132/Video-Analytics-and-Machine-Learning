clear all;
close all;

haar_cascade = GetHaarCasade('haarcascade_frontalface_alt.mat');

% step 1 completed in function 'ShowHaarCasade' (line 50)
% step 2 completed in function 'ObjectDetection' (line 81) 
% step 3 completed in function 'IntegralImageCalculation'
% step 4 completed in function 'ObjectDetection'(line 90)
% step 5 completed in function 'GetSumRect' (line 17)

% step 6
ObjectDetection('1.jpg','haarcascade_frontalface_alt.mat');

% step 7 completed in function 'ObjectDetection' and (line 110)
%                  in function 'simpleNMS'

% step 8 completed in function 'ObjectDetection' and (line 118)
