function [prediction maxi] = SVMDetection(image,model)

if strcmp(model.type,'binary')
   
    [pred,b, c] = predict(model.modelFITSVM, image);
    if pred>0
        prediction = 1;
    else
        prediction = 0;
    end
    maxi=max(c);
else
    
    [pred,NegLoss,Pb] = predict(model.classifier,image);
    
    maxi=max(Pb);

     prediction = round(pred)-1;
    
end
    
end