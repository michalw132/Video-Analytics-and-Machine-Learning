function [prediction maxi]= SVMTesting_v2(image,model)

if strcmp(model.type,'binary')
   
    pred = predict(model.modelFITSVM, image);
    if pred>0
        prediction = 1;
    else
        prediction = 0;
    end
    
else
    
   [pred,NegLoss,Pb] = predict(model.classifier,image);
    
    maxi=max(Pb);

     prediction = round(pred)-1;
    
end
    
end