function [prediction maxi]= SVMTesting(image,model,kernel)

if strcmp(model.type,'binary')
    
    kerneloption.matrix=svmkernel(image,kernel,model.param.sigmakernel,model.xsup);
    % svmkernel kernel options:
    % gaussian, poly, polyhomog, htrbf, wavelet, frame
    % poly and polyhomog are linear kernels

    pred = svmval(image,model.xsup,model.w,model.w0,model.param.kernel,kerneloption);

    %Change this threshold
    if pred>.8
        prediction = 1;
    else
        prediction = 0;
    end
    
else
    
    [pred maxi] = svmmultival(image,model.xsup,model.w,model.b,model.nbsv,model.param.kernel,model.param.kerneloption);

     prediction = round(pred)-1;
    
end
    
end