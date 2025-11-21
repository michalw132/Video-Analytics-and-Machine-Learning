function Iout = IntegralImageCalculation(Iin)
    %% step 3
    % Calculate the integral image
    I = double(Iin);
    Iout = cumsum(cumsum(I, 1), 2);
end