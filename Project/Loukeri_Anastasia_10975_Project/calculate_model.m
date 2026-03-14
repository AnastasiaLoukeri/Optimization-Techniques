function [mse, out] = calculate_model(p, u, y, n_g, w_fixed)
    G = zeros(size(u,1), n_g);
    for j = 1:n_g
        idx = (j-1)*4 + 1;
        c1 = p(idx); c2 = p(idx+1);
        s1 = p(idx+2); s2 = p(idx+3);
        % gaussian
        G(:,j) = exp(-( (u(:,1)-c1).^2/(2*s1^2) + (u(:,2)-c2).^2/(2*s2^2) ));
    end
    
    if nargin < 5
        % best weight with least squares
        w = G \ y; 
        mse = mean((y - G*w).^2);
        out = w;
    else
        
        y_pred = G * w_fixed;
        mse = mean((y - y_pred).^2);
        out = y_pred;
    end
end