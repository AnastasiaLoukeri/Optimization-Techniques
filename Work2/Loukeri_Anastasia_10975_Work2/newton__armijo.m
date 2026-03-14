clear;
clc; close all;

syms x y

f = x^3 * exp(-x^2 - y^4);

tol = 0.0001;

% --- Case 1: Start at (1, -1) ---
fprintf('Processing Newton (Armijo) for (1, -1)...\n');
[~, k1, x1, y1, g1, msg1] = newton_armijo(f, [-1, -1], tol, x, y);
visualize_results(k1, x1, y1, f, g1, msg1);

% --- Case 2: Start at (-1, 1) ---
fprintf('Processing Newton (Armijo) for (-1, 1)...\n');
[~, k2, x2, y2, g2, msg2] = newton_armijo(f, [1, 1], tol, x, y);
visualize_results(k2, x2, y2, f, g2, msg2);

% --- Case 3: Start at (0, 0) ---
fprintf('Processing Newton (Armijo) for (0, 0)...\n');
[~, k3, x3, y3, g3, msg3] = newton_armijo(f, [0, 0], tol, x, y);
visualize_results(k3, x3, y3, f, g3, msg3);




function visualize_results(k, x_hist, y_hist, f_sym, g_array, msg)

    f_num = matlabFunction(f_sym);
    [x_vals, y_vals] = meshgrid(-2.5:0.05:2.5, -2.5:0.05:2.5);
    z_vals = f_num(x_vals, y_vals);

    x0_str = num2str(x_hist(1), '%.2f');
    y0_str = num2str(y_hist(1), '%.2f');

    % 
    figure;
    contourf(x_vals, y_vals, z_vals, 30); 
    colorbar;
    hold on;
    
    plot(x_hist, y_hist, '-o', 'Color', 'k', 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'r', 'MarkerSize', 4);
    plot(x_hist(1), y_hist(1), 's', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    % Marker logic
    if contains(msg, 'Stopped') || contains(msg, 'NaN')
        plot(x_hist(end), y_hist(end), 'x', 'MarkerSize', 15, 'LineWidth', 2, 'Color', 'r');
    else
        plot(x_hist(end), y_hist(end), 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end

    xlabel('x'); ylabel('y');
    title({sprintf('Newton with Armijo [%s, %s]', x0_str, y0_str), ['Result: ' msg]});
    grid on; axis equal;
    xlim([-2.5 2.5]); ylim([-2.5 2.5]);
    

    % Convergence
    if ~isempty(x_hist)
        f_array = zeros(1, length(x_hist));
        for i = 1:length(x_hist)
            f_array(i) = f_num(x_hist(i), y_hist(i));
        end
        figure;
        plot(1:k, f_array, 'LineWidth', 2);
        xlabel('Iteration (k)'); ylabel('f(x_k)');
        title('Convergence History');
        grid on;
        
    end

    % Step Size 
    if ~isempty(g_array)
        figure;
        plot(1:length(g_array), g_array, '-o');
        xlabel('Iteration (k)'); ylabel('Step Size \gamma');
        title('Step Size Evolution (Armijo)');
        grid on;
        
    end
end


% NEWTON WITH ARMIJO

function [x_curr, k, x_hist, y_hist, g_hist, msg] = newton_armijo(f_sym, x_0, tol, sym_x, sym_y)
    
    % Pre-calculate handles
    f_func = matlabFunction(f_sym);
    grad_sym = gradient(f_sym, [sym_x, sym_y]);
    hess_sym = hessian(f_sym, [sym_x, sym_y]);
    
    grad_func = matlabFunction(grad_sym);
    hess_func = matlabFunction(hess_sym);

    k = 1;
    x_curr = x_0;
    
    x_hist = [x_curr(1)];
    y_hist = [x_curr(2)];
    g_hist = [];
    msg = 'Converged';
    
    max_iter = 200;

    % Armijo Params
    sigma = 0.0001; % alpha
    beta = 0.2;     % reduction factor
    gamma_init = 2; % Start testing with step size 2

    while k < max_iter
        % 1. Evaluate Gradient
        g_val_temp = grad_func(x_curr(1), x_curr(2));
        if numel(g_val_temp) == 1, grad_val = [g_val_temp; 0]; else, grad_val = g_val_temp; end
        
        if norm(grad_val) < tol
            break;
        end
        
        % 2. Evaluate Hessian
        H_val = hess_func(x_curr(1), x_curr(2));
        if numel(H_val) == 1, H_val = [H_val 0; 0 0]; end

        % 3. Check Positive Definiteness
        eigenvalues = eig(H_val);
        isPositiveDefinite = all(eigenvalues > 1e-6); 
        
        if isPositiveDefinite
            dk = -H_val \ grad_val;
        else 
            msg = 'Stopped (Hessian not Positive Definite)';
            disp(['Stopping at k=' num2str(k) ': ' msg]);
            break;
        end

        % 4. ARMIJO LINE SEARCH
        gamma = armijo_search(f_func, grad_val, dk, x_curr, gamma_init, sigma, beta);

        g_hist = [g_hist gamma];
        
        % 5. Update
        x_curr(1) = x_curr(1) + gamma * dk(1);
        x_curr(2) = x_curr(2) + gamma * dk(2);
        
        x_hist = [x_hist x_curr(1)];
        y_hist = [y_hist x_curr(2)];
        
        k = k + 1;
    end
end

% 
% ARMIJO SEARCH FUNCTION

function gamma = armijo_search(f_func, grad_val, dk, x_k, g_0, sigma, beta)
    
    m = 0;
    fx = f_func(x_k(1), x_k(2));
    
    % Pre-calculate descent product: (grad' * d)
    descent_product = grad_val.' * dk;
    
    % Limit iterations to prevent infinite loops
    max_armijo_iter = 20; 
    
    while m < max_armijo_iter
        gamma_test = g_0 * (beta^m);
        
        x_next = x_k + gamma_test * dk.';
        f_next = f_func(x_next(1), x_next(2));
        
        % Armijo Condition
        if f_next <= fx + sigma * gamma_test * descent_product
            gamma = gamma_test;
            return;
        end
        
        m = m + 1;
    end
    
    % If loop finishes without success, return the smallest tested gamma
    gamma = g_0 * (beta^max_armijo_iter);
end