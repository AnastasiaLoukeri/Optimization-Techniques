clear;
clc; close all;

syms x y
f = x^3 * exp(-x^2 - y^4);

tol = 0.0001;

% --- Case 1: Start at (1, -1) ---
fprintf('Processing Armijo for (-1, -1)...\n');
[~, k1, x1, y1, g1] = steepest_armijo(f, [-1, -1], tol, x, y);
visualize_results(k1, x1, y1, f, g1);

% --- Case 2: Start at (-1, 1) ---
fprintf('Processing Armijo for (1, 1)...\n');
[~, k2, x2, y2, g2] = steepest_armijo(f, [1, 1], tol, x, y);
visualize_results(k2, x2, y2, f, g2);

% --- Case 3: Start at (0, 0) ---
fprintf('Processing Armijo for (0, 0)...\n');
[~, k4, x4, y4, g4] = steepest_armijo(f, [0, 0], tol, x, y);
visualize_results(k4, x4, y4, f, g4);



function visualize_results(k, x_hist, y_hist, f_sym, g_array)

    % Convert to numeric handle
    f_num = matlabFunction(f_sym);
    
    % Grid for Contour
    [x_vals, y_vals] = meshgrid(-2.5:0.05:2.5, -2.5:0.05:2.5);
    z_vals = f_num(x_vals, y_vals);

    x0_str = num2str(x_hist(1), '%.2f');
    y0_str = num2str(y_hist(1), '%.2f');

    % --- FIGURE 1: Contour & Path ---
    figure;
    set(gcf, 'Color', 'w');
    contourf(x_vals, y_vals, z_vals, 30); 
    colorbar;
    hold on;
    
    % Path
    plot(x_hist, y_hist, '-o', 'Color', 'k', 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'r', 'MarkerSize', 4);
    % Start (Green)
    plot(x_hist(1), y_hist(1), 's', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    % End (Yellow)
    plot(x_hist(end), y_hist(end), 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

    xlabel('x'); ylabel('y');
    title(sprintf('Steepest Descent (Armijo) from [%s, %s]', x0_str, y0_str));
    grid on; axis equal;
    xlim([-2.5 2.5]); ylim([-2.5 2.5]);
    

    %  Convergence 
    f_array = zeros(1, length(x_hist));
    for i = 1:length(x_hist)
        f_array(i) = f_num(x_hist(i), y_hist(i));
    end
    figure;
    set(gcf, 'Color', 'w');
    plot(1:k, f_array, 'LineWidth', 2);
    xlabel('Iteration (k)'); ylabel('f(x_k)');
    title('Convergence History');
    grid on;
    

    %  Step Size 
    if ~isempty(g_array)
        figure;
        set(gcf, 'Color', 'w');
        plot(1:length(g_array), g_array, '-');
        xlabel('Iteration (k)'); ylabel('Step Size \gamma');
        title('Step Size Evolution (Armijo)');
        grid on;
        
    end
end


% STEEPEST DESCENT WITH ARMIJO

function [x_curr, k, x_hist, y_hist, g_hist] = steepest_armijo(f_sym, x_0, tol, sym_x, sym_y)
    
    % 
    f_func = matlabFunction(f_sym);
    grad_sym = gradient(f_sym, [sym_x, sym_y]);
    grad_func = matlabFunction(grad_sym);

    k = 1;
    x_curr = x_0;
    
    x_hist = [x_curr(1)];
    y_hist = [x_curr(2)];
    g_hist = [];
    
    max_iter = 50;

    % Armijo Parameters
    alpha = 0.0001; % Acceptance parameter (sigma)
    beta = 0.2;     % Reduction factor (0.2 or 0.5)
    gamma_init = 1; % Initial large step attempt

    while k < max_iter
        % Evaluate gradient
        % Handle case if gradient function returns scalar
        g_val_temp = grad_func(x_curr(1), x_curr(2));
        if numel(g_val_temp) == 1
             grad_val = [g_val_temp; 0]; % Fallback dimensions
        else
             grad_val = g_val_temp;
        end
        
        if norm(grad_val) < tol
            break;
        end
        
        dk = -grad_val;

        % ARMIJO SEARCH
        % Find step size gamma
        gamma = armijo_rule(f_func, grad_val, dk, x_curr, gamma_init, alpha, beta);

        g_hist = [g_hist gamma];
        
        % Update
        x_curr(1) = x_curr(1) + gamma * dk(1);
        x_curr(2) = x_curr(2) + gamma * dk(2);
        
        x_hist = [x_hist x_curr(1)];
        y_hist = [y_hist x_curr(2)];
        
        k = k + 1;
    end
end


% ARMIJO RULE FUNCTION

function gamma = armijo_rule(f_func, grad_val, dk, x_k, g_0, alpha, beta)
    
    m = 0;
    % Current function value
    fx = f_func(x_k(1), x_k(2));
    % Directional derivative term: alpha * gamma * (grad' * d)
    % We pre-calculate (grad' * d)
    descent_product = grad_val.' * dk; 
    
    while true
        gamma_test = g_0 * (beta^m);
        
        % Next point candidate
        x_next = x_k + gamma_test * dk.'; 
        % Note: dk.' ensures matching dimensions if needed
        
        f_next = f_func(x_next(1), x_next(2));
        
        % Armijo Condition: f(x+gd) <= f(x) + alpha * gamma * grad'*d
        if f_next <= fx + alpha * gamma_test * descent_product
            gamma = gamma_test;
            return;
        end
        
        m = m + 1;
        
        % Safety break
        if m > 20
            gamma = gamma_test; 
            break; 
        end
    end
end