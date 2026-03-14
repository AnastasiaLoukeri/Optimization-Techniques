clear;
clc; close all;

% Define symbolic variables
syms x y



f = x^3 * exp(-x^2 - y^4);

% Tolerance
tol = 0.0001;

% --- Case 1: Start at (1, -1) ---
fprintf('Processing Start Point (-1, -1)...\n');
[~, k1, x1, y1, g1] = steepest_descent_exact(f, [-1, -1], tol, x, y);
visualize_results(k1, x1, y1, f, g1);

% --- Case 2: Start at (-1, 1) ---
fprintf('Processing Start Point (1, 1)...\n');
[~, k2, x2, y2, g2] = steepest_descent_exact(f, [1, 1], tol, x, y);
visualize_results(k2, x2, y2, f, g2);

% --- Case 3: Start at (0, 0) ---
fprintf('Processing Start Point (0, 0)...\n');
[~, k4, x4, y4, g4] = steepest_descent_exact(f, [0, 0], tol, x, y);
visualize_results(k4, x4, y4, f, g4);


% --------------------------------------------------------
% VISUALIZATION FUNCTION
% --------------------------------------------------------
function visualize_results(k, x_hist, y_hist, f_sym, g_array)

    % Convert symbolic function to numeric
    f_num = matlabFunction(f_sym);
    
    % Create Grid (Adjusted range for the new function)
    [x_vals, y_vals] = meshgrid(-2.5:0.05:2.5, -2.5:0.05:2.5);
    z_vals = f_num(x_vals, y_vals);

    x0_str = num2str(x_hist(1), '%.2f');
    y0_str = num2str(y_hist(1), '%.2f');

    %FIG 1
    figure;
    set(gcf, 'Color', 'w');
    contourf(x_vals, y_vals, z_vals, 30); 
    colorbar;
    hold on;
    
    % Plot Path
    plot(x_hist, y_hist, '-', 'Color', 'k', 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'r', 'MarkerSize', 4);
     
    % Start Point (Green Square)
    plot(x_hist(1), y_hist(1), 's', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
     
    % End Point (Yellow Star)
    plot(x_hist(end), y_hist(end), 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

    xlabel('x'); ylabel('y');
    title(sprintf('Exact Steepest Descent Path from [%s, %s]', x0_str, y0_str));
    grid on; axis equal;
    xlim([-2.5 2.5]); ylim([-2.5 2.5]);
    
    

    % Convergence 
    f_array = zeros(1, length(x_hist));
    for i = 1:length(x_hist)
        f_array(i) = f_num(x_hist(i), y_hist(i));
    end
    
    figure;
    set(gcf, 'Color', 'w');
    plot(1:k, f_array, 'LineWidth', 2);
    legend('f(x_k) Value');
    xlabel('Iteration (k)');
    ylabel('Function Value');
    title('Convergence History');
    grid on;
    

    % Step Size 
    if ~isempty(g_array)
        figure;
        set(gcf, 'Color', 'w');
        plot(1:length(g_array), g_array, '-');
        xlabel('Iteration (k)');
        ylabel('Step Size \gamma');
        title('Step Size Evolution (Exact Line Search)');
        grid on;
        
    end
end


% STEEPEST DESCENT WITH EXACT LINE SEARCH

function [x_curr, k, x_hist, y_hist, g_hist] = steepest_descent_exact(f_sym, x_0, tol, sym_x, sym_y)
    
    % Calculate gradient symbolically once
    grad_sym = gradient(f_sym, [sym_x, sym_y]);
    
    k = 1;
    x_curr = x_0;
    
    % Initialize History
    x_hist = [x_curr(1)];
    y_hist = [x_curr(2)];
    g_hist = [];
    
    max_iter = 12;

    % Create a symbolic variable for the step size gamma
    syms g_sym_var 

    while true
        % 1. Evaluate Gradient at current point
        grad_val = double(subs(grad_sym, {sym_x, sym_y}, {x_curr(1), x_curr(2)}));
        
        % Check Convergence
        if norm(grad_val) < tol || k > max_iter
            break;
        end
        
        % 2. Descent Direction
        dk = -grad_val;

        % 3. EXACT LINE SEARCH (Using Derivative Bisection)
        % Define phi(g) = f(x_k + g*d_k)
        x_next_sym = x_curr(1) + g_sym_var * dk(1);
        y_next_sym = x_curr(2) + g_sym_var * dk(2);
        
        phi_sym = subs(f_sym, {sym_x, sym_y}, {x_next_sym, y_next_sym});
        
        % Find optimal gamma where phi'(g) = 0
        % Searching in range [0, 2] usually suffices for this function
        [~, a, b, ~, ~] = bisection_derivative_search(0.0001, 0, 2, phi_sym, g_sym_var);
        
        gamma_opt = (a + b) / 2;
        
        % 4. Update
        g_hist = [g_hist gamma_opt];
        
        x_curr(1) = x_curr(1) + gamma_opt * dk(1);
        x_curr(2) = x_curr(2) + gamma_opt * dk(2);
        
        x_hist = [x_hist x_curr(1)];
        y_hist = [y_hist x_curr(2)];
        
        k = k + 1;
    end
end


% LINE SEARCH: BISECTION ON DERIVATIVE
% Finds root of phi'(g) = 0

function [i, a, b, a_list, b_list] = bisection_derivative_search(epsilon, a, b, phi_sym, g_var)
    
    % Calculate derivative d(phi)/dg
    dphi_sym = diff(phi_sym, g_var);
    dphi_func = matlabFunction(dphi_sym);
    
    % Determine iterations based on tolerance
    n = floor(log(epsilon / (b - a)) / log(0.5));
    
    a_list = [a];
    b_list = [b];
    
    for i = 1:n
        x_mid = (a + b) / 2;
        
        % Evaluate derivative at midpoint
        deriv_val = dphi_func(x_mid);
        
        if deriv_val == 0
            break;
        elseif deriv_val < 0
            % If slope is negative, minimum is to the right
            a = x_mid;
        else
            % If slope is positive, minimum is to the left
            b = x_mid;
        end
        
        a_list = [a_list a];
        b_list = [b_list b];
    end
end