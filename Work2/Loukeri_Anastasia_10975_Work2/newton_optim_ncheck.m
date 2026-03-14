clear;
clc; close all;

syms x y
% --- OBJECTIVE FUNCTION ---
% f(x,y) = x^3 * exp(-x^2 - y^4)
f = x^3 * exp(-x^2 - y^4);

tol = 0.0001;

% --- Case 1: Start at (1, -1) ---
fprintf('Processing Newton (No Check) for (-1, -1)...\n');
[~, iter1, x1, y1, gam1, msg1] = newton_no_check(f, [1, 1], tol, x, y);
plot_results(iter1, x1, y1, f, gam1, msg1);

% --- Case 2: Start at (-1, 1) ---
fprintf('Processing Newton (No Check) for (1, 1)...\n');
[~, iter2, x2, y2, gam2, msg2] = newton_no_check(f, [-1, -1], tol, x, y);
plot_results(iter2, x2, y2, f, gam2, msg2);

% --- Case 3: Start at (0, 0) ---
fprintf('Processing Newton (No Check) for (0, 0)...\n');
[~, iter3, x3, y3, gam3, msg3] = newton_no_check(f, [0, 0], tol, x, y);
plot_results(iter3, x3, y3, f, gam3, msg3);

% --- Case 4: Start at (-1.2, -0.5) ---
% fprintf('Processing Newton (No Check) for (-1.2, -0.1)...\n');
% [~, iter4, x4, y4, gam4, msg4] = newton_no_check(f, [-1.2, -0.5], tol, x, y);
% plot_results(iter4, x4, y4, f, gam4, msg4);


function plot_results(k, x_hist, y_hist, f_sym, g_array, msg)

    f_num = matlabFunction(f_sym);
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
    title({sprintf('Newton (No Hessian Check) [%s, %s]', x0_str, y0_str), ['Result: ' msg]});
    grid on; axis equal;
    xlim([-2.5 2.5]); ylim([-2.5 2.5]);
    %saveas(gcf, ['contour_newton_nocheck_x0_', x0_str, '_y0_', y0_str, '.png']);

    %  Convergence 
    if ~isempty(x_hist)
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
        
    end

    % Step Size 
    if ~isempty(g_array)
        figure;
        set(gcf, 'Color', 'w');
        plot(1:length(g_array), g_array, '-o');
        xlabel('Iteration (k)'); ylabel('Step Size \gamma');
        title('Step Size Evolution');
        grid on;
        
    end
end


% NEWTON (NO HESSIAN CHECK) + EXACT LINE SEARCH

function [x_curr, k, x_hist, y_hist, g_hist, msg] = newton_no_check(f, x_0, tol, x, y)
    
    grad_sym = gradient(f, [x, y]);
    hess_sym = hessian(f, [x, y]);
    
    grad_func = matlabFunction(grad_sym);
    hess_func = matlabFunction(hess_sym);

    k = 1;
    x_curr = x_0;
    
    x_hist = [x_curr(1)];
    y_hist = [x_curr(2)];
    g_hist = [];
    msg = ' Not Converged';
    
    syms g_var

    max_iter = 200;

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

        % 
        % We proceed even if Eigenvalues are negative.
        
        % Only check for Singularity (to prevent crash)
        if rcond(H_val) < 1e-15
            msg = 'Stopped (Singular Matrix)';
            break;
        end

        % 3. Newton Direction (Blind)
        descent_dir = -H_val \ grad_val;

        % 4. EXACT LINE SEARCH (Derivative Bisection)
        x_next_sym = x_curr(1) + g_var * descent_dir(1);
        y_next_sym = x_curr(2) + g_var * descent_dir(2);
        
        phi_sym = subs(f, {x, y}, {x_next_sym, y_next_sym});
        
        % Find optimal gamma [0, 2]
        
        % the line search might return 0 or move towards a max.
        [~, a, b] = derivative_bisection(0.0001, 0, 2, phi_sym);
        gamma = (a + b) / 2;

        g_hist = [g_hist gamma];
        
        % 5. Update
        x_curr(1) = x_curr(1) + gamma * descent_dir(1);
        x_curr(2) = x_curr(2) + gamma * descent_dir(2);
        
        % Safety check for explosion
        if any(isnan(x_curr)) || any(isinf(x_curr))
             msg = 'Stopped (NaN/Inf values)';
             break;
        end
        
        x_hist = [x_hist x_curr(1)];
        y_hist = [y_hist x_curr(2)];
        
        k = k + 1;
    end
end


% LINE SEARCH: BISECTION ON DERIVATIVE

function [iterations, a, b] = derivative_bisection(tol, a, b, f_sym)
    n = floor(log(tol / (b - a)) / log(0.5));
    
    df_sym = diff(f_sym);
    df = matlabFunction(df_sym); 
    
    iterations = 0;
    for i = 1:n
        x_mid = (a + b) / 2;
        val = df(x_mid);
        
        if val == 0
            break;
        elseif val < 0
            a = x_mid;
        elseif val > 0
            b = x_mid;
        end
        iterations = i;
    end
end