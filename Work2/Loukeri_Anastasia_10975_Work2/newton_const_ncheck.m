clear;
clc; close all;

syms x y

f = x^3 * exp(-x^2 - y^4);

tol = 0.0001;

% --- Case 1: Start at (-1, -1) ---
fprintf('Processing Newton (No Check) for (-1, -1)...\n');
[~, k1, x1, y1, g1, msg1] = newton_no_check(f, [1, 1], tol, x, y);
visualize_results(k1, x1, y1, f, g1, msg1);

% --- Case 2: Start at (1, 1) ---
fprintf('Processing Newton (No Check) for (1, 1)...\n');
[~, k2, x2, y2, g2, msg2] = newton_no_check(f, [-1.2, -0.5], tol, x, y);
visualize_results(k2, x2, y2, f, g2, msg2);

% --- Case 3: Start at (0, 0) ---
fprintf('Processing Newton (No Check) for (0, 0)...\n');
[~, k3, x3, y3, g3, msg3] = newton_no_check(f, [0, 0], tol, x, y);
visualize_results(k3, x3, y3, f, g3, msg3);

% --- Case 4: Start at (-1.2, -0.5) ---
fprintf('Processing Newton (No Check) for (-1.2, -0.5)...\n');
[~, k4, x4, y4, g4, msg4] = newton_no_check(f, [-1.2, -0.5], tol, x, y);
visualize_results(k4, x4, y4, f, g4, msg4);



function visualize_results(k, x_hist, y_hist, f_sym, g_array, msg)

    f_num = matlabFunction(f_sym);
    [x_vals, y_vals] = meshgrid(-2.5:0.05:2.5, -2.5:0.05:2.5);
    z_vals = f_num(x_vals, y_vals);

    x0_str = num2str(x_hist(1), '%.2f');
    y0_str = num2str(y_hist(1), '%.2f');

    % Contour & Path 
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
    if contains(msg, 'Singular') || contains(msg, 'NaN')
        plot(x_hist(end), y_hist(end), 'x', 'MarkerSize', 15, 'LineWidth', 2, 'Color', 'r');
    else
        plot(x_hist(end), y_hist(end), 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end

    xlabel('x'); ylabel('y');
    title({sprintf('Newton (No Hessian Check) from [%s, %s]', x0_str, y0_str), ['Result: ' msg]});
    grid on; axis equal;
    xlim([-2.5 2.5]); ylim([-2.5 2.5]);


    % Convergence 
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
        title('Step Size (Newton Unchecked)');
        grid on;
   
    end
end


%  NEWTON'S METHOD (NO POSITIVE DEFINITE CHECK)

function [x_curr, k, x_hist, y_hist, g_hist, msg] = newton_no_check(f_sym, x_0, tol, sym_x, sym_y)

    grad_sym = gradient(f_sym, [sym_x, sym_y]);
    hess_sym = hessian(f_sym, [sym_x, sym_y]);

    grad_func = matlabFunction(grad_sym);
    hess_func = matlabFunction(hess_sym);

    k = 1;
    x_curr = x_0;

    x_hist = [x_curr(1)];
    y_hist = [x_curr(2)];
    g_hist = [];
    msg = 'Not Converged';

    max_iter = 500;

    while k < max_iter
        % 1. Gradient
        g_val_temp = grad_func(x_curr(1), x_curr(2));
        if numel(g_val_temp) == 1
             grad_val = [g_val_temp; 0]; 
        else
             grad_val = g_val_temp;
        end

        if norm(grad_val) < tol
            break;
        end

        % 2. Hessian
        H_val = hess_func(x_curr(1), x_curr(2));
        if numel(H_val) == 1, H_val = [H_val 0; 0 0]; end


        % We blindly try to solve the system H*d = -g

        % Check for singularity (div by zero) strictly to prevent crash
        if rcond(H_val) < 1e-15
            msg = 'Stopped (Singular Matrix - Cannot Invert)';
            break;
        end

        % 3. Newton Step
        dk = -H_val \ grad_val;

        % 4. Update
        gamma = 0.2; % Fixed step for Newton
        g_hist = [g_hist gamma];

        x_curr(1) = x_curr(1) + gamma * dk(1);
        x_curr(2) = x_curr(2) + gamma * dk(2);

        % Check for NaN/Inf explosion
        if any(isnan(x_curr)) || any(isinf(x_curr))
            msg = 'Stopped (Values exploded to NaN/Inf)';
            break;
        end

        x_hist = [x_hist x_curr(1)];
        y_hist = [y_hist x_curr(2)];

        k = k + 1;
    end
end
