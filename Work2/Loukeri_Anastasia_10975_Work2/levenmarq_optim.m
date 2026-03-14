clear;
clc; close all;

syms x y
f = x^3 * exp(-x^2 - y^4);

tol = 0.0001;

% --- Case 1: Start at (1, -1) ---
fprintf('Processing Levenberg  for (-1, -1)...\n');
[~, k1, x1, y1, g1, msg1] = levenberg_exact(f, [-1, -1], tol, x, y);
visualize_results(k1, x1, y1, f, g1, msg1);

% --- Case 2: Start at (-1, 1) ---
fprintf('Processing Levenberg for (1, 1)...\n');
[~, k2, x2, y2, g2, msg2] = levenberg_exact(f, [1, 1], tol, x, y);
visualize_results(k2, x2, y2, f, g2, msg2);

% --- Case 3: Start at (0, 0) ---
fprintf('Processing Levenberg for (0, 0)...\n');
[~, k3, x3, y3, g3, msg3] = levenberg_exact(f, [0, 0], tol, x, y);
visualize_results(k3, x3, y3, f, g3, msg3);


function visualize_results(k, x_hist, y_hist, f_sym, g_array, msg)

    f_num = matlabFunction(f_sym);
    [x_vals, y_vals] = meshgrid(-2.5:0.05:2.5, -2.5:0.05:2.5);
    z_vals = f_num(x_vals, y_vals);

    x0_str = num2str(x_hist(1), '%.2f');
    y0_str = num2str(y_hist(1), '%.2f');

    figure;
    set(gcf, 'Color', 'w');
    contourf(x_vals, y_vals, z_vals, 30); 
    colorbar;
    hold on;
    
    % Path
    plot(x_hist, y_hist, '-o', 'Color', 'k', 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'r', 'MarkerSize', 4);
    % Start (Green Square)
    plot(x_hist(1), y_hist(1), 's', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    % End Marker logic
    if contains(msg, 'NaN') || k >= 200
        % Red Cross if failed/max iter
        plot(x_hist(end), y_hist(end), 'x', 'MarkerSize', 15, 'LineWidth', 2, 'Color', 'r');
    else
        % Yellow Star if converged
        plot(x_hist(end), y_hist(end), 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    end

    xlabel('x'); ylabel('y');
    title({sprintf('Levenberg-Marquardt [%s, %s]', x0_str, y0_str), ['Result: ' msg]});
    grid on; axis equal;
    xlim([-2.5 2.5]); ylim([-2.5 2.5]);
    %saveas(gcf, ['contour_levenberg_exact_x0_', x0_str, '_y0_', y0_str, '.png']);

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
        title('Step Size Evolution (Exact Search)');
        grid on;
        %saveas(gcf, ['step_size_levenberg_exact_x0_', x0_str, '_y0_', y0_str, '.png']);
    end
end



%  LEVENBERG-MARQUARDT with EXACT LINE SEARCH
function [x_curr, k, x_hist, y_hist, g_hist, msg] = levenberg_exact(f_sym, x_0, tol, sym_x, sym_y)
    
    % Pre-calculate handles for speed
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
    
    % Symbolic variable for step size gamma
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

        % 3. LEVENBERG MODIFICATION
        eig_vals = eig(H_val);
        min_eig = min(eig_vals);
        
        % If Hessian is not positive definite, shift eigenvalues
        if min_eig <= 0
            delta = 0.1; 
            mu = abs(min_eig) + delta;
            H_mod = H_val + mu * eye(2);
        else
            H_mod = H_val;
        end

        % 4. Solve system
        dk = -H_mod \ grad_val;

        % 5. EXACT LINE SEARCH (Derivative Bisection)
        % Create symbolic phi(g) = f(x + g*d)
        x_next_sym = x_curr(1) + g_var * dk(1);
        y_next_sym = x_curr(2) + g_var * dk(2);
        
        phi_sym = subs(f_sym, {sym_x, sym_y}, {x_next_sym, y_next_sym});
        
        % Search range [0, 2] usually works for Newton-like steps
        [~, a, b] = derivative_bisection(0.0001, 0, 2, phi_sym);
        gamma = (a + b) / 2;

        g_hist = [g_hist gamma];
        
        % 6. Update
        x_curr(1) = x_curr(1) + gamma * dk(1);
        x_curr(2) = x_curr(2) + gamma * dk(2);
        
        if any(isnan(x_curr)) || any(isinf(x_curr))
             msg = 'Stopped (Values exploded)';
             break;
        end
        
        x_hist = [x_hist x_curr(1)];
        y_hist = [y_hist x_curr(2)];
        
        k = k + 1;
    end
end

% --------------------------------------------------------
% LINE SEARCH: BISECTION ON DERIVATIVE (dixotomos_der)
% --------------------------------------------------------
function [iterations, a, b] = derivative_bisection(tol, a, b, phi_sym)
    n = floor(log(tol / (b - a)) / log(0.5));
    
    % Differentiate symbolic phi w.r.t g_var
    % We need to find the variable in phi_sym to differentiate
    vars = symvar(phi_sym);
    if isempty(vars)
        % Constant function, derivative is 0 everywhere
        iterations = 0;
        return; 
    end
    g_var = vars(1); % Assuming only one variable (g_var)
    
    df_sym = diff(phi_sym, g_var);
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