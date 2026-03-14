clear;
clc; close all;

% Define symbolic variables
syms x y

% Define the Objective Function: f(x,y) = x^3 * exp(-x^2 - y^4)

obj_func = x^3 * exp(-x^2 - y^4);

% Simulation Parameters
tol_val = 0.0001; % Tolerance for stopping criteria

% --- Case 1: Start at ( -1, -1) ---
fprintf('Processing Case 1: Start Point (-1, -1)...\n');
[~, iters1, path_u1, path_v1, step_hist1] = grad_descent_solver(obj_func, [-1, -1], tol_val, x, y);
visualize_results(iters1, path_u1, path_v1, obj_func, step_hist1);

% --- Case 2: Start at (1, 1) ---
fprintf('Processing Case 2: Start Point (1, 1)...\n');
[~, iters2, path_u2, path_v2, step_hist2] = grad_descent_solver(obj_func, [1, 1], tol_val, x, y);
visualize_results(iters2, path_u2, path_v2, obj_func, step_hist2);



% --- Case 4: Start at (0, 0) ---
fprintf('Processing Case 3: Start Point (0, 0)...\n');
[~, iters4, path_u4, path_v4, step_hist4] = grad_descent_solver(obj_func, [0, 0], tol_val, x, y);
visualize_results(iters4, path_u4, path_v4, obj_func, step_hist4);



% VISUALIZATION FUNCTION 

function visualize_results(iterations, hist_u, hist_v, func_sym, step_history)
    
    % Convert symbolic function to a numeric handle for plotting
    f_num_handle = matlabFunction(func_sym);
    
    % Generate Grid for Contour Plot
    [grid_u, grid_v] = meshgrid(-3:0.05:3, -3:0.05:3); 
    grid_z = f_num_handle(grid_u, grid_v); 

    % Format strings for filenames
    start_u_str = num2str(hist_u(1), '%.2f');
    start_v_str = num2str(hist_v(1), '%.2f');

    %  FIG 1: Contour Plot & Trajectory 
    figure;
    set(gcf, 'Color', 'w');
   
    contourf(grid_u, grid_v, grid_z, 20); 
    colorbar;
    hold on;
    
    % Plot the Trajectory (Black line, Red dots)
    plot(hist_u, hist_v, '-o', 'Color', 'k', 'LineWidth', 1.5, ...
         'MarkerFaceColor', 'r', 'MarkerSize', 4);
     
    % Plot START point (Green Square)
    plot(hist_u(1), hist_v(1), 's', 'MarkerSize', 10, ...
         'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

    % Plot END point (Yellow Star)
    plot(hist_u(end), hist_v(end), 'p', 'MarkerSize', 14, ...
         'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
    
    xlabel('x axis'); ylabel('y axis');
    title(sprintf('Optimization Path from x_0 = [%.2f, %.2f]', hist_u(1), hist_v(1)));
    grid on; axis equal;
    xlim([-3 3]); ylim([-3 3]);
    



    % FIGURE 2: Convergence of Function Value 
    % Calculate f value at each step of the path
    f_vals_history = arrayfun(@(i) f_num_handle(hist_u(i), hist_v(i)), 1:length(hist_u));
    
    figure;
    set(gcf, 'Color', 'w');
    plot(0:length(f_vals_history)-1, f_vals_history, 'LineWidth', 2, 'Color', 'b');
    title('Convergence of Objective Function f(x,y)');
    xlabel('Iteration Number (k)');
    ylabel('Function Value f(x_k)');
    grid on;


    %  FIGURE 3: Step Size History 
    if ~isempty(step_history)
        figure;
        set(gcf, 'Color', 'w');
        plot(1:length(step_history), step_history, '-o', 'LineWidth', 1.5);
        xlabel('Iteration Number (k)');
        ylabel('Step Size \gamma_k');
        title('Step Size Evolution');
        grid on;
        
    end
end


% SOLVER FUNCTION 

function [curr_pos, iter_count, trace_u, trace_v, trace_gamma] = grad_descent_solver(func_sym, start_pos, tol, sym_u, sym_v)
    
    % Calculate Symbolic Gradient
    g_sym = gradient(func_sym, [sym_u, sym_v]);
    
    % Convert gradient to numeric function handle for better performance
    calc_grad = matlabFunction(g_sym); 

    iter_count = 0;
    curr_pos = start_pos;
    
    % Initialize history arrays
    trace_u = [curr_pos(1)];
    trace_v = [curr_pos(2)];
    trace_gamma = [];

    max_loops = 50; % Safety break

    while iter_count < max_loops
        % Evaluate gradient at current position
        grad_val = calc_grad(curr_pos(1), curr_pos(2)); 
        
        % Ensure gradient is a column vector
        if numel(grad_val) == 1 
             
             grad_vec = [grad_val; 0]; % Simplistic fallback
        else
             grad_vec = grad_val;
        end
        
        % Check stopping criterion (Norm of gradient)
        if norm(grad_vec) < tol
            break;
        end

        % Descent Direction
        dir_vec = -grad_vec;

        % Fixed Step Size (Gamma)
        % Keeping it small (0.1) for stability
        step_val = 0.1; 
        
        trace_gamma = [trace_gamma step_val];
        
        % Update Position: x_new = x_old + gamma * direction
        curr_pos(1) = curr_pos(1) + step_val * dir_vec(1);
        curr_pos(2) = curr_pos(2) + step_val * dir_vec(2);
        
        % Store new position
        trace_u = [trace_u curr_pos(1)];
        trace_v = [trace_v curr_pos(2)];
        
        iter_count = iter_count + 1;
    end
end