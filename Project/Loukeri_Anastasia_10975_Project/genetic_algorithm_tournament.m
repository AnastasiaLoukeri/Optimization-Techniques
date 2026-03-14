%% Project: Function Approximation with Genetic Algorithm

clear; clc; close all;
set(0, 'DefaultFigureColor', 'w');
% Data generation
f_target = @(u1, u2) sin(u1 + u2) .* sin(u2.^2);
u1_lims = [-1, 2];
u2_lims = [-2, 1];

% Training Set
[U1_tr, U2_tr] = meshgrid(linspace(u1_lims(1), u1_lims(2), 20));
u_train = [U1_tr(:), U2_tr(:)];
y_train = f_target(u_train(:,1), u_train(:,2));

% Test Set
[U1_ts, U2_ts] = meshgrid(linspace(u1_lims(1), u1_lims(2), 30));
u_test = [U1_ts(:), U2_ts(:)];
y_test = f_target(u_test(:,1), u_test(:,2));

% Genetic algorithm parameters
n_gaussians = 15;      
pop_size = 300;        
max_generations = 300; 
mutation_rate = 0.2;   
target_mse = 0.001;

% 10 * 4 = 40 genes (c1, c2, s1, s2 for every gaussian)
n_vars = n_gaussians * 4;
lb = repmat([u1_lims(1), u2_lims(1), 0.1, 0.1], 1, n_gaussians);
ub = repmat([u1_lims(2), u2_lims(2), 1.5, 1.5], 1, n_gaussians);

% Initialization
pop = lb + (ub - lb) .* rand(pop_size, n_vars);
mse_history = [];
best_mse = inf;

fprintf('Optimization');

% Main Loop of Genetic algorithm
for gen = 1:max_generations
    mse_vals = zeros(pop_size, 1);
    weights_store = cell(pop_size, 1);
    
    % Fitness Evaluation
    for i = 1:pop_size
        [mse_vals(i), weights_store{i}] = get_fitness(pop(i,:), u_train, y_train, n_gaussians);
    end
    
    % Best
    [current_min_mse, best_idx] = min(mse_vals);
    if current_min_mse < best_mse
        best_mse = current_min_mse;
        best_individual = pop(best_idx, :);
        best_w = weights_store{best_idx};
    end
    
    mse_history(gen) = best_mse;
    fprintf('Generation %d: Best MSE = %.6f\n', gen, best_mse);
    
    if best_mse <= target_mse, break; end
    
    % Tournament Selection
    new_pop = zeros(size(pop));
    for i = 1:pop_size
        pool = randi(pop_size, [3, 1]);
        [~, winner] = min(mse_vals(pool));
        new_pop(i,:) = pop(pool(winner), :);
    end
    
    % Arithmetic Crossover
    for i = 1:2:pop_size-1
        if rand < 0.8
            r = rand();
            p1 = new_pop(i,:); p2 = new_pop(i+1,:);
            new_pop(i,:) = r*p1 + (1-r)*p2;
            new_pop(i+1,:) = r*p2 + (1-r)*p1;
        end
    end
    
    % Mutation
    mutation_mask = rand(size(new_pop)) < mutation_rate;
    new_pop(mutation_mask) = new_pop(mutation_mask) + 0.1 * randn(sum(mutation_mask(:)), 1);
    
    % Bound Constraint
    pop = max(min(new_pop, ub), lb);
end

% Evaluation on test set
[final_mse_test, y_hat_test] = evaluate_final(best_individual, best_w, u_test, y_test, n_gaussians);
fprintf('Final MSE on Test Set: %.6f\n', final_mse_test);


%Plots

% Convergence of MSE
figure('Name', 'Convergence');
plot(mse_history, 'LineWidth', 2); hold on;

yline(target_mse, 'r--', 'Target MSE');
xlabel('Generations'); ylabel('Mean Squared Error');
title('Error per generation'); grid on;

% Real vs Approximated function
figure('Name', 'Results', 'Position', [100, 100, 1000, 400]);

subplot(1,2,1);
surf(U1_ts, U2_ts, reshape(y_test, size(U1_ts)));
title('Real Function'); shading interp; colorbar;

subplot(1,2,2);
surf(U1_ts, U2_ts, reshape(y_hat_test, size(U1_ts)));
title('Approximated Function'); shading interp; colorbar;

% Absolute of error
figure('Name', 'Error Surface');

surf(U1_ts, U2_ts, reshape(abs(y_test - y_hat_test), size(U1_ts)));
title('Mean Square Error'); xlabel('u_1'); ylabel('u_2');
shading interp; colorbar;

fprintf('\n Analytical Expression of Function f(u_1,u_2)\n');
for j = 1:n_gaussians
    idx = (j-1)*4 + 1;
    % parameters of f 
    c = best_individual(idx:idx+1);
    s = best_individual(idx+2:idx+3);
    weight = best_w(j);
    
    % print every gaussian
    fprintf(' Term %d: (%f) * exp( -((u1 - %f)^2 / %f) - ((u2 - %f)^2 / %f) )+\n', ...
        j, weight, c(1), 2*s(1)^2, c(2), 2*s(2)^2);
end

function [mse, w] = get_fitness(params, u, y, n_g)
    G = zeros(size(u,1), n_g);
    for j = 1:n_g
        idx = (j-1)*4 + 1;
        c = params(idx:idx+1); s = params(idx+2:idx+3);
        G(:,j) = exp( -( (u(:,1)-c(1)).^2/(2*s(1)^2) + (u(:,2)-c(2)).^2/(2*s(2)^2) ) );
    end
    w = G \ y; % Least Squares for weights
    mse = mean((y - G*w).^2);
end

function [mse, y_hat] = evaluate_final(params, w, u, y, n_g)
    G = zeros(size(u,1), n_g);
    for j = 1:n_g
        idx = (j-1)*4 + 1;
        c = params(idx:idx+1); s = params(idx+2:idx+3);
        G(:,j) = exp( -( (u(:,1)-c(1)).^2/(2*s(1)^2) + (u(:,2)-c(2)).^2/(2*s(2)^2) ) );
    end
    y_hat = G*w;
    mse = mean((y - y_hat).^2);
end