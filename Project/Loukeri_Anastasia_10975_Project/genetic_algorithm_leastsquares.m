
%  Roulette Wheel Selection


clear; clc; close all;
set(0, 'DefaultFigureColor', 'w'); 

% Data Generation
f_true = @(u1, u2) sin(u1 + u2) .* sin(u2.^2);
u1_range = [-1, 2];
u2_range = [-2, 1];

% Training
[U1_tr, U2_tr] = meshgrid(linspace(u1_range(1), u1_range(2), 20));
u_train = [U1_tr(:), U2_tr(:)];
y_train = f_true(u_train(:,1), u_train(:,2));

%  (Validation/Test)
[U1_ts, U2_ts] = meshgrid(linspace(u1_range(1), u1_range(2), 35));
u_test = [U1_ts(:), U2_ts(:)];
y_test = f_true(u_test(:,1), u_test(:,2));

% Genetic algorithm selection
n_gaussians = 15;      
pop_size = 120;        
max_gens = 500;        
mutation_rate = 0.15;  
target_mse = 0.01;     

% Gaussian Parameters
n_vars = n_gaussians * 4;
lb = repmat([u1_range(1), u2_range(1), 0.1, 0.1], 1, n_gaussians);
ub = repmat([u1_range(2), u2_range(2), 1.5, 1.5], 1, n_gaussians);

% Initialization of population
pop = lb + (ub - lb) .* rand(pop_size, n_vars);
mse_history = [];
best_mse = inf;

fprintf('Searching with Roulette Wheel Selection\n');

%Genetic Algorithm
for gen = 1:max_gens
    mse_vals = zeros(pop_size, 1);
    weights_all = cell(pop_size, 1);
    
    % Fitness 
    for i = 1:pop_size
        [mse_vals(i), weights_all{i}] = calculate_model(pop(i,:), u_train, y_train, n_gaussians);
    end
    
    % best solution
    [min_mse, idx] = min(mse_vals);
    if min_mse < best_mse
        best_mse = min_mse;
        best_params = pop(idx, :);
        best_w = weights_all{idx};
    end
    
    mse_history(gen) = best_mse;
    fprintf('Generation %d: MSE = %.6f\n', gen, best_mse);
    
    if best_mse <= target_mse, break; end
    
    % Roulette Wheel Selection 
    fitness = 1 ./ (mse_vals + eps); 
    prob = fitness / sum(fitness);
    cum_prob = cumsum(prob);
    
    new_pop = zeros(size(pop));
    for i = 1:pop_size
        r = rand();
        sel_idx = find(cum_prob >= r, 1, 'first');
        new_pop(i,:) = pop(sel_idx, :);
    end
    
    % Arithmetic Crossover 
    for i = 1:2:pop_size-1
        if rand < 0.8
            alpha = rand();
            p1 = new_pop(i,:); p2 = new_pop(i+1,:);
            new_pop(i,:) = alpha*p1 + (1-alpha)*p2;
            new_pop(i+1,:) = alpha*p2 + (1-alpha)*p1;
        end
    end
    
    % Mutation 
    mask = rand(size(new_pop)) < mutation_rate;
    new_pop(mask) = new_pop(mask) + 0.1 * randn(sum(mask(:)), 1);
    
    % Limit check
    pop = max(min(new_pop, ub), lb);
end

% Evaluation on Test set
[final_mse_test, y_pred] = calculate_model(best_params, u_test, y_test, n_gaussians, best_w);

% Plots

% Convergence of MSE
figure('Name', 'Convergence Curve');
semilogy(mse_history, 'LineWidth', 2, 'Color', 'b');
hold on; grid on;
yline(target_mse, 'r--', 'Target 0.01');
xlabel('Generations'); ylabel('MSE');
title('Convergence per generation');


figure('Name', 'Function Comparison', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.4]);
subplot(1,2,1);
surf(U1_ts, U2_ts, reshape(y_test, size(U1_ts)));
shading interp; colormap parula; colorbar;
title('Real Function');
xlabel('u_1'); ylabel('u_2');

subplot(1,2,2);
surf(U1_ts, U2_ts, reshape(y_pred, size(U1_ts)));
shading interp; colormap parula; colorbar;
title('Approximated Function');
xlabel('u_1'); ylabel('u_2');


figure('Name', 'Absolute Error');
surf(U1_ts, U2_ts, reshape(abs(y_test - y_pred), size(U1_ts)));
shading interp; colormap parula; colorbar;
title('Absolute Error');
