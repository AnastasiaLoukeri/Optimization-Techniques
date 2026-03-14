% full optimization with genetic algorithm

clear; clc; close all;
set(0, 'DefaultFigureColor', 'w');

% Data generation
f_target = @(u1, u2) sin(u1 + u2) .* sin(u2.^2);
u1_lim = [-1, 2]; 
u2_lim = [-2, 1];

% Train Set
[U1_tr, U2_tr] = meshgrid(linspace(u1_lim(1), u1_lim(2), 20));
u_train = [U1_tr(:), U2_tr(:)];
y_train = f_target(u_train(:,1), u_train(:,2));

% Test Set
[U1_ts, U2_ts] = meshgrid(linspace(u1_lim(1), u1_lim(2), 35));
u_test = [U1_ts(:), U2_ts(:)];
y_test = f_target(u_test(:,1), u_test(:,2));

% GA params
n_gaussians = 15; 
pop_size = 300;  
max_gens = 1500;  
mutation_rate = 0.15;
target_mse = 0.01;

% chromosome: 15 * (c1, c2, s1, s2, w) = 75 genes
n_vars = n_gaussians * 5;

%limits of parameters [c1, c2, s1, s2, w]
lb = repmat([u1_lim(1), u2_lim(1), 0.1, 0.1, -5], 1, n_gaussians);
ub = repmat([u1_lim(2), u2_lim(2), 1.5, 1.5,  5], 1, n_gaussians);

% Initialization
pop = lb + (ub - lb) .* rand(pop_size, n_vars);
mse_history = [];
best_mse = inf;

fprintf('Optimization for all the parameters\n');

% Main loop of GA
for gen = 1:max_gens
    mse_vals = zeros(pop_size, 1);
    
    % MSE
    for i = 1:pop_size
        mse_vals(i) = calculate_mse(pop(i,:), u_train, y_train, n_gaussians);
    end
    
    [current_min, idx] = min(mse_vals);
    if current_min < best_mse
        best_mse = current_min;
        best_params = pop(idx, :);
    end
    
    mse_history(gen) = best_mse;
    
    
    if mod(gen, 20) == 0 || gen == 1
        fprintf('Generation %d: Best MSE = %.6f\n', gen, best_mse);
    end
    
    if best_mse <= target_mse, break; end
    
    %  Roulette Wheel Selection 
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
    pop = max(min(new_pop, ub), lb);
end

% Evaluation
final_mse_test = calculate_mse(best_params, u_test, y_test, n_gaussians);
fprintf('\nFinal MSE on test set %.6f\n', final_mse_test);

% Plot
figure('Name', 'Convergence');
semilogy(mse_history, 'LineWidth', 2); grid on;
xlabel('Generations'); ylabel('MSE');
title('Convergence per generation');

figure('Name', 'Comparison', 'Units', 'normalized', 'Position', [0.1 0.1 0.8 0.4]);
subplot(1,2,1);
surf(U1_ts, U2_ts, reshape(y_test, size(U1_ts)));
shading interp; colormap parula; colorbar;
title('Real Function');

subplot(1,2,2);
[~, y_pred] = calculate_mse(best_params, u_test, y_test, n_gaussians);
surf(U1_ts, U2_ts, reshape(y_pred, size(U1_ts)));
shading interp; colormap parula; colorbar;
title('Approximated Function');


function [mse, y_hat] = calculate_mse(p, u, y, n_g)
    y_hat = zeros(size(u,1), 1);
    for j = 1:n_g
        idx = (j-1)*5 + 1;
        c1 = p(idx); c2 = p(idx+1);
        s1 = p(idx+2); s2 = p(idx+3);
        w  = p(idx+4);
        % output
        G = exp(-( (u(:,1)-c1).^2/(2*s1^2) + (u(:,2)-c2).^2/(2*s2^2) ));
        y_hat = y_hat + w * G;
    end
    mse = mean((y - y_hat).^2);
end