clc; clear; close all;
f_data = struct(...
    'f', { @(x) 5.^x + (2 - cos(x)).^2, @(x) (x-1).^2 + exp(x-5).*sin(x+3), @(x) exp(-3*x)-(sin(x-2)-2).^2}, ...
    'a', {-1, -1, -1}, ...
    'b', {3, 3, 3}, ...
    'L0', {5, 6, 9}, ...
    'title', {'f_1(x) = 5^x + (2 - cos(x))^2 on [-1, 3]', 'f_2(x) = (x-1)^2 + exp(x-5)sin(x+3) on [-1, 3]', 'f_3(x) = exp(-3*x)-(sin(x-2)-2)^2 on [-1, 3]'});


l_values = [0.003, 0.006, 0.01, 0.1]; % Διάφορες τιμές του l
l_colors = {'r', 'b', 'g', 'm'};
l_styles = {'-', '--', ':', '-.'};

% Εκτέλεση και Σχεδίαση (3 Διαγράμματα)
for i = 1:3 % Loop μέσω των 3 συναρτήσεων
    figure;
    hold on;
    
    f = f_data(i).f;
    a = f_data(i).a;
    b = f_data(i).b;
    function_title = f_data(i).title;
    
    %fprintf('Execution for: %s\n', function_title);
    
    for j = 1:length(l_values) % Loop μέσω των διαφόρων τιμών του l
        l = l_values(j);
        
        % 1.Ληψη ιστορικού
        [a_hist, b_hist, k_hist] = golden_section_history(f, a, b, l);
        
        if isempty(a_hist) || length(a_hist) < 2
            fprintf('  l=%.3f: Αποτυχία ή άμεσος τερματισμός.\n', l);
            continue;
        end
        
        % 2. Σχεδίαση (k, ak)
        plot(k_hist, a_hist, [l_colors{j}, l_styles{j}], 'LineWidth', 2, ...
             'DisplayName', sprintf('a_k, l=%.3f (K=%d)', l, k_hist(end)));
        
        % 3. Σχεδίαση (k, bk)
        plot(k_hist, b_hist, [l_colors{j}, l_styles{j}], 'LineWidth', 2, ...
             'DisplayName', sprintf('b_k, l=%.3f (K=%d)', l, k_hist(end)));
    end
    
    
    title(sprintf('Convergence of Interval [a_k, b_k] vs Iterations k for %s ', function_title));
    xlabel('Iterations k');
    ylabel('Interval limits (a_k, b_k)');
    grid on;
    legend('Location', 'best', 'NumColumns', 2);
    hold off;
end