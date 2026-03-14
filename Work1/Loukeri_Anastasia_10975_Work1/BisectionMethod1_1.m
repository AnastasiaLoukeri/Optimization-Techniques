clear; clc; close all;
f1 = @(x) 5.^x +(2-cos(x)).^2;
f2 = @(x) (x-1).^2 +exp(x-5).*sin(x+3);
f3 = @(x) exp(-3*x)-(sin(x-2)-2).^2;
% 2. Ορισμός Παραμέτρων
a = -1;           % Κάτω όριο
b = 3;           % Άνω όριο
epsilon = 0.0000001:0.0001:0.004999;  % Σταθερά διαχωρισμού ε
l = 1e-2; % Επιθυμητή ακρίβεια τελικού μήκους 



for i = 1:length(epsilon)
    current_eps = epsilon(i);
    % Εκτέλεση και καταγραφή N
    [~, ~, N1_results(i)] = bisection_search(f1, a, b, current_eps, l);
    [~, ~, N2_results(i)] = bisection_search(f2, a, b, current_eps, l);
    [~, ~, N3_results(i)] = bisection_search(f3, a, b, current_eps, l);
end
figure('Name', 'Dichotomous Method');
figure(1);
plot(epsilon, N1_results, 'LineStyle', '--', 'LineWidth', 2);
grid on;
ylabel('Total f_1(x) calculations', 'FontSize', 15);
xlabel('epsilon ε', 'FontSize', 15);
title('Dichotomous Search: N(e)', 'FontSize', 17);
legu1 = legend('f1');
set(legu1, 'FontSize', 12);
figure(2);
plot(epsilon, N2_results, 'LineStyle', '--', 'LineWidth', 2);
grid on;
ylabel('Total f_2(x) calculations', 'FontSize', 15);
xlabel('epsilon ε', 'FontSize', 15);
title('Dichotomous Search: N(e)', 'FontSize', 17);
legu1 = legend('f2');
set(legu1, 'FontSize', 12);
figure(3);
plot(epsilon, N3_results, 'LineStyle', '--', 'LineWidth', 2);
grid on;
ylabel('Total f_3(x) calculations', 'FontSize', 15);
xlabel('epsilon ε', 'FontSize', 15);
title('Dichotomous Search: N(e)', 'FontSize', 17);
legu1 = legend('f3');
set(legu1, 'FontSize', 12);


