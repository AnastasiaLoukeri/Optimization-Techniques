clear; clc; close all;


f1 = @(x) 5.^x +(2-cos(x)).^2;
f2 = @(x) (x-1).^2 +exp(x-5).*sin(x+3);
f3 = @(x) exp(-3*x)-(sin(x-2)-2).^2; 
a=-1;
b=3;
% Μεταβλητές τιμές για το tolerance - limit l
l_values = 0.01:0.001:0.1;

% Πίνακες για τα αποτελέσματα (N)
N1_results = zeros(size(l_values));
N2_results = zeros(size(l_values));
N3_results = zeros(size(l_values));
for i = 1:length(l_values)
    current_l = l_values(i);
    
    % Εκτέλεση και συμπλήρωση N
    % Golden Section Search
    [~, ~, N1_results(i)] = golden_section_search(f1, a, b, current_l);
    [~, ~, N2_results(i)] = golden_section_search(f2, a, b, current_l);
    [~, ~, N3_results(i)] = golden_section_search(f3, a, b, current_l);
    
    %fprintf('%9.2e  | %6d | %6d | %6d \n', current_l, N1_results(i), N2_results(i), N3_results(i));
end

figure('Name', 'Golden Section Method');
figure(1);
plot(l_values, N1_results, 'LineStyle', '--', 'LineWidth', 2);
grid on;
ylabel('Total f_1(x) calculations', 'FontSize', 15);
xlabel('Limit l', 'FontSize', 15);
title('Golden Section Search: N(l)', 'FontSize', 17);
legu1 = legend('f1');
set(legu1, 'FontSize', 12);
figure(2);
plot(l_values, N2_results, 'LineStyle', '--', 'LineWidth', 2);
grid on;
ylabel('Total f_2(x) calculations', 'FontSize', 15);
xlabel('Limit l', 'FontSize', 15);
title('Golden Section Search: N(l)', 'FontSize', 17);
legu1 = legend('f2');
set(legu1, 'FontSize', 12);
figure(3);
plot(l_values, N3_results, 'LineStyle', '--', 'LineWidth', 2);
grid on;
ylabel('Total f_3(x) calculations', 'FontSize', 15);
xlabel('Limit l', 'FontSize', 15);
title('Golden Section Search: N(l)', 'FontSize', 17);
legu1 = legend('f3');
set(legu1, 'FontSize', 12);