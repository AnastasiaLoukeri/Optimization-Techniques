clear; close all;
[X, Y] = meshgrid(-3:0.1:3, -3:0.1:3);
Z = (1/3)*X.^2 + 3*Y.^2;
figure;
set(gcf, 'Color', 'w');
surf(X, Y, Z);
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
title('Plot of f(x_1,x_2)');
colorbar;