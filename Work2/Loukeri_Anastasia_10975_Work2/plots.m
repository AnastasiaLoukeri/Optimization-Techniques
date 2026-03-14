[X, Y] = meshgrid(-3:0.1:3, -3:0.1:3);
Z = X.^3 .* exp(-X.^2 - Y.^4);
figure;
set(gcf, 'Color', 'w');
surf(X, Y, Z);
xlabel('x'); ylabel('y'); zlabel('f(x,y)');
title('Plot of f(x,y) = x^3 e^{-x^2-y^4}');
colorbar;