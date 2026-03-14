clear;
close all;

syms x y
f=(1/3)*x^2+3*y^2;
%CASE 1
[sol1,k1,x1,y1,g1]=steepest(f,[5 -5],0.01,x,y,0.5,5);
visualize_results(k1,x1,y1,f,g1);
%CASE 2
[sol4,k4,x4,y4,g4]=steepest(f,[-5 10],0.01,x,y,0.1,15);
visualize_results(k4,x4,y4,f,g4);
%IMPROVED CASE 2
[sol5,k5,x5,y5,g5]=steepest(f,[-5 10],0.01,x,y,0.1,3);
visualize_results(k5,x5,y5,f,g5);
%CASE 3
[sol3,k3,x3,y3,g3]=steepest(f,[8 -10],0.01,x,y,0.2,0.1);
visualize_results(k3,x3,y3,f,g3);


function visualize_results(iterations, hist_u, hist_v, func_sym, step_history)
    
    % Convert symbolic function to a numeric handle for plotting
    f_num_handle = matlabFunction(func_sym);
    
    % Generate Grid for Contour Plot
    [grid_u, grid_v] = meshgrid(-3:0.05:3, -3:0.05:3); 
    grid_z = f_num_handle(grid_u, grid_v); 
    
    %  FIGURE 1: Contour Plot & Trajectory 
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
    
    %  FIGURE 2: Convergence of Function Value 
    % Calculate f value at each step of the path
    f_vals_history = arrayfun(@(i) f_num_handle(hist_u(i), hist_v(i)), 1:length(hist_u));
    
    figure;
    set(gcf, 'Color', 'w');
    plot(0:length(f_vals_history)-1, f_vals_history, 'LineWidth', 2, 'Color', 'b');
    title('Convergence of Objective Function f(x,y)');
    xlabel('Iteration Number (k)');
    ylabel('Function Value f(x_k)');
    grid on;

    %  FIGURE 3: Evolution of Variables x and y 
    
    figure;
    set(gcf, 'Color', 'w');
    
    iter_axis = 0:length(hist_u)-1; % Create x-axis based on actual history length
    
    % Plot x 
    plot(iter_axis, hist_u, '-', 'LineWidth', 1, 'DisplayName', 'x_1');
    hold on;
    % Plot y 
    plot(iter_axis, hist_v, '-', 'LineWidth', 1, 'DisplayName', 'x_2');
    
    title('Convergence of x_1,x_2');
    xlabel('Iteration Number (k)');
    
    legend('show', 'Location', 'best'); % label for each line
    grid on;
    
end



function [x_k,k,x_array,y_array,g_array]=steepest(f,x_0,e,x,y,g,sk)
    grad=gradient(f,[x,y]);

    k=1;

    x_k(1)=x_0(1);
    x_k(2)=x_0(2);

    grad_f=subs(grad,{x,y},{x_k(1),x_k(2)});
    x_array=[x_k(1)];
    y_array=[x_k(2)];

    g_array=[];
    
    while norm(grad_f)>e

        grad_f=subs(grad,{x,y},{double(x_k(1)),double(x_k(2))});
        dk=-grad_f;
        
        a=x_k(1)+dk(1)*sk;
        b=x_k(2)+dk(2)*sk;

        if (a<=5 && a>=-10)
            x_bar(1)=a;
        elseif(a<-10)
            x_bar(1)=-10;
        elseif(a>5)
            x_bar(1)=5;
        end

        if (b<=12 && b>=-8)
            x_bar(2)=b;
        elseif(b<-8)
            x_bar(2)=-8;
        elseif(b>12)
            x_bar(2)=12;
        end

        x_k(1)=x_k(1)+g*(x_bar(1)-x_k(1));
        
        x_k(2)=x_k(2)+g*(x_bar(2)-x_k(2));

        g_array=[g_array g];
        x_array=[x_array x_k(1)];
        y_array=[y_array x_k(2)];
        k=k+1;

        if k>400
            break;
        end
    end

end