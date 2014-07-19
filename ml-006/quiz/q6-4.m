function g = sigmoid(z)
  g = zeros(size(z));
  g = 1.0 ./ (1.0 + exp(-z));
end


function [J, grad] = linearCostFunction(theta, X, y)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  H_theta = X * theta;
  J = (1.0/2/m) * (H_theta - y)' * (H_theta - y);
  grad = (1.0/m) .* X' * (H_theta - y);
end

function [J, grad] = logitCostFunction(theta, X, y)
  m = length(y);
  J = 0;
  grad = zeros(size(theta));
  H_theta = sigmoid(X * theta);
  J = (1.0/m) * sum(-y .* log(H_theta) - (1.0 - y) .* log(1.0 - H_theta));
  grad = (1.0/m) .* X' * (H_theta - y);
end

function [theta, cost] = linearRegression(X, y)
  [m, n] = size(X);
  X = [ones(m, 1), X];

  initial_theta = zeros(n + 1, 1);
  [cost, grad] = linearCostFunction(initial_theta, X, y);
  fprintf('Cost at initial theta:  %f\n', cost);

  options = optimset('GradObj', 'on', 'MaxIter', 10000);
  [theta, cost] = fminunc(@(t)(linearCostFunction(t, X, y)), initial_theta, options);

  fprintf('Cost at theta found by fminunc: %f\n', cost);
  fprintf('theta: \n');
  fprintf(' %f \n', theta);
end


function [theta, cost] = logitRegression(X, y)
  [m, n] = size(X);
  X = [ones(m, 1), X];

  initial_theta = zeros(n + 1, 1);
  [cost, grad] = logitCostFunction(initial_theta, X, y);
  fprintf('Cost at initial theta:  %f\n', cost);

  options = optimset('GradObj', 'on', 'MaxIter', 10000);
  [theta, cost] = fminunc(@(t)(logitCostFunction(t, X, y)), initial_theta, options);

  fprintf('Cost at theta found by fminunc: %f\n', cost);
  fprintf('theta: \n');
  fprintf(' %f \n', theta);
end


function plotLinearSolution(theta, X, y)
  figure
  [xx, yy] = ndgrid(1:0.2:3, 0:0.2:2);
  zz = theta(1) + theta(2) * xx + theta(3) * yy;
  surf(xx, yy, zz);
  hold on;
  scatter3(X(:,1), X(:,2), y, 37);
  xlabel('x1');
  ylabel('x2');
  zlabel('y');
  hold off;
end

function plotLogitSolution(theta, X, y)
  figure
  [xx, yy] = ndgrid(1:0.2:3, 0:0.2:2);
  zz = sigmoid(theta(1) + theta(2) * xx + theta(3) * yy);
  surf(xx, yy, zz);
  hold on;
  scatter3(X(:,1), X(:,2), y, 37);
  xlabel('x1');
  ylabel('x2');
  zlabel('y');
  hold off;
end

y = [0, 0, 1, 0]';
X = [1, 0.5; 1, 1.5; 2, 1; 3, 1];

theta = linearRegression(X, y);
plotLinearSolution(theta, X, y);

theta = logitRegression(X, y);
plotLogitSolution(theta, X, y);
