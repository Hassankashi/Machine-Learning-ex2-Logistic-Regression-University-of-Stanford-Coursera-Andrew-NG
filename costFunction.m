function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

i = 1;
costSum = 0;
mahsa=0;
mahsa=sigmoid(transpose(theta)*transpose(X));
costSum = (log(mahsa))*(y) + (log (1 - mahsa))*(1 - y);

J = (-1/m) * costSum;

grad = (1/m) * ((mahsa - transpose(y))*X);


end
