function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=size(X,2);
% You need to return the following variables correctly
mahsa=0;
mahsa=sigmoid(transpose(theta)*transpose(X));
costSum = (log(mahsa))*(y) + (log (1 - mahsa))*(1 - y);
theta(1)=0;
stheta=sum(theta.^2);
costT=(lambda/(2*m)).* stheta;
J = ((-1/m) * costSum) + costT;

if J==0
    grad = (1/m) * ((mahsa - transpose(y))*X);
else 
   grad = ((1/m) * ((mahsa - transpose(y))*X)) + ((lambda/m) * transpose(theta)) ; 
end


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
