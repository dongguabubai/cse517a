function bias=recoverBias(K,yTr,alphas,C)
% function bias=recoverBias(K,yTr,alphas,C);
%
% INPUT:	
% K : nxn kernel matrix
% yTr : nx1 input labels
% alphas  : nx1 vector or alpha values
% C : regularization constant
% 
% Output:
% bias : the hyperplane bias of the kernel SVM specified by alphas
%
% Solves for the hyperplane bias term, which is uniquely specified by the support vectors with alpha values
% 0<alpha<C
%

temp = (C-alphas).*alphas;
ind = find(temp == max(temp));
K = K(:,ind);
yi = yTr(ind);
bias = 1./yi - alphas'.*(yTr')*K;
