function logratio = naivebayes(x,y,x1)
% function logratio = naivebayes(x,y);
%
% Computation of log P(Y|X=x1) using Bayes Rule
% Input:
% x : n input vectors of d dimensions (dxn)
% y : n labels (-1 or +1)
% x1: input vector of d dimensions (dx1)
%
% Output:
% logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))

[d,n] = size(x);
%% fill in code here
[pos,neg] = naivebayesPY(x,y);
[posprob,negprob] = naivebayesPXY(x,y);
logratio=x1'*log(posprob) + log(pos) - x1'*log(negprob) - log(neg);
