function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
% function [bestC,bestP,bestval,allvalerrs]=crossvalidate(xTr,yTr,ktype,Cs,paras)
%
% INPUT:	
% xTr : dxn input vectors
% yTr : 1xn input labels
% ktype : (linear, rbf, polynomial)
% Cs   : interval of regularization constant that should be tried out
% paras: interval of kernel parameters that should be tried out
% 
% Output:
% bestC: best performing constant C
% bestP: best performing kernel parameter
% bestval: best performing validation error
% allvalerrs: a matrix where allvalerrs(i,j) is the validation error with parameters Cs(i) and paras(j)
%
% Trains an SVM classifier for all possible parameter settings in Cs and paras and identifies the best setting on a
% validation split. 
%



%% Split off validation data set
K = 10;
n = size(xTr,2);
Indices = crossvalind('Kfold', n, K);
cn = size(Cs,2);
pn = size(paras,2);

%% Evaluate all parameter settings
allvalerrs = zeros(cn,pn);
for i=1:cn
    for j=1:pn
        for q=1:K
            Test = (Indices == q);
            Train = ~Test;
            svmclassify=trainsvm(xTr(:,Train),yTr(Train),Cs(i),ktype,paras(j));
            allvalerrs(i,j)=allvalerrs(i,j)+mean(sign(svmclassify(xTr(:,Test)))~=yTr(Test)');
        end
        allvalerrs(i,j)= allvalerrs(i,j)/K;
    end
end

%% Identify best setting
[index1,index2]=find(allvalerrs == min(min(allvalerrs)));
ind = randi([1,size(index1,1)]);
index1=index1(ind,:);
index2=index2(ind,:);
bestC=Cs(index1);
bestP=paras(index2);
bestval=allvalerrs(index1,index2);


