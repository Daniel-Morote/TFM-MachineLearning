% Solving the following Quadratic Problems with CVX solver
% SVM (linear case)

% Min 0.5*|w|^2+ C*(Xi2'*e2+Xi1'*e1)
% s.t. 
%       A*w+b*e1>=e1-Xi1;
%      -B*w-b*e2>=e2-Xi2,
%        Xi1,Xi2>=0.


function [Ytest,Tf,Val_TestX]=SVM_softcvx(X,Y,TestX,FunPara)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%       Ytest =SVM_softcvx(X,Y,TestX,P)
%
%       Input:
%               X           - Training data matrix (Each row vector is a data point).
%               Y           - Training Label data (-1 or 1)
%               TestX       - Test Data matrix (Each row vector is a data point).
%
%               FunPara - Struct value in Matlab. The fields in options
%                         that can be set:
%                   c: [0,inf] Parameter to tune the weight.

%                   kerfPara:Kernel parameters. See kernelfun.m.
%
%       Output:
%               Ytest - Predict value of the TestX.
%
%       Example:
%           A = rand(50,10);
%           B = rand(60,10);
%           X=[A;B];
%           Y=[ones(50,1);-ones(60,1)];
%           TestX=rand(20,10);
%           FunPara.c=2^(2);
%
%           Ytest=SVM_softcvx(X,Y,TestX,P);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initailization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%tic;

fin1=find(Y==1);
fin2=find(Y==-1);
A=X(fin1,:);
B=X(fin2,:);

c= FunPara.c;

m1=size(A,1);
m2=size(B,1);
e1=ones(m1,1);
e2=ones(m2,1);
n=size(A,2);

t0=cputime;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Compute (w,b) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cvx_begin quiet
    cvx_precision('low')
    cvx_solver sedumi  
    variables w(n) b Xi1(m1) Xi2(m2)  
    minimize(0.5*sum_square(w)+c*(sum(Xi2)+sum(Xi1)))
subject to
A*w+b*e1>=e1-Xi1;
-B*w-b*e2>=e2-Xi2;
Xi2>=0;
Xi1>=0;

cvx_end
Tf=cputime-t0;
%toc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m=size(TestX,1);

H=TestX;
Val_TestX=H*w+b*ones(m,1);
Ytest=sign(Val_TestX);
end
