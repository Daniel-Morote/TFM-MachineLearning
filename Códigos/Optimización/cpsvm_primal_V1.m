% Primal CPSVM, with linear kernel.
% Article: Twin SVM for conditional probability estimation in binary and multiclass classification by Shao et al. 2023

function [Prediction,tf, Sol] = cpsvm_primal_V1(data, labels, Xtest, FunPara)
% INPUT
% data: num-by-dim matrix. num is the number of data points,
% dim is the dimension of a point
% labels: num-by-1 vector, specifying the class that each point
% belongs to either be +1 or be -1
% FunPara.C1 and FunPara.C2: the tuning parameters
% FunPara.epsilon: epsilon parameter
%
% OUTPUT
% Sol.w: num-by-1 vector, optimal weights
% Sol.b: a scalar, the bias


[num, dim] = size(data);
C1=FunPara.C1;
C2=FunPara.C2;
epsilon=FunPara.epsi;

t0=cputime;
cvx_begin
cvx_quiet true
cvx_precision('low')
    variable w(dim);
    variable b;
    variable xi(num);
    minimize(0.5*sum_square(w)  + C1 * sum(xi)/epsilon -C2 * sum(labels .* (data * w + b)));
    subject to
    labels .* (data * w + b - 0.5*epsilon) >=  - xi;
    data * w + b >= 0;
    data * w + b <= 1;
    xi >= 0;
cvx_end
tf=cputime-t0;
Prob_Xt=Xtest*w + b;
Prediction=sign(Prob_Xt-0.5*epsilon);
Sol.w=w;
Sol.b=b;
Sol.Prob=Prob_Xt;


end
