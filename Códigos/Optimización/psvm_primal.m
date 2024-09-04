% Primal PSVM with linear kernel

function [Prediction,tf, Sol] = psvm_primal(data, labels, Xtest, FunPara)
% INPUT
% data: num-by-dim matrix. num is the number of data points,
% dim is the dimension of a point
% labels: num-by-1 vector, specifying the class that each point
% belongs to either be +1 or be -1
% FunPara.C: the tuning parameter
% FunPara.epsilon: epsilon parameter
%
% OUTPUT
% w: num-by-1 vector, optimal weights
% b: a scalar, the bias
%Paper :Twin SVM for conditional probability estimation in binary and multiclass classification
% Shao et al. 2023
% model PSVM (3)

[num, dim] = size(data);
C=FunPara.C;
epsilon=FunPara.epsi;

t0=cputime;
cvx_begin
cvx_quiet true
cvx_precision('low')
    variable w(dim);
    variable b;
    variable xi(num);
    minimize(0.5*sum_square(w)  + C * sum(xi) / epsilon);
    subject to
    labels .* (data * w + b - 0.5) >= 0.5 * epsilon - xi;
    data * w + b >= 0;
    data * w + b <= 1;
    xi >= 0;
cvx_end
tf=cputime-t0;
Prob_Xt=Xtest*w + b;
Prediction=sign(Prob_Xt-0.5);
Sol.w=w;
Sol.b=b;
Sol.Prob=Prob_Xt;

%     % Calcular AUC y matriz de confusión
%     auc = AUCcalc(Predict, labels);
%     [~, Accu, Sens, Spec, cm] = medi_auc_accu(Predict, labels);
end
