% Thomas Verbraken - 03/03/2011 - Katholieke Universiteit Leuven
%
% This Matlab toolbox comes with no warranty. If you use it for your
% research paper, please cite the following paper:
% Verbraken, T., Verbeke, W. & Baesens, B., 2011. Novel Profit Maximizing
% Metrics for Measuring Classification Performance of Customer Churn 
% Prediction Models. IEEE Transactions on Knowledge and Data Engineering, 
% under review.
%
% function [H AUC AUCH] = calcPerformance(ROC,alpha,beta)
% Function calculates traditional performance measures (H,AUC,AUCH).
% Input:
%       - ROC: a ROC structure array, created by the calcROCfunction. 
%       - alpha, beta: parameters of the beta distribution for H-measure; 
%         only applicable to the EMPC measure (default alpha=2 and beta=2)
% Output:
%       - H: the H-measure
%       - AUC: area under the ROC curve
%       - AUCH: area under the convex hull

function [H AUC AUCH] = calcPerformance(ROC,alpha,beta)
% define a function handle to calculate B(x,alpha,beta) = ...
%                           ... int_0^x [x^(alpha-1)(1-x)^(beta-1)]dx.
B = @(x,alpha,beta) betainc(x,alpha,beta) .* exp(gammaln(alpha)+gammaln(beta)-gammaln(alpha+beta));

% AUC
AUC = trapz(ROC.F1roc,ROC.F0roc);

% AUCH
AUCH = trapz(ROC.F1ch,ROC.F0ch);

% H-measure
if nargin < 3
    alpha = 2;
    beta = 2;
end
pi0 = ROC.pi0;
pi1 = ROC.pi1;
F0 = ROC.F0ch;
F1 = ROC.F1ch;
% Calculate cost vector
c = [0 ; pi1*diff(F1)./(pi0*diff(F0)+pi1*diff(F1)) ; 1];
% Calculate Lhat
cii = c(1:end-1);
cie = c(2:end);
contr0 = (pi0*(1-F0).*(B(cie,1+alpha,beta)-B(cii,1+alpha,beta)))/B(1,alpha,beta);
contr1 = (pi1*F1.*(B(cie,alpha,1+beta)-B(cii,alpha,1+beta)))/B(1,alpha,beta);
Lhat = sum(contr0+contr1);
% Calculate Lmax and H
Lmax = (pi0*B(pi1,1+alpha,beta) + pi1*B(1,alpha,1+beta)-pi1*B(pi1,alpha,1+beta))/B(1,alpha,beta);
H = 1-Lhat/Lmax;