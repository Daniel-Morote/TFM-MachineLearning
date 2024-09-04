% Thomas Verbraken - 03/03/2011 - Katholieke Universiteit Leuven
%
% This Matlab toolbox comes with no warranty. If you use it for your
% research paper, please cite the following paper:
% Verbraken, T., Verbeke, W. & Baesens, B., 2011. Novel Profit Maximizing
% Metrics for Measuring Classification Performance of Customer Churn 
% Prediction Models. IEEE Transactions on Knowledge and Data Engineering, 
% under review.
%
% function ROC = fn_calcROC(score,class,swap)
% Function calculates the ROC curve, Convex Hull, prior probabilities and
% number of observations and stores it in a structure array.
% Input:
%   - score:    A score for each instance, where higher scores correspond 
%               to cases - dimension = (Nx1).
%   - class:    A vector with the known class. A one (1) is representing a
%               case (e.g. a churner) - dimension = (Nx1).
%   - positive: A boolean indicating which label (0/1) represents a 
%               positive (case). Default value is 1.
% Output:
%   - ROC: a structure array containing information about the ROC curve
%
% IMPORTANT: This algorithm always takes the zero class as the positives 
% (cases). Assuming this, the ROC curve and convex hull can be directly 
% plotted by taking F1 as x-coordinate and F0 as y-coordinate.


function ROC = calcROC(score,class,positive)
% Set default value for positive, if missing
if nargin < 3
    positive = 1;
end

% Swap the labels to construct a ROC curve according to the convention that
% zeros represent cases
if positive==1
    score = -score;
    class = 1-class;
end

% Rank the score from small to large and apply the same to the class vector
[score,idx] = sort(score,'ascend');
class = class(idx);
class = [1-class class];

% Prior probabilities
n0 = sum(class(:,1)==1);
n1 = sum(class(:,2)==1);
pi0 = n0/(n0+n1);
pi1 = n1/(n0+n1);

% Structure array for the ROC curve characteristics
ROC = struct();
ROC.n0 = n0;
ROC.n1 = n1;
ROC.pi0 = pi0;
ROC.pi1 = pi1;
%-------------------------------------------------------------------------
%% Empirical ROC
%  -------------
% Fast Method
[~,id1] = unique(score,'last');
R = cumsum(class);
R = [0 0; R(id1,:)./repmat([n0 n1],[size(id1,1),1])];
% Store in structure array
ROC.F1roc = R(:,2); 
ROC.F0roc = R(:,1);
%-------------------------------------------------------------------------
%% Construct the Convex Hull
%  -------------------------
ROC.F1ch = 0;
ROC.F0ch = 0;
j = 1;
while j<size(R,1)
    toler = 10e-9;
    slope_next = (ROC.F0roc(j+1:end)-ROC.F0roc(j))./(ROC.F1roc(j+1:end)-ROC.F1roc(j));
    imax = find(slope_next >= (max(slope_next)-toler),1,'last');
    ROC.F1ch(end+1,1) = ROC.F1roc(j+imax);
    ROC.F0ch(end+1,1) = ROC.F0roc(j+imax);
    j = j+imax;
end
