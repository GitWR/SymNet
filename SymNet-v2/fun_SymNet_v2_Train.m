function [eigvector, eigvalue, elapse] = fun_SymNet_v2_Train(options,gnd,data)

% This KLDA code is from Dr. Deng Cai (courtesy) at http://www.cad.zju.edu.cn/home/dengcai/

options.ReguAlpha = 0.01;
K = data;
clear data;
K = max(K,K');

% ====== Initialization
nSmp = size(K,1);
if length(gnd) ~= nSmp
    error('gnd and data mismatch!');
end

classLabel = unique(gnd);
nClass = length(classLabel);
Dim = nClass - 1;

K_orig = K;

sumK = sum(K,2);
H = repmat(sumK./nSmp,1,nSmp);
K = K - H - H' + sum(sumK)/(nSmp^2);
K = max(K,K');
clear H;

%======================================
% SVD decomposition
%======================================

Hb = zeros(nClass,nSmp);
for i = 1:nClass
    index = find(gnd==classLabel(i));
    classMean = mean(K(index,:),1);
    Hb (i,:) = sqrt(length(index))*classMean;
end
B = Hb'*Hb;
T = K*K;

for i=1:size(T,1)
    T(i,i) = T(i,i) + options.ReguAlpha;
end

B = double(B);
T = double(T);

B = max(B,B');
T = max(T,T');

option = struct('disp',0);
[eigvector, eigvalue] = eigs(B,T,Dim,'la',option);
eigvalue = diag(eigvalue);

  
tmpNorm = sqrt(sum((eigvector'*K_orig).*eigvector',2));
eigvector = eigvector./repmat(tmpNorm',size(eigvector,1),1); 
