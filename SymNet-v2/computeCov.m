function cov_data = computeCov( data )

 s = size(data,2); % the number of training or test samples
 cov_data = cell(1,s);
 
 % .*10 for both the training data and test data is to facilitate the subsequent computations. 
 % Because the original values of the 3D coordinates of the FPHA dataset are small
 for i = 1:s
     T = data{i} .* 10; % obtain each sample 
     [ a , b ] = size(T);
     T_mean = mean(T,2); 
     T_mean_matrix = repmat(T_mean,1,b);
     T_center = T - T_mean_matrix; 
     T_cov = T_center * T_center'/(b-1); 
     T_cov = T_cov + trace(T_cov) * (1e-6) * eye(a);
     cov_data{i} = T_cov; 
 end
 
end

