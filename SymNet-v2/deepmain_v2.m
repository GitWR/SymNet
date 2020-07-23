clear;
clc;

%author: Rui Wang
%department: School of artificial intelligence and computer science, Jiangnan university 

%% load data

load FPHA_train_label
load FPHA_val_label
load FPHA_train_seq
load FPHA_val_seq

%% Preparation

Train_labels = train_labels;
Test_labels = val_labels;

cov_sum = zeros(63,63); 
cov_all = zeros(63,63);

num_layers_1 = 3; % $m_1^{v2}$
p_dim_1 = 20; % $d_{m1}^{v2}$

num_layers_2 = 4; % $m_2^{v2}$
p_dim_2 = 5; % $d_{m2}^{v2}$

rectified_data_cell = cell(1,size(Train_labels,2));
maps_sum = zeros(p_dim_1,p_dim_1);
Sum_CovMaps = zeros(p_dim_1,p_dim_1);

transfer_each = cell(1,num_layers_2);

second_layer_singletr = cell(1,num_layers_1); 
second_layer_singlete = cell(1,num_layers_1); 

final_train_branch = zeros(p_dim_2^2 * num_layers_2, num_layers_1);
final_test_branch = zeros(p_dim_2^2 * num_layers_2, num_layers_1);

% activation thresholds of the two rectifying layers of SymNet-v2
eps_1 = 4e-3; % $\epsilon_1^{v2}$
eps_2 = 1e-3; % $\epsilon_2^{v2}$

eta_1 = 1e-6;  % $\eta_1^{v2}$
eta_2 = 1e-6; % $\eta_2^{v2}$

%% training stage
Train_data = train_seq; % training samples
cov_train = computeCov(Train_data); % the computed training SPD matrices

% the first SPD matrix mapping layer

for k = 1:size(Train_labels,2)
    cov_sum = cov_sum + cov_train{k};
end

mean_train = cov_sum / k;

for i = 1:size(Train_labels,2)
    cov_all = cov_all + (cov_train{i}-mean_train)'*(cov_train{i}-mean_train);
end

cov_all = cov_all/(size(Train_labels,2)-1);
[e_vectors,e_values] = eig(cov_all);
[~,order] = sort(diag(-e_values));
e_vectors = e_vectors(:,order);

for i = 1:num_layers_1
    T_1{i} = e_vectors(:,(i-1)*p_dim_1+1:i*p_dim_1); % the connection weights of the first mapping layer
end

for i = 1:size(Train_labels,2)
    single_sample_tr = cov_train{i};
    for j = 1:num_layers_1
        second_layer_singletr{j} = T_1{j}'*single_sample_tr*T_1{j}; % first-stage (2D)^2PCA projection
    end
    
    % the first rectifying layer
    for k = 1:num_layers_1
        mid = second_layer_singletr{k};
        idx1 = mid <= 0;
        idx2 = mid > -eta_1;
        idx = idx1 & idx2;
        mid(idx) = -eta_1;
        [U,V,D] = svd(mid);
        [a,b]=size(V);
        tol_1 = trace(V)*eps_1;
        for l = 1:a
            if V(l,l) <= tol_1
                V(l,l) = tol_1;
            end
        end
        second_layer_singletr{k} = U*V*D';
    end
    
    rectified_data_cell{i} = second_layer_singletr;
    
    for s = 1:size(second_layer_singletr,2)
        rectified_maps_matrix(:,:,s) = second_layer_singletr{s};
    end
    
    m = size(second_layer_singletr,2);
    all_rectified_maps(:,:,(i-1)*m+1:i*m) = rectified_maps_matrix;
end

% the second SPD matrix mapping layer

for i = 1:size(all_rectified_maps,3)
    maps_sum = maps_sum + all_rectified_maps(:,:,i);
end

mean_maps = maps_sum / (num_layers_1*size(Train_labels,2));

for j = 1:size(all_rectified_maps,3)
    Sum_CovMaps = Sum_CovMaps + (all_rectified_maps(:,:,j)-mean_maps)'*(all_rectified_maps(:,:,j)-mean_maps);
end

Sum_CovMaps = Sum_CovMaps / (size(all_rectified_maps,3)-1);
[e_vectors,e_values] = eig(Sum_CovMaps); 
[dummy,order] = sort(diag(-e_values)); 
C_Weight = e_vectors(:,order); 

for k = 1:num_layers_2
    W_1{k} = C_Weight(:,(k-1)*p_dim_2+1:k*p_dim_2); % the connection weights of the second mapping layer
end

for l = 1:size(Train_labels,2)
    temp_ch = rectified_data_cell{l}; 
    for r = 1:size(temp_ch,2)
        temp1 = temp_ch{r};
        for s = 1:num_layers_2
            transfer_each{s} = W_1{s}'*temp1*W_1{s}; % second-stage (2D)^2PCA projection
        end
        
        % the second rectifying layer
        for k = 1:num_layers_2
            mid = transfer_each{k};
            idx1 = mid <= 0;
            idx2 = mid > -eta_2;
            idx = idx1 & idx2;
            mid(idx) = -eta_2;
            [U,V,D] = svd(mid);
            [a,b] = size(V);
            tol_2 = trace(V)*eps_2;
            for l1 = 1:a
                if V(l1,l1) <= tol_2
                    V(l1,l1) = tol_2;
                end
            end
            transfer_each{k} = U*V*D';
        end
        
        % log-map layer
        for p = 1:num_layers_2
            log_map{p} = transfer_each{p};
            [u,v,w] = svd(log_map{p});
            logv = log(diag(v));
            log_map{p} = u * diag(logv) * w';
        end
        
        for n = 1:num_layers_2
            [U,V,D] = svd(log_map{n});
            all_trace(n) = trace(V);
        end
        
        for n = 1:num_layers_2
            T_2(n) = all_trace(n)/sum(all_trace);
        end
        
        for o = 1:num_layers_2
            temp2 = reshape(T_2(o)*log_map{o},size(log_map{1},1)*size(log_map{1},2),1);
            temp_final(:,o) = temp2;
        end
        final_train_branch(:,r) = temp_final(:);
    end
 
    train_matrix(:,l) = final_train_branch(:);
    
end

% building the training kernel matrix
deta = 3.50; 

kmatrix_train = zeros(size(Train_labels,2),size(Train_labels,2)); 

for i = 1:size(Train_labels,2)
    for j = 1:size(Train_labels,2)
        cov_i_Train = train_matrix(:,i);
        cov_j_Train = train_matrix(:,j);
        kmatrix_train(i,j) = exp((-norm(cov_i_Train-cov_j_Train)^2)/(2*deta^2));
        kmatrix_train(j,i) = kmatrix_train(i,j);
    end
end

%% test stage

Test_data = val_seq; % test samples
cov_test = computeCov(Test_data); % the computed test SPD matrices 

for i = 1:size(Test_labels,2)
    single_sample_te = cov_test{i};
    for j = 1:num_layers_1
        second_layer_singlete{j} = T_1{j}'*single_sample_te*T_1{j}; % first-stage (2D)^2PCA projection
    end
    
    % first rectifying layer
    for k = 1:num_layers_1
        mid = second_layer_singlete{k};
        idx1 = mid <= 0;
        idx2 = mid > -eta_1;
        idx = idx1 & idx2;
        mid(idx) = -eta_1;
        [U,V,D] = svd(mid);
        [a,b]=size(V);
        tol_1 = trace(V)*eps_1;
        for l=1:a
            if V(l,l) <= tol_1
                V(l,l) = tol_1; 
            end
        end
        second_layer_singlete{k} = U*V*D';
    end
    
    rectified_data_cell{i}=second_layer_singlete;
    
end

for l=1:size(Test_labels,2)
    temp_ch = rectified_data_cell{l};
    for r = 1:size(temp_ch,2)
        temp1 = temp_ch{r};
        for s = 1:num_layers_2
            transfer_each{s} = W_1{s}'*temp1*W_1{s}; % second-stage (2D)^2PCA projection
        end
        
        % second rectifying layer
        for k = 1:num_layers_2
            mid = transfer_each{k};
            idx1 = mid <= 0;
            idx2 = mid > -eta_2;
            idx = idx1 & idx2;
            mid(idx) = -eta_2;
            [U,V,D] = svd(mid);
            [a,b] = size(V);
            tol_2 = trace(V)*eps_2;
            for l1=1:a
                if V(l1,l1) <= tol_2
                    V(l1,l1) = tol_2;
                end
            end
            transfer_each{k} = U*V*D';
        end
        
        % log-map layer
        for p = 1:num_layers_2
            log_map{p} = transfer_each{p};
            [u,v,w] = svd(log_map{p});
            logv = log(diag(v));
            log_map{p} = u * diag(logv) * w';
        end
        
        for n = 1:num_layers_2
            [U,V,D] = svd(log_map{n});
            all_trace(n) = trace(V);
        end
        for n = 1:num_layers_2
            T_2(n) = all_trace(n)/sum(all_trace);
        end
        for o = 1:num_layers_2
            temp2 = reshape(T_2(o)*log_map{o},size(log_map{1},1)*size(log_map{1},2),1);
            temp_final(:,o) = temp2;
        end
        
        final_test_branch(:,r) = temp_final(:);
    end
    
    test_matrix(:,l) = final_test_branch(:);
    
end

% building the test kernel matrix
kmatrix_test=zeros(size(Test_labels,2),size(Test_labels,2));

for i = 1:size(Train_labels,2)
    for j = 1:size(Test_labels,2)
        cov_i_Train = train_matrix(:,i);
        cov_j_Test = test_matrix(:,j);
        kmatrix_test(i,j) = exp((-norm(cov_i_Train-cov_j_Test)^2)/(2*deta^2)); 
    end
end

% KDA training
gnd = Train_labels(1,:); 
options.Regu = 1;
[eigvector, eigvalue] = fun_SymNet_v2_Train(options, gnd, kmatrix_train);
t_star_test=cputime;
nClasses=unique(gnd);
Dim = length(nClasses) - 1; 
kk = Dim;
alpha_matrix = eigvector(:,1:kk); 
data_DR_train = alpha_matrix'*kmatrix_train; 
data_DR_test = alpha_matrix'*kmatrix_test;

%% Classification
mdl = fitcknn(data_DR_train', Train_labels'); 
Class = predict(mdl,data_DR_test'); 
[a_test,b_test] = size(cov_test);
right_number = sum(Class'==Test_labels);
accuracy = right_number/b_test;

fprintf(1,'the right classified test samples are: %d\n',right_number);
fprintf(1,'the classification score is: %d\n',accuracy*100);


