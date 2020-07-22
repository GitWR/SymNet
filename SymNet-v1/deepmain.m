clear;
clc;
% author：RuiWang
% department: School of artificial intelligence and computer science,Jiangnan university
% version: 1.0

%% loda data

load FPHA_train_label
load FPHA_val_label
load FPHA_train_seq
load FPHA_val_seq

%% preparation
Train_labels = train_labels;
Test_labels = val_labels;

cov_sum = zeros(63,63);% 
cov_all = zeros(63,63);

num_layers = 3; % $m^{v1}$
p_dim = 20; % $d_m^{v1}$

second_layer_singletr = cell(1,num_layers); 
second_layer_singlete = cell(1,num_layers); 

final_train = cell(1,size(Train_labels,2));
final_test = cell(1,size(Test_labels,2));

used_to_pool = cell(1,600); 
pool_data = cell(1,10); 

%% two activation thresholds
eps_1 = 6e-5; % $\epsilon$
eta_1 = 1e-6; % $\eta$

%% training stage
Train_data = train_seq; 

cov_train = computeCov(Train_data); % the computed SPD matrices for the training image sets

% SPD matrix mapping layer

for k = 1:size(Train_labels,2)
    cov_sum = cov_sum + cov_train{k}; 
end

mean_train = cov_sum / k; 

for i = 1:size(Train_labels,2)
    cov_all = cov_all + (cov_train{i}-mean_train)' * (cov_train{i}-mean_train);
end

cov_all = cov_all/(size(Train_labels,2)-1);
[e_vectors,e_values] = eig(cov_all); 
[dummy,order] = sort(diag(-e_values)); 
e_vectors = e_vectors(:,order);

for i = 1:num_layers
    T_1{i} = e_vectors(:,(i-1)*p_dim+1:i*p_dim); % obtaining the connection weights
end

for i = 1:size(Train_labels,2)
    single_sample_tr = cov_train{i};
    for j = 1:num_layers
        second_layer_singletr{j} = T_1{j}'*single_sample_tr*T_1{j}; %(2D)^2PCA transformation
    end
    
    % Rectifying layer
    for k = 1:num_layers
        mid = second_layer_singletr{k};
        idx1 = mid <= 0;
        idx2 = mid > -eta_1;
        idx = idx1 & idx2;
        mid(idx) = -eta_1;
        [U,V,D] = svd(mid);
        [a,b] = size(V);
        tol = trace(V) * eps_1;
        for l = 1:a
            if V(l,l) <= tol
                V(l,l) = tol;
            end
        end
        second_layer_singletr{k} = U * V * D';
    end
    
    % SPD matrix pooling layer: tangent space pooling tactic
    for m = 1:num_layers
        temp = second_layer_singletr{m};
        [u,v,w] = svd(temp);
        logv = log(diag(v));
        temp = u * diag(logv) * w';
        temp_patch = mat2cell(temp,[2 2 2 2 2 2 2 2 2 2],[2 2 2 2 2 2 2 2 2 2]);%分层10个2*2的小矩阵
        pooling = cellfun(@mean,cellfun(@mean,temp_patch,'UniformOutput',false));
        [u,v,w] = svd(pooling);
        logv = exp(diag(v));
        mean_pool{m} = u * diag(logv) * w';
    end
    
    % log-map layer
    for p=1:num_layers
        log_map{p} = mean_pool{p}; 
        [u,v,w] = svd(log_map{p});
        logv = log(diag(v));
        log_map{p} = u * diag(logv) * w';
    end
    
    % output layer
    for n=1:num_layers
        [U,V,D] = svd(log_map{n});
        all_trace(n) = trace(V);
    end
    for n = 1:num_layers
        T_2(n) = all_trace(n)/sum(all_trace);
    end
    
    for o=1:num_layers
        temp = reshape(T_2(o)*log_map{o},size(log_map{1},1)*size(log_map{1},2),1);
        temp_final(:,o)=temp;
    end
    final_train{i}=temp_final;
end

% constructing the training kernel matrix
kmatrix_train=zeros(size(final_train,2),size(final_train,2)); % training kernel matrix

deta = 1.30;

for i = 1:size(final_train,2)
    for j = 1:size(final_train,2)
        cov_i_Train = final_train{i};
        cov_j_Train = final_train{j};
        cov_i_Train_reshape = reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);
        cov_j_Train_reshape = reshape(cov_j_Train,size(cov_j_Train,1)*size(cov_j_Train,2),1);
        kmatrix_train(i,j) = exp((-norm(cov_i_Train_reshape-cov_j_Train_reshape)^2)/(2*deta^2));
        kmatrix_train(j,i) = kmatrix_train(i,j);
    end
end

%% test stage
Test_data = val_seq;
cov_test = computeCov(Test_data); % the computed SPD matrices for the test image sets

for i = 1:size(Test_labels,2)
    single_sample_te = cov_test{i};
    for j = 1:num_layers
        second_layer_singlete{j} = T_1{j}'*single_sample_te*T_1{j}; % (2D)^2PCA transformation
    end
    
    % Rectifying layer
    for k = 1:num_layers
        mid = second_layer_singlete{k};
        idx1 = mid <= 0;
        idx2 = mid > -eta_1;
        idx = idx1 & idx2;
        mid(idx) = -eta_1;
        [U,V,D] = svd(mid);
        [a,b] = size(V);
        tol = trace(V) * eps_1;  
        for l=1:a
            if V(l,l) <= tol
                V(l,l) = tol;
            end
        end
        second_layer_singlete{k} = U * V * D';
    end
    
    % SPD matrix pooling layer: tangent space pooling tactic
    for m=1:num_layers
        temp = second_layer_singlete{m};
        [u,v,w] = svd(temp);
        logv = log(diag(v));
        temp = u * diag(logv) * w';
        temp_patch = mat2cell(temp,[2 2 2 2 2 2 2 2 2 2],[2 2 2 2 2 2 2 2 2 2]);
        pooling = cellfun(@mean,cellfun(@mean,temp_patch,'UniformOutput',false));
        [u,v,w] = svd(pooling);
        logv = exp(diag(v));
        mean_pool{m} = u * diag(logv) * w';
    end
    
    % log-map layer
    for p = 1:num_layers
        log_map{p} = mean_pool{p}; 
        [u,v,w] = svd(log_map{p});
        logv = log(diag(v));
        log_map{p} = u * diag(logv) * w';
    end
    
    % output layer
    for n = 1:num_layers
        [U,V,D] = svd(log_map{n});
        all_trace(n) = trace(V);
    end
    for n = 1:num_layers
        T_2(n) = all_trace(n)/sum(all_trace);
    end
    
    for o = 1:num_layers
        temp = reshape(T_2(o)*log_map{o},size(log_map{1},1)*size(log_map{1},2),1);
        temp_final(:,o) = temp;
    end
    final_test{i}=temp_final;
end

% building the test kernel matrix
kmatrix_test=zeros(size(final_train,2),size(final_test,2)); % test kernel matrix

for i = 1:size(final_train,2)
    for j = 1:size(final_test,2)
        cov_i_Train = final_train{i};
        cov_j_Test = final_test{j};
        cov_i_Train_reshape = reshape(cov_i_Train,size(cov_i_Train,1)*size(cov_i_Train,2),1);
        cov_j_Test_reshape = reshape(cov_j_Test,size(cov_j_Test,1)*size(cov_j_Test,2),1);
        kmatrix_test(i,j) = exp((-norm(cov_i_Train_reshape-cov_j_Test_reshape)^2)/(2*deta^2));
    end
end

%% KDA 
gnd = Train_labels(1,:); 
options.Regu = 1;
[eigvector, eigvalue] = fun_SymNet_Train(options, gnd, kmatrix_train);
nClasses = unique(gnd);
Dim = length(nClasses) - 1; 
kk = Dim;
alpha_matrix = eigvector(:,1:kk); 
data_DR_train = alpha_matrix'*kmatrix_train; 
data_DR_test = alpha_matrix'*kmatrix_test;

%% Classification
mdl = fitcknn(data_DR_train', Train_labels'); 
Class = predict(mdl,data_DR_test');
[a_test,b_test] = size(cov_test);
accuracy_number = sum(Class'==Test_labels); % the right recognised samples
accuracy = accuracy_number/b_test;

fprintf(1,'the right classified test samples are：%d\n',accuracy_number);
fprintf(1,'the classification accuracy is: %d\n',accuracy * 100);

