close all; clear; clc

% % 
%   [unlabel,unlabel_data] = libsvmread('./semidata/mushrooms_5K_unlabel.0');
%   [label,label_data] = libsvmread('./semidata/mushrooms_5K_train.0');
% %  load ./semidata/a9a_5K_label_0.mat
% %  load ./semidata/a9a_5K_unlabel_0.mat
% L = length(label);
% U = length(unlabel);
% 
% traindata = [label_data;unlabel_data];
% trainlabel = [label;unlabel];
global KTYPE  added_num size_training
added_num=0;
size_training=2000;

k_fold = 3;
log_C_min = -3;
log_C_max = 7;
log_s_min = -7;
log_s_max = 3;
% log_gama_min = -3;
% log_gama_max = 3;
% v_min = 0;
% v_max = 0;

iters = (log_C_max - log_C_min +1) * (log_s_max - log_s_min + 1);% * (log_gama_max - log_gama_min +1)*(v_max-v_min +1);
train_err_arr = zeros(iters, k_fold);
test_err_arr = zeros(iters, k_fold);
SV_num=zeros(iters, k_fold);
parameter = zeros(iters, 2);
time_mat = zeros(iters, k_fold);

% L = length(trainlabel)-unlabel_size;
% U = unlabel_size;
balance_flag = 1;

% ratio = length(label)/(length(unlabel)+length(label));
data_flag=7;
[x,y]=read_data(data_flag);
[x,y,~,~]=random_select(x,y) ;
%traindata = full(traindata);
n = length(x);
rp = randperm(n);
% ruffle
x = x(rp, :);
y = y(rp, :);
size_test = floor(n / k_fold);



KTYPE=6;
%size_training=size_test;
iter = 1;
for i = log_C_min:log_C_max
    for j = log_s_min:log_s_max
        for k = 1:3
            d = (1:n)';          
            C = 2^i;
            KSCALE=2^j;    
            parameter(iter, 1) = C;
            parameter(iter, 2) = KSCALE;
            train=(k-1)*size_test+1:k*size_test;
            d(train)=[];
            original_x=x(d(1:end-added_num),:);
            original_y=y(d(1:end-added_num));
            added_x=x(d(end-added_num+1:end),:);
            added_y=y(d(end-added_num+1:end));
            test_x=x(train,:);
            test_y=y(train);
             original_x=sparse(original_x);
               test_x=sparse(test_x);
            %   [time,train_error_mat,test_error_mat,SV]=New_svm(original_x,original_y,added_x,added_y,test_x,test_y);
            model = svmtrain(original_y, original_x,[ '-c ',num2str(C),' -g ' ,num2str(KSCALE)]);
            [predict_label, accuracy, dec_values] = svmpredict(test_y, test_x, model);
            test_err_arr(iter, k) = accuracy(1,1)/100;
       %     train_err_arr(iter, k) = train_error_mat;
      %      time_mat(iter,k)=time;
        %    SV_num(iter,k)=SV;
         end
            iter = iter + 1;
    end
end

[bestValue,bestIdx] = max(mean(test_err_arr, 2));
fprintf('---bets C, KSCALE: %f, %f\n', parameter(bestIdx, 1), parameter(bestIdx, 2));



