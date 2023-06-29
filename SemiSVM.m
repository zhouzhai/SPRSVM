classdef SemiSVM
    %SMO Summary of this class goes here
    %Detailed explanation goes here

    properties
        alpha=[];   %dual variables
        b=0;        %offset
        pms=[];       %five parameters C_low,C_up
        active_size=0;
        iteration=0;
        x=[];
        y=[];
        KKT=0;
        local_minimal=0;
        index_M=[];
        index_E=[];
        index_O=[];
        steps=0;
        change_mu=0;
        singular=0;
    end
    methods(Static = true)

        function obj=SemiSVM(Data,Label,initLowIn,initUbIn,alpha0)
            global  x y alpha_home sample_length p_term fake_zero
            x=Data;
            y=Label;
            p_term=y;
            sample_length=length(y);
            if  alpha0==0
            	alpha0=zeros(sample_length,1);
            end
            [b,active_size,iteration]=SemiSVM.quadsmo(alpha0,initLowIn,initUbIn);
            obj.pms=[initLowIn initUbIn];
            obj.alpha=alpha_home;
            obj.b=b;
            obj.active_size=active_size;
            obj.iteration=iteration;
            obj.x=x;
            obj.y=y;
            obj.index_M=find((alpha_home>initLowIn+fake_zero) & (alpha_home<initUbIn-fake_zero)); 
            obj.index_E=find((alpha_home>=initUbIn-fake_zero));
            obj.index_O=find((alpha_home<=initLowIn+fake_zero));
            outflag = SemiSVM.TestKKT(obj,initLowIn,initUbIn);
            obj.KKT = outflag;
        end
        
        function k = Kernel(x, y)%the kernel function
            % function k = kernel(x, y);
            %
            %	x: (Lx,N) with Lx: number of points; N: dimension
            %	y: (Ly,N) with Ly: number of points
            %	k: (Lx,Ly)
            %
            %	KTYPE = 1:      linear kernel:      x*y'
            %	KTYPE = 2,3,4:  polynomial kernel:  (x*y'*KSCALE+1)^KTYPE
            %	KTYPE = 5:      sigmoidal kernel:   tanh(x*y'*KSCALE)
            %	KTYPE = 6:	gaussian kernel with variance 1/(2*KSCALE)
            %
            %       assumes that x and y are in the range [-1:+1]/KSCALE (for KTYPE<6)

            global KTYPE
            global KSCALE

            k = x*y';
            if KTYPE == 1				% linear
                % take as is
            elseif KTYPE <= 4			% polynomial
                k = (k*KSCALE+1).^KTYPE;
            elseif KTYPE == 5			% sigmoidal
                k = tanh(k*KSCALE);
            elseif KTYPE == 6			% gaussian
                [Lx,~] = size(x);       % the number of x rows
                [Ly,~] = size(y);
                k = 2*k;
                k = k-sum(x.^2,2)*ones(1,Ly);   %sum(A,2) means compute the sum of the elements in each row
                k = k-ones(Lx,1)*sum(y.^2,2)';
                k = exp(k*KSCALE);
            end
        end
        
        function [b,a_size,index_iteration] = quadsmo(alpha0,LowIn,UbIn)
            global fake_zero x y max_iteration_smo epsilon sample_length index_home shrink_state...
                alpha_home flag_up flag_low cache_size  regrad_count cache_memory p_term initUbIn initLowIn
            initUbIn =UbIn;
            initLowIn=LowIn;
            max_iteration_smo=10000000;
            fake_zero=10^-12;
            epsilon=10^-10;
            cache_memory=40;%MB
            sample_length=length(y);%n
            counter=min(sample_length,1000)+1;
            index_home=(1:sample_length);


            alpha_home=alpha0;%n*1  initialize the alpha
%%%%%%~~~~~~~~~~~~~I_up & I_low & 判别函数 & gi_bar
            flag_up=(alpha0<initUbIn -fake_zero);%I_up
            flag_low=(alpha0>initLowIn+fake_zero);%I_low
            %初始化判别函数 -grad(f)_i
            alpha0_nonZero_index=find(~(alpha0==0));
            Q_LnonZero=SemiSVM.Kernel(x,x(alpha0_nonZero_index,:));
            Q_alpha0=Q_LnonZero*alpha0(alpha0_nonZero_index);
            gi=p_term-Q_alpha0;
            % gi_bar指的是边界alpha
            length_nonZero=length(alpha0_nonZero_index);
            Q_alpha0_nonZero_index=1:length_nonZero;
            alpha0_bound_flag=(abs(alpha0(alpha0_nonZero_index)-initUbIn(alpha0_nonZero_index))<fake_zero)|(abs(alpha0(alpha0_nonZero_index)-initLowIn(alpha0_nonZero_index))<fake_zero);
            Q_alpha0_bound_index=Q_alpha0_nonZero_index(alpha0_bound_flag);
            gi_bar=Q_LnonZero(:,Q_alpha0_bound_index)*alpha0(alpha0_nonZero_index(alpha0_bound_flag));
            index_iteration=1;
            b=0;
            stopnum=0;
            Q_Li=zeros(sample_length,1);
            regrad_count=0;
            used_cache=0;
            shrink_state=0;

            %初始化活动集的大小和索引
            %%%%%%~~~~~~~~~~~~~~~initLowIn
            [leave_index,active_index,active_size,local_gi]=SemiSVM.DoShrinking(gi,gi_bar,flag_up,flag_low,alpha0,initLowIn,initUbIn,index_home);
            shrink_count=1;
            a_size(shrink_count)=active_size;
            other_active_index=(1:sample_length);
            other_active_index(active_index)=[];
            %%更新当前活动集
            local_flag_up=flag_up(active_index);
            local_flag_low=flag_low(active_index);
            %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~local_initLowIn
            local_initLowIn=initLowIn(active_index);
            local_initUbIn=initUbIn(active_index);
            
            %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            local_x=x(active_index,:);
            %local_y=y(active_index);
            local_alpha=alpha_home(active_index);
            %%%%%%更新local_gi
            local_gi=local_gi(leave_index);
            %%%%%%%%%%%%%%初始化缓存
            cache_size=floor(2^17*cache_memory/active_size);
            Cache_ind=repmat(int32(0),cache_size,1);
            Cache_iter=repmat(int32(0),cache_size,1);
            main_problem_fail=0;
            while index_iteration < max_iteration_smo
                counter=counter-1;
                if(counter==0)      %%%%%缩减
                    counter=min(sample_length,1000)+1;
                    if main_problem_fail==1
                        main_problem_fail=0;
                        flag_up(active_index)=local_flag_up;
                        flag_low(active_index)=local_flag_low;
                        alpha_home(active_index)=local_alpha;
                        [leave_index,active_index,active_size,local_gi]=SemiSVM.DoShrinking(gi_home,gi_bar,flag_up,flag_low,alpha_home,initLowIn,initUbIn,index_home);
                        shrink_count=shrink_count+1;
                        a_size(shrink_count)=active_size;
                        other_active_index=(1:sample_length);
                        other_active_index(active_index)=[];

                        clear Cache;
                        clear Cache_ind;
                        clear Cache_iter;
                        cache_size=floor(2^17*cache_memory/active_size);
                        Cache_ind=repmat(int32(0),cache_size,1);
                        Cache_iter=repmat(int32(0),cache_size,1);
                        used_cache=0;
                    else
                        %%%保存之前的结果
                        alpha_home(active_index)=local_alpha;
                        flag_up(active_index)=local_flag_up;
                        flag_low(active_index)=local_flag_low;
                        [leave_index,active_index,active_size,local_gi]=SemiSVM.DoShrinking(local_gi,gi_bar,local_flag_up,local_flag_low,local_alpha,local_initLowIn,local_initUbIn,active_index);
                        shrink_count=shrink_count+1;
                        a_size(shrink_count)=active_size;
                        other_active_index=(1:sample_length);
                        other_active_index(active_index)=[];

                        if length(leave_index)==sample_length
                            clear Cache;
                            clear Cache_ind;
                            clear Cache_iter;
                            cache_size=floor(2^17*cache_memory/active_size);
                            Cache_ind=repmat(int32(0),cache_size,1);
                            Cache_iter=repmat(int32(0),cache_size,1);
                            used_cache=0;
                        else
                            for i=1:used_cache
                                Cache{i}=Cache{i}(leave_index);
                            end
                        end
                    end
                    %%更新当前活动集
                    local_flag_up=flag_up(active_index);
                    local_flag_low=flag_low(active_index);
                    %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~local_initLowIn
                    local_initLowIn=initLowIn(active_index);
                    local_initUbIn=initUbIn(active_index);
                    %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    local_x=x(active_index,:);
%                     local_y=y(active_index);
                    local_alpha=alpha_home(active_index);
                    %%%%%%更新local_gi
                    local_gi=local_gi(leave_index);
                end    %%%%%%%%缩减完成
                %%%选择两个变量
                max_value1=-inf;
                min_value1=inf;
                index=(1:active_size);
                 if any(local_flag_up)
                    index_up=index(local_flag_up);
                    [max_value1,max_index]=max(local_gi(local_flag_up));%%  i
                    local_index(1)=index_up(max_index);
                 end
                 if any(local_flag_low)
                    index_low=index(local_flag_low);
                    [min_value1,min_index]=min(local_gi(local_flag_low));
                    local_index(2)=index_low(min_index);
                 end

                %%%%%%判断
                if max_value1-min_value1<=epsilon   %stopping condition (subproblem)
                    stopnum=stopnum+1;
                    fprintf('sub_mM:%e\n ',max_value1-min_value1);
                    alpha_home(active_index)=local_alpha;
                    flag_up(active_index)=local_flag_up;
                    flag_low(active_index)=local_flag_low;
                    if active_size==sample_length;
                        gi_home=local_gi;
                    else
                        [gi_home]=SemiSVM.Reconstruct_gradient(local_gi,gi_bar,active_index);
                    end
                 max_value2=-inf;
                 min_value2=inf;
                 if any(flag_up)
                [max_value2]=max(gi_home(flag_up));
                 end
                 if any(flag_low)
                [min_value2]=min(gi_home(flag_low));
                 end
%                     [max_value2]=max(gi_home(flag_up));
%                     [min_value2]=min(gi_home(flag_low));

                    if (max_value2-min_value2<=epsilon)
                        b=(max_value2+min_value2)/2;
                        fprintf('main_mM:%e\n ',max_value2-min_value2);
                        break;
                    else
                        main_problem_fail=1;
                        counter=1;
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%%
                %%更新两个alpha
                %%%查找缓存块
                cache_index1=find(Cache_ind==active_index(local_index(1)),1);
                cache_index2=find(Cache_ind==active_index(local_index(2)),1);
%                 fprintf('%d',index_iteration);
                if isempty(cache_index1) && isempty(cache_index2)
                    if used_cache==cache_size
                        initQAB=SemiSVM.Kernel(local_x,(local_x(local_index,:)));%n*2
                        [~,replace_index]=sort(Cache_iter);
                        Cache{replace_index(1)}=initQAB(:,1);
                        Cache{replace_index(2)}=initQAB(:,2);
                        Cache_ind(replace_index([1,2]))=active_index(local_index);
                        Cache_iter(replace_index([1,2]))=index_iteration;
                    elseif used_cache==cache_size-1
                        initQAB=SemiSVM.Kernel(local_x,(local_x(local_index(1),:)));
                        Cache{used_cache+1}=initQAB(:,1);
                        Cache_ind(used_cache+1)=active_index(local_index(1));
                        Cache_iter(used_cache+1)=index_iteration;
                        used_cache=used_cache+1;
                        initQAB(:,2)=SemiSVM.Kernel(local_x,(local_x(local_index(2),:)));
                        [~,replace_index]=min(Cache_iter);
                        Cache{replace_index}=initQAB(:,2);
                        Cache_ind(replace_index)=active_index(local_index(2));
                        Cache_iter(replace_index)=index_iteration;
                    else
                        initQAB=SemiSVM.Kernel(local_x,(local_x(local_index,:)));%n*2

                        Cache{used_cache+1}=initQAB(:,1);
                        Cache{used_cache+2}=initQAB(:,2);
                        Cache_ind([used_cache+1,used_cache+2])=active_index(local_index);
                        Cache_iter([used_cache+1,used_cache+2])=index_iteration;
                        used_cache=used_cache+2;
                    end
                else
                    if cache_index1
                        initQAB=Cache{cache_index1};
                        Cache_iter(cache_index1)=index_iteration;
                    else
                        if used_cache==cache_size
                            initQAB=SemiSVM.Kernel(local_x,(local_x(local_index(1),:)));
                            [~,replace_index]=min(Cache_iter);
                            if replace_index==cache_index2
                                [~,replace_index]=sort(Cache_iter);
                                Cache{replace_index(2)}=initQAB(:,1);
                                Cache_ind(replace_index(2))=active_index(local_index(1));
                                Cache_iter(replace_index(2))=index_iteration;

                            else
                                Cache{replace_index}=initQAB(:,1);
                                Cache_ind(replace_index)=active_index(local_index(1));
                                Cache_iter(replace_index)=index_iteration;
                            end
                        else
                            initQAB=SemiSVM.Kernel(local_x,(local_x(local_index(1),:)));
                            Cache{used_cache+1}=initQAB(:,1);
                            Cache_ind(used_cache+1)=active_index(local_index(1));
                            Cache_iter(used_cache+1)=index_iteration;
                            used_cache=used_cache+1;

                        end
                    end
                    if cache_index2
                        initQAB(:,2)=Cache{cache_index2};
                        Cache_iter(cache_index2)=index_iteration;
                    else
                        if used_cache==cache_size
                            initQAB(:,2)=SemiSVM.Kernel(local_x,(local_x(local_index(2),:)));
                            [~,replace_index]=min(Cache_iter);
                            Cache{replace_index}=initQAB(:,2);
                            Cache_ind(replace_index)=active_index(local_index(2));
                            Cache_iter(replace_index)=index_iteration;
                        else
                            initQAB(:,2)=SemiSVM.Kernel(local_x,(local_x(local_index(2),:)));
                            Cache{used_cache+1}=initQAB(:,2);
                            Cache_ind(used_cache+1)=active_index(local_index(2));
                            Cache_iter(used_cache+1)=index_iteration;
                            used_cache=used_cache+1;
                        end
                    end
                end

                initQBB=initQAB(local_index,:);
                %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~initf=grad(alpha)-QBB*alphaB
                %%%%%% &  sum(alpha1+alpha2)
                initf=-local_gi(local_index)-initQBB*local_alpha(local_index);
                sum_two_alpha=sum(local_alpha(local_index)); %alpha1+alpha2=zeta(constant)
            %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                old_alpha=local_alpha(local_index);
%                  %能量函数
%                 fval0=0.5*local_alpha(local_index)'*initQBB*local_alpha(local_index)+initf'*local_alpha(local_index);%the old value of two original alpha

                [initAlpha] = SemiSVM.OneSmo(initQBB,initf,sum_two_alpha,local_index,local_initLowIn,local_initUbIn);
                new_alpha=initAlpha;
                local_alpha(local_index)=initAlpha; %update the two new alpha


                    local_flag_up(local_index)=initAlpha<local_initUbIn(local_index) -fake_zero;
                    local_flag_low(local_index)=initAlpha>local_initLowIn(local_index)+fake_zero;
                local_gi=local_gi-initQAB*(new_alpha-old_alpha);
                %%%%%%更新gi_bar
                    uij=1*(abs(old_alpha-local_initUbIn(local_index))<fake_zero)+2*(abs(old_alpha-local_initLowIn(local_index))<fake_zero);
                    update_uij=1*(abs(new_alpha-local_initUbIn(local_index))<fake_zero)+2*(abs(new_alpha-local_initLowIn(local_index))<fake_zero);

                if active_size==sample_length
                    if ~(uij(1)==update_uij(1))
                        Q_Li=initQAB(:,1);
                        if uij(1)*update_uij(1)==2 %%%up和low互换
                         gi_bar=gi_bar+(new_alpha(1)-old_alpha(1))*Q_Li;
                        elseif (uij(1)-update_uij(1))>0 %%%bound 转化为 自由变量
                             gi_bar=gi_bar-old_alpha(1)*Q_Li;
                        else  %%% 自由变量 转化为 bound
                            gi_bar=gi_bar+new_alpha(1)*Q_Li;
                        end
                    end
                    if ~(uij(2)==update_uij(2))
                        Q_Li=initQAB(:,2);
                        if uij(2)*update_uij(2)==2 %%%up和low互换
                            gi_bar=gi_bar+(new_alpha(2)-old_alpha(2))*Q_Li;
                        elseif (uij(2)-update_uij(2))>0 %%%bound 转化为 自由变量
                            gi_bar=gi_bar-old_alpha(2)*Q_Li;
                        else  %%% 自由变量 转化为 bound
                            gi_bar=gi_bar+new_alpha(2)*Q_Li;
                        end
                    end
                else
                    if ~(uij(1)==update_uij(1))
                        Q_Li(active_index,:)=initQAB(:,1);
                        Q_Li(other_active_index,:)=SemiSVM.Kernel(x(other_active_index,:),(x(active_index(local_index(1)),:)));
                        if uij(1)*update_uij(1)==2 %%%up和low互换
                         gi_bar=gi_bar+(new_alpha(1)-old_alpha(1))*Q_Li;
                        elseif (uij(1)-update_uij(1))>0 %%%bound 转化为 自由变量
                             gi_bar=gi_bar-old_alpha(1)*Q_Li;
                        else  %%% 自由变量 转化为 bound
                            gi_bar=gi_bar+new_alpha(1)*Q_Li;
                        end
                    end
                    if ~(uij(2)==update_uij(2))
                        Q_Li(active_index,:)=initQAB(:,2);
                        Q_Li(other_active_index,:)=SemiSVM.Kernel(x(other_active_index,:),(x(active_index(local_index(2)),:)));
                        if uij(2)*update_uij(2)==2 %%%up和low互换
                            gi_bar=gi_bar+(new_alpha(2)-old_alpha(2))*Q_Li;
                        elseif (uij(2)-update_uij(2))>0 %%%bound 转化为 自由变量
                            gi_bar=gi_bar-old_alpha(2)*Q_Li;
                        else  %%% 自由变量 转化为 bound
                            gi_bar=gi_bar+new_alpha(2)*Q_Li;
                        end
                    end
                end
                index_iteration = index_iteration+1;
            end

            %fprintf('%d\n',index_iteration);
            if index_iteration >= max_iteration_smo
                fprintf('index_iteration >= max_iteration_smo\n');
            end

        end
        
        function [gi]=Reconstruct_gradient(local_gi,gi_bar,active_index)
            global x  p_term initLowIn initUbIn fake_zero sample_length alpha_home regrad_count;
            other_active_index=(1:sample_length);
            other_active_index(active_index)=[];
            gi=zeros(sample_length,1);
            gi(active_index,:)=-local_gi;
            gi(other_active_index,:)=gi_bar(other_active_index)-p_term(other_active_index);  %初始化非活动集的gi
            free_flag=abs(alpha_home(active_index)-initUbIn(active_index))>fake_zero & abs(alpha_home(active_index)-initLowIn(active_index))>fake_zero;
            free_index=active_index(free_flag);
            reSize=100000;
            if length(other_active_index)>reSize
                reGroup=length(other_active_index)/reSize;
                for i=1:reGroup
                    QPiAF=SemiSVM.Kernel(x(other_active_index(((i-1)*reSize+1):i*reSize),:),x(free_index,:));
                    gi(other_active_index(((i-1)*reSize+1):i*reSize))=gi(other_active_index(((i-1)*reSize+1):i*reSize))+QPiAF*alpha_home(free_index);
                end
                if i*reSize<length(other_active_index)
                    QPiAF=SemiSVM.Kernel(x(other_active_index(i*reSize+1:end),:),x(free_index,:));
                    gi(other_active_index(i*reSize+1:end))=gi(other_active_index(i*reSize+1:end))+QPiAF*alpha_home(free_index);
                end
            else
                %%循环很耗时，但是不循环可能矩阵太大，怎么解决？
                QNAf=SemiSVM.Kernel(x(other_active_index,:),(x(free_index,:)));
                gi(other_active_index)=gi(other_active_index)+QNAf*alpha_home(active_index(free_flag));
            end
            gi=-gi;
            regrad_count=regrad_count+1;
        end
        
        function [leave_index,active_index,active_size,local_gi]=DoShrinking(local_gi,gi_bar,local_flag_up,local_flag_low,local_alpha,local_initLowIn,local_initUbIn,active_index)
            %%%%%%~~~~~~~ initLowIn
            global fake_zero shrink_state epsilon initLowIn initUbIn alpha_home flag_up flag_low index_home
            %%% 最大违反对
            max_value=-inf;
            min_value=inf;
            if any(local_flag_up)
            [max_value]=max(local_gi(local_flag_up));
            end
            if any(local_flag_low)
            [min_value]=min(local_gi(local_flag_low));
            end
            a_index=active_index;
            if(shrink_state==0) && (max_value-min_value)<=10*epsilon
                shrink_state=1;
                %%%%
                [gi_home]=SemiSVM.Reconstruct_gradient(local_gi,gi_bar,active_index);
                local_gi=gi_home;
                %%% 最大违反对 这里需要重新求吗？
                 max_value=-inf;
                 min_value=inf;
                 if any(flag_up)
                [max_value]=max(gi_home(flag_up));
                 end
                 if any(flag_low)
                [min_value]=min(gi_home(flag_low));
                 end
                  %%%%%%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    initLowIn
                flag_up_bound=alpha_home<initLowIn+fake_zero;
                flag_low_bound=alpha_home>initUbIn-fake_zero;
                leave_index=(gi_home>max_value+fake_zero & flag_low_bound)...
                    |(gi_home<min_value-fake_zero & flag_up_bound);
                leave_index=~leave_index;
                active_index=index_home(leave_index);
                active_size=length(active_index);

                if active_size==0
                    active_size=length(leave_index);
                    leave_index=~leave_index;
                    active_index=index_home;
                end
            else
                flag_up_bound=local_alpha<local_initLowIn+fake_zero;
                flag_low_bound=local_alpha>local_initUbIn-fake_zero;
                leave_index=(local_gi>max_value+fake_zero & flag_low_bound)...
                    |(local_gi<min_value-fake_zero & flag_up_bound);
                leave_index=~leave_index;
                active_index=active_index(leave_index);
                active_size=length(active_index);

                if active_size==0
                    active_size=length(leave_index);
                    leave_index=~leave_index;
                    active_index=a_index;
                end
            end
        end
        
        function [initAlpha] = OneSmo(initQ,initf,sum_two_alpha,two_index,lb,ub) %update the two alpha
            %             global y
            first_index=two_index(1);
            second_index=two_index(2);
            a=(initQ(1,1)+initQ(2,2)-2*initQ(1,2))/2;
            b=-sum_two_alpha*initQ(2,2)+initQ(1,2)*sum_two_alpha+initf'*[1;-1];
            left_bound=max(lb(first_index),sum_two_alpha-ub(second_index));     %L
            right_bound=min(ub(first_index),sum_two_alpha-lb(second_index));      %H
            opitimal_alpha1=-b/(2*a);
            if opitimal_alpha1 > right_bound
                opitimal_alpha1=right_bound;
            end
            if opitimal_alpha1 < left_bound
                opitimal_alpha1=left_bound;
            end
            initAlpha=[opitimal_alpha1; sum_two_alpha-opitimal_alpha1];
        end

        function out_KKT=TestKKT(os,C_low,C_up)
            global fake_zero2
            out_KKT=0;
            local_g=IncSemiSVM.GetG(os.x,os.y,os);
%             local_g=SemiSVM.CaculateF(os);
            alpha=os.alpha;
            zero=sum(alpha);
            if zero<fake_zero2 && zero>-fake_zero2
                index=[1:length(alpha)];
                local_tmp=SemiSVM.SubKKT(alpha,local_g,C_low,C_up,index);
                if local_tmp==1
                    out_KKT=1;
                end
            end
        end
        function out_KKT=SubKKT(alpha,local_g,C_low,C_up,index)
            global fake_zero2
            out_KKT=1;
            sum_length=length(alpha);
            for i=1:sum_length
                if alpha(i)<C_low(i)+fake_zero2
                    if local_g(i)<-fake_zero2
                        out_KKT=0;
                        break;
                    end
                else
                    if alpha(i)>C_up(i)-fake_zero2
                        if local_g(i)>fake_zero2
                            out_KKT=0;
                            break;
                        end
                    else
                        if local_g(i)>fake_zero2 || local_g(i)<-fake_zero2
                            out_KKT=0;
                            break;
                        end
                    end
                end
            end
        end
        
        function terminal_flag=CheckTerminial(PreLowBoud,PreUppBound,NowLowBoud,NowUppBound)
            terminal_flag=0;
            tmp=sum(abs(PreLowBoud-NowLowBoud))+sum(abs(PreUppBound-NowUppBound));
            if tmp==0
                terminal_flag=1;
            end
        end
        function [LowBoud,UppBound]=ComputerCCstar(original_x,original_y,objcs)
            global C 
            Q=SMO.Kernel(original_x,objcs.x).*(original_y*ones(1,length(objcs.y)));
            local_Q=[original_y Q];
            local_alpha=[objcs.b;objcs.alpha];
            local_f=local_Q*local_alpha;
            mu=C*(local_f<0);
            LowBoud=(-mu).*(original_y>0)+(mu-C).*(original_y<0);
            UppBound=(C-mu).*(original_y>0)+(mu).*(original_y<0);
        end
        function local_minimal = TestMinimal(os)
            [NowLowBound,NowUppBound]=SemiSVM.ComputerCCstar(os.x,os.y,os);
            local_minimal=SemiSVM.CheckTerminial(os.pms(:,1),os.pms(:,2),NowLowBound,NowUppBound);
        end
    end
end