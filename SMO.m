classdef SMO
    %SMO Summary of this class goes here
    %Detailed explanation goes here
    
    properties
        alpha=[];   %dual variables
        b=0;        %offset
        pms=0;       %five parameters C,C+,C-,eps,svm_type
        active_size=0;
        iteration=0;
        x=[];    % input of samples
        y=[];    % output of samples
        index_M=[];
        index_E=[];
        index_O=[];
    end
    methods(Static = true)
        function obj=SMO(Data,Label,C,C_plus,C_minus,eps,svm_type,ktype,kscale,alpha0)
            global  x y KTYPE KSCALE initUbIn alpha_home sample_length p_term fake_zero      
            KTYPE = ktype;
            KSCALE =kscale;
            if svm_type==1      %C_SVC
                sample_length=length(Label);
                x=Data;
                y=Label;
                p_term=1;
                initUbIn=C;
                [b,active_size,iteration]=SMO.quadsmo(alpha0,svm_type);
                obj.pms=C;
            elseif svm_type==2  %e_SVR
                x=[Data;Data];
                z=Label;
                sample_length=length(z);
                y=[ones(sample_length,1);-ones(sample_length,1)];
                p_term=-[(eps+z);(eps-z)];
                initUbIn=C;
                [b,active_size,iteration]=SMO.quadsmo(alpha0,svm_type);
                obj.pms(1)=C;
                obj.pms(2)=eps;
            elseif svm_type==3  %CS_SVC
                sample_length=length(Label);
                x=Data;
                y=Label;
                p_term=1*(y==1)+(1/(2*C_minus-1))*(y==-1);
                initUbIn=C*C_plus*(y==1)+C*(2*C_minus-1)*(y==-1);  %set the upper bound
                [b,active_size,iteration]=SMO.quadsmo(alpha0,svm_type);
                obj.pms(1)=C;
                obj.pms(2)=C_plus;
                obj.pms(3)=C_minus;
            end
            obj.index_M=find((alpha_home>0+fake_zero) & (alpha_home<C-fake_zero)); 
            obj.index_E=find((alpha_home>=C-fake_zero));
            obj.index_O=find((alpha_home<=0+fake_zero));
            obj.alpha=alpha_home;
            obj.b=b;
            obj.active_size=active_size;
            obj.iteration=iteration;
        end
        function [out]=InitialSolution(x,y,C)
            global fake_zero  KTYPE KSCALE
            alpha0=zeros(length(y),1);
            out=SMO(x,y,C,0,0,fake_zero,1,KTYPE,KSCALE,alpha0);
            out.x=x;
            out.y=y;
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
        function [b,a_size,index_iteration] = quadsmo(alpha0,svm_type)
            global fake_zero x y max_iteration_smo epsilon sample_length index_home shrink_state...
                initUbIn alpha_home flag_up flag_low cache_size  regrad_count cache_memory p_term  fake_zero2
            max_iteration_smo=1000000;
            fake_zero=10^-10;
            fake_zero2=10^-8;
            epsilon=10^-10;
            sample_length=length(y);%n
            cache_memory=40;%MB
            counter=min(sample_length,1000)+1;
            index_home=(1:sample_length);
            alpha_home=alpha0;%n*1  initialize the alpha
            flag_up=((alpha0>fake_zero) & y==-1) | ((alpha0<initUbIn -fake_zero) & y==1);%I_up
            flag_low=((alpha0>fake_zero) & y==1) | ((alpha0<initUbIn -fake_zero) & y==-1);%I_low
            %初始化判别函数 -y_i*grad(f)_i
            alpha0_nonZero_index=find(~(alpha0==0));
            Q_LnonZero=SMO.Kernel(x,x(alpha0_nonZero_index,:)).*(y*y(alpha0_nonZero_index)');
            Q_alpha0=Q_LnonZero*alpha0(alpha0_nonZero_index);
            gi=(p_term-Q_alpha0).*y;
            length_nonZero=length(alpha0_nonZero_index);
            Q_alpha0_nonZero_index=1:length_nonZero;
            alpha0_bound_flag=(abs(alpha0(alpha0_nonZero_index)-initUbIn(alpha0_nonZero_index))<fake_zero);
            Q_alpha0_bound_index=Q_alpha0_nonZero_index(alpha0_bound_flag);
            gi_bar=Q_LnonZero(:,Q_alpha0_bound_index)*alpha0(alpha0_nonZero_index(alpha0_bound_flag));
            
            index_iteration=1;
            b=0;
            stopnum=0;
            Q_Li=zeros(sample_length,1);
            regrad_count=0;

            used_cache=0;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            shrink_state=0;

            %初始化活动集的大小和索引
            [leave_index,active_index,active_size,local_gi]=SMO.DoShrinking(gi,gi_bar,flag_up,flag_low,alpha0,initUbIn,y,index_home);
            shrink_count=1;
            a_size(shrink_count)=active_size;
            other_active_index=(1:sample_length);
            other_active_index(active_index)=[];
            %%更新当前活动集
            local_flag_up=flag_up(active_index);
            local_flag_low=flag_low(active_index);
            if svm_type==3
                local_initUbIn=initUbIn(active_index);
            else
                local_initUbIn=initUbIn;
            end
            local_x=x(active_index,:);
            local_y=y(active_index);
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
                if(counter==0)   
                    counter=min(sample_length,1000)+1;
                    if main_problem_fail==1
                        main_problem_fail=0;
                        flag_up(active_index)=local_flag_up;
                        flag_low(active_index)=local_flag_low;
                        alpha_home(active_index)=local_alpha;
                        [leave_index,active_index,active_size,local_gi]=SMO.DoShrinking(gi_home,gi_bar,flag_up,flag_low,alpha_home,initUbIn,y,index_home,svm_type);
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
                        alpha_home(active_index)=local_alpha;
                        flag_up(active_index)=local_flag_up;
                        flag_low(active_index)=local_flag_low;
                        [leave_index,active_index,active_size,local_gi]=SMO.DoShrinking(local_gi,gi_bar,local_flag_up,local_flag_low,local_alpha,local_initUbIn,local_y,active_index,svm_type);
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
                    if svm_type==3
                    local_initUbIn=initUbIn(active_index);
                    end
                    local_x=x(active_index,:);
                    local_y=y(active_index);
                    local_alpha=alpha_home(active_index);
                    %%%%%%更新local_gi
                    local_gi=local_gi(leave_index);
                end    %%%%%%%%缩减完成
                %%%选择两个变量
                index=(1:active_size);
                index_up=index(local_flag_up);
                index_low=index(local_flag_low);
                [max_value1,max_index]=max(local_gi(local_flag_up));%%  i
                [min_value1,min_index]=min(local_gi(local_flag_low));
                local_index=[index_low(min_index);index_up(max_index)]; %min & max index
                
                %%更新两个alpha
                %%%查找缓存块
                cache_index1=find(Cache_ind==active_index(local_index(1)),1);
                cache_index2=find(Cache_ind==active_index(local_index(2)),1);
                
                if isempty(cache_index1) && isempty(cache_index2)
                    if used_cache==cache_size
                        initQAB=SMO.Kernel(local_x,(local_x(local_index,:))).*(local_y*(local_y(local_index))');%n*2
                        [~,replace_index]=sort(Cache_iter);
                        Cache{replace_index(1)}=initQAB(:,1);
                        Cache{replace_index(2)}=initQAB(:,2);
                        Cache_ind(replace_index([1,2]))=active_index(local_index);
                        Cache_iter(replace_index([1,2]))=index_iteration;
                    elseif used_cache==cache_size-1
                        initQAB=SMO.Kernel(local_x,(local_x(local_index(1),:))).*(local_y*(local_y(local_index(1)))');
                        Cache{used_cache+1}=initQAB(:,1);
                        Cache_ind(used_cache+1)=active_index(local_index(1));
                        Cache_iter(used_cache+1)=index_iteration;
                        used_cache=used_cache+1;
                        initQAB(:,2)=SMO.Kernel(local_x,(local_x(local_index(2),:))).*(local_y*(local_y(local_index(2)))');
                        [~,replace_index]=min(Cache_iter);
                        Cache{replace_index}=initQAB(:,2);
                        Cache_ind(replace_index)=active_index(local_index(2));
                        Cache_iter(replace_index)=index_iteration;
                    else
                        initQAB=SMO.Kernel(local_x,(local_x(local_index,:))).*(local_y*(local_y(local_index))');%n*2
                        
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
                            initQAB=SMO.Kernel(local_x,(local_x(local_index(1),:))).*(local_y*(local_y(local_index(1)))');
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
                            initQAB=SMO.Kernel(local_x,(local_x(local_index(1),:))).*(local_y*(local_y(local_index(1)))');
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
                            initQAB(:,2)=SMO.Kernel(local_x,(local_x(local_index(2),:))).*(local_y*(local_y(local_index(2)))');
                            [~,replace_index]=min(Cache_iter);
                            Cache{replace_index}=initQAB(:,2);
                            Cache_ind(replace_index)=active_index(local_index(2));
                            Cache_iter(replace_index)=index_iteration;
                        else
                            initQAB(:,2)=SMO.Kernel(local_x,(local_x(local_index(2),:))).*(local_y*(local_y(local_index(2)))');
                            Cache{used_cache+1}=initQAB(:,2);
                            Cache_ind(used_cache+1)=active_index(local_index(2));
                            Cache_iter(used_cache+1)=index_iteration;
                            used_cache=used_cache+1;
                        end
                    end
                end
                
                initQBB=initQAB(local_index,:);
                
                initf=-local_y(local_index).*local_gi(local_index)-initQBB*local_alpha(local_index);
            
                sum_two_alpha=sum(local_alpha(local_index).*local_y(local_index)); %y1alpha1+y2alpha=zeta(constant)
                old_alpha=local_alpha(local_index);
                [initAlpha] = SMO.OneSmo(initQBB,initf,sum_two_alpha,local_index,local_initUbIn,local_y,svm_type);
                new_alpha=initAlpha;
                
                local_alpha(local_index)=initAlpha; %update the two new alpha
                if svm_type==3
                    local_flag_up(local_index)=(((initAlpha>fake_zero) & local_y(local_index)==-1) | ((initAlpha<local_initUbIn(local_index) -fake_zero) & local_y(local_index)==1));
                    local_flag_low(local_index)=((initAlpha>fake_zero) & local_y(local_index)==1) | ((initAlpha<local_initUbIn(local_index) -fake_zero) & local_y(local_index)==-1);
                else
                    local_flag_up(local_index)=(((initAlpha>fake_zero) & local_y(local_index)==-1) | ((initAlpha<local_initUbIn -fake_zero) & local_y(local_index)==1));
                    local_flag_low(local_index)=((initAlpha>fake_zero) & local_y(local_index)==1) | ((initAlpha<local_initUbIn -fake_zero) & local_y(local_index)==-1);
                end
             
                local_gi=-local_gi.*local_y;
                local_gi=local_gi+initQAB*(new_alpha-old_alpha);
                local_gi=-local_gi.*local_y;
                
             
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%更新gi_bar
                if svm_type==3
                    uij=(abs(old_alpha-local_initUbIn(local_index))<fake_zero);
                    update_uij=(abs(new_alpha-local_initUbIn(local_index))<fake_zero);
                else
                    uij=(abs(old_alpha-local_initUbIn)<fake_zero);
                    update_uij=(abs(new_alpha-local_initUbIn)<fake_zero);
                end
                
                if active_size==sample_length
                    if ~(uij(1)==update_uij(1))
                        Q_Li=initQAB(:,1);
                        if uij(1)
                            gi_bar=gi_bar-old_alpha(1)*Q_Li;
                        else
                            gi_bar=gi_bar+new_alpha(1)*Q_Li;
                        end
                    end
                    if ~(uij(2)==update_uij(2))
                        Q_Li=initQAB(:,2);
                        if uij(2)
                            gi_bar=gi_bar-old_alpha(2)*Q_Li;
                        else
                            gi_bar=gi_bar+new_alpha(2)*Q_Li;
                        end
                    end
                else
                    if ~(uij(1)==update_uij(1))
                        Q_Li(active_index,:)=initQAB(:,1);
                        Q_Li(other_active_index,:)=SMO.Kernel(x(other_active_index,:),(x(active_index(local_index(1)),:))).*(y(other_active_index)*(y(active_index(local_index(1))))');
                        if uij(1)
                            gi_bar=gi_bar-old_alpha(1)*Q_Li;
                        else
                            gi_bar=gi_bar+new_alpha(1)*Q_Li;
                        end
                    end
                    if ~(uij(2)==update_uij(2))
                        Q_Li(active_index,:)=initQAB(:,2);
                        Q_Li(other_active_index,:)=SMO.Kernel(x(other_active_index,:),(x(active_index(local_index(2)),:))).*(y(other_active_index)*(y(active_index(local_index(2))))');
                        if uij(2)
                            gi_bar=gi_bar-old_alpha(2)*Q_Li;
                        else
                            gi_bar=gi_bar+new_alpha(2)*Q_Li;
                        end
                    end
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
                        [gi_home]=SMO.Reconstruct_gradient(local_gi,gi_bar,active_index,svm_type);
                    end
                    [max_value2]=max(gi_home(flag_up));
                    [min_value2]=min(gi_home(flag_low));
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
                index_iteration = index_iteration+1;
            end

            fprintf('%d\n',index_iteration);
            if index_iteration >= max_iteration_smo
                fprintf('index_iteration >= max_iteration_smo\n');
            end

        end
        function [gi]=Reconstruct_gradient(local_gi,gi_bar,active_index,svm_type)
            global x y p_term initUbIn fake_zero sample_length alpha_home regrad_count;
            other_active_index=(1:sample_length);
            other_active_index(active_index)=[];
            gi=zeros(sample_length,1);
            gi(active_index,:)=-y(active_index).*local_gi;
            if svm_type==1
                gi(other_active_index,:)=gi_bar(other_active_index)-p_term;  %初始化非活动集的gi
            else
                gi(other_active_index,:)=gi_bar(other_active_index)-p_term(other_active_index);  %初始化非活动集的gi
            end
            if svm_type==3
                free_flag=abs(alpha_home(active_index)-initUbIn(active_index))>fake_zero & alpha_home(active_index)>fake_zero;
            else
                free_flag=abs(alpha_home(active_index)-initUbIn)>fake_zero & alpha_home(active_index)>fake_zero;
            end
            
            free_index=active_index(free_flag);
            reSize=100000;
            if length(other_active_index)>reSize
                reGroup=length(other_active_index)/reSize;
                for i=1:reGroup
                    QPiAF=SMO.Kernel(x(other_active_index(((i-1)*reSize+1):i*reSize),:),x(free_index,:)).*(y(other_active_index(((i-1)*reSize+1):i*reSize))*(y(free_index))');
                    gi(other_active_index(((i-1)*reSize+1):i*reSize))=gi(other_active_index(((i-1)*reSize+1):i*reSize))+QPiAF*alpha_home(free_index);
                end
                if i*reSize<length(other_active_index)
                    QPiAF=SMO.Kernel(x(other_active_index(i*reSize+1:end),:),x(free_index,:)).*(y(other_active_index(i*reSize+1:end))*(y(free_index))');
                    gi(other_active_index(i*reSize+1:end))=gi(other_active_index(i*reSize+1:end))+QPiAF*alpha_home(free_index);
                end
            else
                %%循环很耗时，但是不循环可能矩阵太大，怎么解决？
                QNAf=SMO.Kernel(x(other_active_index,:),(x(free_index,:))).*(y(other_active_index)*(y(free_index))');
                gi(other_active_index)=gi(other_active_index)+QNAf*alpha_home(active_index(free_flag));
            end
            gi=-y.*gi;
            regrad_count=regrad_count+1;
        end
        function [leave_index,active_index,active_size,local_gi]=DoShrinking(local_gi,gi_bar,local_flag_up,local_flag_low,local_alpha,local_initUbIn,local_y,active_index,svm_type)
            global y fake_zero shrink_state epsilon initUbIn alpha_home flag_up flag_low index_home
            %%% 最大违反对
            [max_value]=max(local_gi(local_flag_up));
            [min_value]=min(local_gi(local_flag_low));
            
            if(shrink_state==0) && (max_value-min_value)<=10*epsilon
                shrink_state=1;
                %%%%
                [gi_home]=SMO.Reconstruct_gradient(local_gi,gi_bar,active_index,svm_type);
                local_gi=gi_home;
                %%% 最大违反对 这里需要重新求吗？
                [max_value]=max(gi_home(flag_up));
                [min_value]=min(gi_home(flag_low));
                flag_low_bound=(alpha_home>initUbIn-fake_zero & y==1) | (alpha_home<fake_zero & y==-1);
                flag_up_bound=(alpha_home<fake_zero & y==1) | (alpha_home>initUbIn-fake_zero & y==-1);
                leave_index=(gi_home>max_value+fake_zero & flag_low_bound)...
                    |(gi_home<min_value-fake_zero & flag_up_bound);
                leave_index=~leave_index;
                active_index=index_home(leave_index);
                active_size=length(active_index);
            else
                flag_low_bound=(local_alpha>local_initUbIn-fake_zero & local_y==1) | (local_alpha<fake_zero & local_y==-1);
                flag_up_bound=(local_alpha<fake_zero & local_y==1) | (local_alpha>local_initUbIn-fake_zero & local_y==-1);
                leave_index=(local_gi>max_value+fake_zero & flag_low_bound)...
                    |(local_gi<min_value-fake_zero & flag_up_bound);
                leave_index=~leave_index;
                active_index=active_index(leave_index);
                active_size=length(active_index);
            end
        end
        function [initAlpha] = OneSmo(initQ,initf,sum_two_alpha,two_index,ub,local_y,svm_type) %update the two alpha
            %             global y
            first_index=two_index(1);
            second_index=two_index(2);
            y1=local_y(first_index);
            y2=local_y(second_index);
            yy=y1*y2;
            %convert to quadratic equation of one variable
            a=(initQ(1,1)+initQ(2,2)-2*yy*initQ(1,2))/2;
            b=-sum_two_alpha*y1*initQ(2,2)+initQ(1,2)*y2*sum_two_alpha+initf'*[1;-yy];
            % c=0.5*initQ(2,2)*sum_two_alpha*sum_two_alpha + initf(2)*sum_two_alpha*y2;
            if svm_type==3
                if yy==1    %y1=y2  y1*alpha1+y2*alpha2=zeta --y1(y1*alpha1+y2*alpha)=alpha1+alpha2
                    left_bound=max(0,y1*sum_two_alpha-ub(second_index));     %L
                    right_bound=min(ub(first_index),y1*sum_two_alpha);      %H
                else    %y1<>y2  y1(y1*alpha1+y2*alpha)=alpha1-alpha2
                    left_bound=max(0,y1*sum_two_alpha);
                    right_bound=min(ub(first_index),ub(second_index)+y1*sum_two_alpha);
                end
            else
                if yy==1    %y1=y2  y1*alpha1+y2*alpha2=zeta --y1(y1*alpha1+y2*alpha)=alpha1+alpha2
                    left_bound=max(0,y1*sum_two_alpha-ub);     %L
                    right_bound=min(ub,y1*sum_two_alpha);      %H
                else    %y1<>y2  y1(y1*alpha1+y2*alpha)=alpha1-alpha2
                    left_bound=max(0,y1*sum_two_alpha);
                    right_bound=min(ub,ub+y1*sum_two_alpha);
                end
            end
            opitimal_alpha1=-b/(2*a);
            if opitimal_alpha1 > right_bound
                opitimal_alpha1=right_bound;
            end
            if opitimal_alpha1 < left_bound
                opitimal_alpha1=left_bound;
            end
            initAlpha=[opitimal_alpha1; y2*sum_two_alpha-yy*opitimal_alpha1];
        end
        
        function out_KKT=TestKKT(os,C,C_plus,C_minus)
            global fake_zero2
            global y
            out_KKT=0;
            local_g=SMO.CaculateF(os);
            alpha=os.alpha;
            zero=alpha'*y;
            if zero<fake_zero2 && zero>-fake_zero2
                local_tmp=SMO.SubKKT(alpha,local_g,C,C_plus,C_minus);
                if local_tmp==1
                    out_KKT=1;
                end
            end
        end

        function f=CaculateF(os)
            global x y p_term;
            Q=SMO.Kernel(x,x).*(y*y');
            local_Q=[y Q];
            local_alpha=[os.b;os.alpha];
            local_f=local_Q*local_alpha;
            f=local_f-p_term;
        end
        
        function out_KKT=SubKKT(alpha,local_g,C,C_plus,C_minus)
            global fake_zero2
            global y;
            out_KKT=1;
            sum_length=length(alpha);
            C=C*C_plus*(y==1)+C*(2*C_minus-1)*(y==-1);
            for i=1:sum_length
                if alpha(i)<fake_zero2
                    if local_g(i)<-fake_zero2
                        out_KKT=0;
                        break;
                    end
                else
                    if alpha(i)>C(i)-fake_zero2
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
        function out_KKT=TestKKT2(os,C)
            global fake_zero2
            global y
            out_KKT=0;
            local_g=SMO.CaculateF(os);
            alpha=os.alpha;
            zero=alpha'*y;
            if zero<fake_zero2 && zero>-fake_zero2
                local_tmp=SMO.SubKKT2(alpha,local_g,C);
                if local_tmp==1
                    out_KKT=1;
                end
            end
        end
        function out_KKT=SubKKT2(alpha,local_g,C)
            global fake_zero2
            out_KKT=1;
            sum_length=length(alpha);
            for i=1:sum_length
                if alpha(i)<fake_zero2
                    if local_g(i)<-fake_zero2
                        out_KKT=0;
                        break;
                    end
                else
                    if alpha(i)>C-fake_zero2
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
        function [LowBound,UppBound]=ComputerCCstar(original_x,original_y,os)
            global C
%             LowBoud=zeros(label_size,1);
%             UppBound=C*ones(label_size,1);
%             if label_size>0
%                 x([1:label_size],:)=[];
%                 y([1:label_size])=[];
%             end
            Q=SMO.Kernel(original_x,os.x).*(original_y*(os.y)');
            local_Q=[original_y Q];
            local_alpha=[os.b;os.alpha];
            local_f=local_Q*local_alpha;
            mu=C*(local_f<0);
            LowBound=(-mu).*(original_y>0)+(mu-C).*(original_y<0);
            UppBound=(C-mu).*(original_y>0)+(mu).*(original_y<0);
        end
    end
end

