function [original_x,original_y,added_x,added_y]=random_select(x,y) 
    global  size_training added_num
    res = [];
    while length(res) ~= size_training
        tmp_res = unique(randi([1,length(y)],size_training,1));
        res = [res;tmp_res];
        res = unique(res);
        if length(res) >= size_training
            res = res(1:size_training);
            res = sort(res);
        end
    end
    original_x=x(res,:);
    original_y=y(res);
    
    index=(1:length(y));
    index(res)=[];
    res=[];
    while length(res)~= added_num
        index1=unique(randi([1,length(index)],added_num,1));
        res=[res;index1];
        res=unique(res);
        if length(res) >= added_num
            res = res(1:added_num);
            res = sort(res);
        end
    end
    added_x=x(index(res),:);
    added_y=y(index(res));
end

