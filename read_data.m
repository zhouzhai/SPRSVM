function [x,y]=read_data(flag)
global label_size 
    if flag==1
        load CodRNA;
        label_size=300;
    end
    if flag==2
        load a9a;
        label_size=700;
    end
    if flag==3
        load ijcnn1;
        label_size=600;
    end
    if flag==4
        load susy;
        label_size=900;
    end
    if flag==5
        load W6a;
        label_size=1800;
    end
    if flag==6
        load usps;
        label_size=300;
    end
    if flag==7
        load covtype;
        label_size=59535;
    end
    if flag==8
        load dota2
    end
    if flag==9
        load card
    end
    x=zscore(x); %±ê×¼»¯
end
