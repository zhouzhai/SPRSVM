function out=initial(original_x,original_y)
    global C
    os=SMO.InitialSolution(original_x,original_y,C);
    [PreLowBound,PreUppBound]=SMO.ComputerCCstar(original_x,original_y,os);
    terminal_flag=0;
    NowLowBound=PreLowBound;
    NowUppBound=PreUppBound;
    times=0;
    while terminal_flag==0
        objcs=SemiSVM(original_x,original_y,NowLowBound,NowUppBound,0);
        PreLowBound=NowLowBound;
        PreUppBound=NowUppBound;
        [NowLowBound,NowUppBound]=SemiSVM.ComputerCCstar(original_x,original_y,objcs);
        terminal_flag=SemiSVM.CheckTerminial(PreLowBound,PreUppBound,NowLowBound,NowUppBound);
        times=times+1;
    end
    fprintf('循环次数为：%d\n',times);
    out=objcs;
    out.local_minimal=SemiSVM.TestMinimal(out);
    
end

