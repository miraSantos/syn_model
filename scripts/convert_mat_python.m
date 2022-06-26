cd /D/MIT-WHOI/github_repos/syn_model/data/raw

%%
load("syndata_current.mat")
alldatenum= datetime(allmatdate,'ConvertFrom','datenum','Format','yyyy-MM-dd''T''HH:mm:ss')
writematrix(alldatenum,"alldatenum.txt","Delimiter","tab")

daily_syn_conc_export = daily_syn(:)
time_syn_export = datetime(time_syn(:),'ConvertFrom','datenum','Format','yyyy-MM-dd')

writematrix(daily_syn_conc_export,"../dailysynconc_matrix.txt","Delimiter","tab")
writematrix(time_syn_export,"../dailysyntime_matrix.txt","Delimiter","tab")


%%
load("mvco_envdata_current.mat")


%%
load("nut_data_reps.mat")
writecell(MVCO_nut_reps, "mvco_nutrients.csv")
writecell(header_nut, "mvco_nutrients_headers.csv")

%%
load("MVCO_Environmental_Tables.mat")
writetimetable(MVCO_Env_Table, "mvco_env_table.csv")
writetable(MVCO_Daily,"mvco_daily.csv")