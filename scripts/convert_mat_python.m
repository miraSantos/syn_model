cd /D/MIT-WHOI/github_repos/syn_model/data/

%%
load("syndata_current.mat")
alldatenum= datetime(allmatdate,'ConvertFrom','datenum','Format','yyyy-MM-dd''T''HH:mm:ss')
writematrix(alldatenum,"alldatenum.txt","Delimiter","tab")


%%
load("mvco_envdata_current.mat")


%%
load("nut_data_reps.mat")
writecell(MVCO_nut_reps, "mvco_nutrients.csv")
writecell(header_nut, "mvco_nutrients_headers.csv")

%%
load("MVCO_Environmental_Tables.mat")
writetimetable(MVCO_Env_Table, "mvco_env_table.csv")