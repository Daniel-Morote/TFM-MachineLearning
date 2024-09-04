% Resultados

clc
clear all



FunPara.kerfPara.type = 'lin'; 

% Nombre del archivo Excel donde se guardarán los resultados
filename_xlsx = 'Resumen_Resultados.xlsx';

% Nombre del archivo .mat donde se guardarán los resultados
filename_mat = "Resumen_Resultados.mat";

% Arreglo de estructuras donde almacenarán los resultados
final_results = struct();

% Conjuntos de datos utilizados:
datasets = ["BreastMNIST" , "DermaMNIST_0vs2", "DermaMNIST_0vs4"];

for dataset = datasets
    
    disp(dataset)

    % Cargamos el conjunto de datos
    load(strcat(dataset,".mat"))

    X = data_train;
    Y = labels_train;

    Xtest=data_test;
    Ytest=labels_test;
    Ylogit=Ytest;
    Ylogit(Ytest==-1)=0;

    % Estructura que contendrá los resultados relativos a 'dataset'
    final_results.(dataset) = struct();

    %% SVM con kernel lineal

    disp("SVM lineal")
    load("CV_SVM_lineal.mat")

    FunPara.kerfPara.type = 'lin'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.c = maxBAC.value(strcmp(maxBAC.param, "C"));
    
    [prediction,T_s,Sol] = SVM_soft_quadsolve(X,Y,Xtest,FunPara); % via Quadolve
    [A,B]=platt_improve(Sol.Val_Xt,Ytest);
    Prob_estimates_pos=1./(1+exp(A*Sol.Val_Xt+B));
    [BAC_s,ACCU_s]=medi_auc_accu(prediction,Ytest);
    ROC_s = calcROC(Prob_estimates_pos(:,1),Ylogit,1);
    [~,AUC_s] = calcPerformance(ROC_s);

    clear FunPara;

    %% PSVM con kernel lineal

    disp("PSVM lineal")
    load("CV_PSVM_lineal.mat")

    FunPara.kerfPara.type = 'lin'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.C = maxBAC.value(strcmp(maxBAC.param, "C"));
    FunPara.epsi = maxBAC.value(strcmp(maxBAC.param, "eps"));

    [prediction,T_p,Solp] = PSVM_quadprog(X,Y,Xtest,FunPara);
    [BAC_p,ACCU_p]=medi_auc_accu(prediction,Ytest);
    ROC_p = calcROC(Solp.Prob(:,1),Ylogit,1);
    [~,AUC_p] = calcPerformance(ROC_p);

    clear FunPara;

    %% CPSVM_1 con kernel lineal

    disp("CPSVM_1 lineal")
    load("CV_CPSVM_V1_lineal.mat")

    FunPara.kerfPara.type = 'lin'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.C1 = maxBAC.value(strcmp(maxBAC.param, "C1"));
    FunPara.C2 = maxBAC.value(strcmp(maxBAC.param, "C2"));
    FunPara.epsi = maxBAC.value(strcmp(maxBAC.param, "eps"));

    [prediction,T_cp1,Solcps] = cpsvm_dual_qpV1(X,Y,Xtest,FunPara);
    [BAC_cp1,ACCU_cp1]=medi_auc_accu(prediction,Ytest);
    ROC_cp1 = calcROC(Solcps.Prob(:,1),Ylogit,1);
    [~,AUC_cp1] = calcPerformance(ROC_cp1);

    clear FunPara;

    %% CPSVM_2 con kernel lineal

    disp("CPSVM_2 lineal")
    load("CV_CPSVM_V2_lineal.mat")
    
    FunPara.kerfPara.type = 'lin'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.C1 = maxBAC.value(strcmp(maxBAC.param, "C1"));
    FunPara.C2 = maxBAC.value(strcmp(maxBAC.param, "C2"));
    FunPara.epsi = maxBAC.value(strcmp(maxBAC.param, "eps"));

    [prediction,T_cp2,Solcps] = cpsvm_dual_qpV2(X,Y,Xtest,FunPara);
    [BAC_cp2,ACCU_cp2]=medi_auc_accu(prediction,Ytest);
    ROC_cp2 = calcROC(Solcps.Prob(:,1),Ylogit,1);
    [~,AUC_cp2] = calcPerformance(ROC_cp2);

    clear FunPara;

    %% Output measure 
    Out_ACCU=[ACCU_s; ACCU_p; ACCU_cp1; ACCU_cp2];
    Out_BAC=[BAC_s; BAC_p; BAC_cp1; BAC_cp2];
    Out_AUC=[AUC_s; AUC_p; AUC_cp1; AUC_cp2];
    Out_Time=[T_s; T_p; T_cp1; T_cp2];

    final_results.(dataset).ACCU = Out_ACCU;
    final_results.(dataset).BAC = Out_BAC;
    final_results.(dataset).AUC = Out_AUC;
    final_results.(dataset).Time = Out_Time;
    
    % Graphic of the ROC curve
    figure
    plot(ROC_s.F1ch,ROC_s.F0ch,'LineWidth',2)
    hold on
    plot(ROC_p.F1ch,ROC_p.F0ch,'LineWidth',2)
    plot(ROC_cp1.F1ch,ROC_cp1.F0ch,'LineWidth',2)
    plot(ROC_cp2.F1ch,ROC_cp2.F0ch,'LineWidth',2)
    xlabel('Tasa de Falsos Positivos (1-Especificidad)')
    ylabel('Tasa de Veraderos Positivos (Sensibilidad)')
    legend({'SVM','PSVM','CPSVM1','CPSVM2'},'Location','southeast')
    hold off

    %% SVM con kernel gaussiano

    disp("SVM nolineal")
    load("CV_SVM_nolineal.mat")

    FunPara.kerfPara.type = 'rbf'; 

    maxBAC = results.(dataset).maxBAC;
    FunPara.c = maxBAC.value(strcmp(maxBAC.param, "C"));
    FunPara.kerfPara.pars = maxBAC.value(strcmp(maxBAC.param, "sigma"));

    [prediction,T_s,Sol] = SVM_soft_quadsolve(X,Y,Xtest,FunPara); % via Quadolve
    [A,B]=platt_improve(Sol.Val_Xt,Ytest);
    Prob_estimates_pos=1./(1+exp(A*Sol.Val_Xt+B));
    [BAC_s,ACCU_s]=medi_auc_accu(prediction,Ytest);
    ROC_s = calcROC(Prob_estimates_pos(:,1),Ylogit,1);
    [~,AUC_s] = calcPerformance(ROC_s);

    clear FunPara;

    %% PSVM con kernel gaussiano

    disp("PSVM nolineal")
    load("CV_PSVM_nolineal.mat")

    FunPara.kerfPara.type = 'rbf'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.C = maxBAC.value(strcmp(maxBAC.param, "C"));
    FunPara.epsi = maxBAC.value(strcmp(maxBAC.param, "eps"));
    FunPara.kerfPara.pars = maxBAC.value(maxBAC.param == "sigma");

    [prediction,T_p,Solp] = PSVM_quadprog(X, Y, Xtest, FunPara);
    [BAC_p,ACCU_p]=medi_auc_accu(prediction,Ytest);
    ROC_p = calcROC(Solp.Prob(:,1),Ylogit,1);
    [~,AUC_p] = calcPerformance(ROC_p);

    clear FunPara;


    %% CPSVM_1 con kernel gaussiano

    disp("CPSVM_1 nolineal")
    load("CV_CPSVM_V1_nolineal.mat")

    FunPara.kerfPara.type = 'rbf'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.C1 = maxBAC.value(strcmp(maxBAC.param, "C1=C2"));
    FunPara.C2 = maxBAC.value(strcmp(maxBAC.param, "C1=C2"));
    FunPara.epsi = maxBAC.value(strcmp(maxBAC.param, "eps"));
    FunPara.kerfPara.pars = maxBAC.value(maxBAC.param == "sigma");

    [prediction,T_cp1,Solcp] = cpsvm_dual_qpV1(X, Y, Xtest, FunPara);
    [BAC_cp1,ACCU_cp1]=medi_auc_accu(prediction,Ytest);
    ROC_cp1 = calcROC(Solcp.Prob(:,1),Ylogit,1);
    [~,AUC_cp1] = calcPerformance(ROC_cp1);

    clear FunPara;


    %% CPSVM_2 con kernel gaussiano

    disp("CPSVM_2 nolineal")
    load("CV_CPSVM_V2_nolineal.mat")

    FunPara.kerfPara.type = 'rbf'; 
    maxBAC = results.(dataset).maxBAC;
    FunPara.C1 = maxBAC.value(strcmp(maxBAC.param, "C1=C2"));
    FunPara.C2 = maxBAC.value(strcmp(maxBAC.param, "C1=C2"));
    FunPara.epsi = maxBAC.value(strcmp(maxBAC.param, "eps"));
    FunPara.kerfPara.pars = maxBAC.value(maxBAC.param == "sigma");

    [prediction,T_cp2,Solcps] = cpsvm_dual_qpV2(X, Y, Xtest, FunPara);
    [BAC_cp2,ACCU_cp2]=medi_auc_accu(prediction,Ytest);
    ROC_cp2 = calcROC(Solcps.Prob(:,1),Ylogit,1);
    [~,AUC_cp2] = calcPerformance(ROC_cp2);

    clear FunPara;

    %% Output measure  
    %Out_meanPk=[mean_Prob_svm;mean_Prob_psvm;mean_Prob_cpsvm1;mean_Prob_cpsvm2];
    Out_ACCUk=[ACCU_s; ACCU_p; ACCU_cp1; ACCU_cp2];
    Out_BACk=[BAC_s; BAC_p; BAC_cp1; BAC_cp2];
    Out_AUCk=[AUC_s; AUC_p; AUC_cp1; AUC_cp2];
    Out_Timek=[T_s; T_p; T_cp1; T_cp2];


    final_results.(dataset).ACCUk = Out_ACCUk;
    final_results.(dataset).BACk = Out_BACk;
    final_results.(dataset).AUCk = Out_AUCk;
    final_results.(dataset).Timek = Out_Timek;

    %% Graphic of the ROC curve
    figure
    plot(ROC_s.F1ch,ROC_s.F0ch,'LineWidth',2)
    hold on
    plot(ROC_p.F1ch,ROC_p.F0ch,'LineWidth',2)
    plot(ROC_cp1.F1ch,ROC_cp1.F0ch,'LineWidth',2)
    plot(ROC_cp2.F1ch,ROC_cp2.F0ch,'LineWidth',2)
    xlabel('Tasa de Falsos Positivos (1-Especificidad)')
    ylabel('Tasa de Veraderos Positivos (Sensibilidad)')
    legend({'SVM','PSVM','CPSVM1','CPSVM2'},'Location','southeast')
     
end

% Guarda 'results' en un archivo .mat 
save(filename_mat, 'final_results');

