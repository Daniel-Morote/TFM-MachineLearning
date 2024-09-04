% Cross Validation SVM soft margin (linear kernel)

clc
clear all

% Definición de los rangos de exploración para el parámetro C
C_l = -7;     % Valor mínimo del exponente de 2 para C
C_h = 7;      % Valor máximo del exponente de 2 para C

FunPara.kerfPara.type='lin'; % Tipo de kernel: lineal

% Nombre del archivo Excel donde se guardarán los resultados
filename_xlsx = 'CV_SVM.xlsx';

% Nombres de los valores posibles del parámetro C
C_names = strcat('2^',string(C_l:C_h));

% Nombre del archivo .mat donde se guardarán los resultados
filename_mat = "CV_SVM_lineal.mat";

% Arreglo de estructuras donde almacenarán los resultados
results = struct();

% Conjuntos de datos utilizados:
datasets = ["BreastMNIST" , "DermaMNIST_0vs2", "DermaMNIST_0vs4"];

for dataset = datasets

    % Estructura que contendrá los resultados relativos a 'dataset'
    results.(dataset) = struct();

    % Inicia el temporizador para medir el tiempo de ejecución
    tic;
    
    % Muestra el nombre del conjunto de datos actual
    disp(dataset) 
    
    % Carga el archivo .mat que contiene los índices de validación cruzada y los datos de entrenamiento
    load(strcat('CV_',dataset,'.mat'))   
    
    folds = max(CV_indices); % Número de iteraciones para la validación cruzada
    X = data_train;
    Y = labels_train;

    % Inicialización de vectores para almacenar la exactitud (ACCU) y la
    % exatitud equilibrada (BAC)
    BACMATRIX=zeros(1,C_h-C_l+1);
    ACCUMATRIX=zeros(1,C_h-C_l+1);

    for i=C_l:C_h
        i
        FunPara.c=2^i;
        for k=1:folds

            idx_test = (CV_indices == k); % Índices de datos de prueba
            idx_train = ~idx_test;        % Índices de datos de entrenamiento

            X_train = X(idx_train, :); % Datos de entrenamiento
            Y_train = Y(idx_train);    % Etiquetas de entrenamiento

            X_test = X(idx_test, :); % Datos de prueba
            Y_test = Y(idx_test);    % Etiquetas de prueba

            Y_predic = SVM_softcvx(X_train,Y_train,X_test,FunPara);   % Predicciones via CVX

            [BAC(k),ACCU(k)]=medi_auc_accu(Y_predic,Y_test);  % Calcula BAC y ACCU para dichas predicciones

        end

        % Calcula y almacena el BAC y ACCU promedio
        BACMATRIX(i-C_l+1)=mean(BAC);
        ACCUMATRIX(i-C_l+1)=mean(ACCU);

    end

    % Tiempo utilizado en la validación cruzada
    tiempoTranscurrido = toc;  % Tiempo transcurrido desde tic hasta toc
    disp(['El tiempo utilizado es: ' num2str(tiempoTranscurrido) ' segundos']);


    % Escribe los resultados en el archivo Excel
    xlswrite(filename_xlsx, "a) Kernel lineal", dataset, 'B3');
    xlswrite(filename_xlsx, "C", dataset, 'B5');
    xlswrite(filename_xlsx, C_names, dataset, 'C5:Q5');
    xlswrite(filename_xlsx, "Accu.", dataset, 'B6');
    xlswrite(filename_xlsx, ACCUMATRIX, dataset, 'C6:Q6');
    xlswrite(filename_xlsx, "Bal. Accu.", dataset, 'B7');
    xlswrite(filename_xlsx, BACMATRIX, dataset, 'C7:Q7');

    % Almacena los resultados en la variable 'results'

    param = ["C"]; % Nombre de los parámetros

    results.(char(dataset)).BACMATRIX = BACMATRIX;                      % Guardamos BACMATRIX  en 'results' 
    [maxBAC, i] = max(BACMATRIX(:));                                    % Calculamos el máximo de BACMATRIX
    value = [2^(i+C_l-1)];                                              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([i+C_l-1]));                         % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxBAC = table(param, value, valueName);    % Guardamos la informacion en una tabla

    results.(char(dataset)).ACCUMATRIX = ACCUMATRIX;                    % Guardamos ACCUMATRIX  en 'results' 
    [maxACCU, i] = max(ACCUMATRIX(:));                                  % Calculamos el máximo de ACCUMATRIX
    value = [2^(i+C_l-1)];                                              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([i+C_l-1]));                         % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxACCU = table(param, value, valueName);   % Guardamos la informacion en una tabla

end

% Guarda 'results' en un archivo .mat 
save(filename_mat, 'results');


