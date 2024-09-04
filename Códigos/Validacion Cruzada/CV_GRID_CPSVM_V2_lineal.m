% Cross Validation CPSVM_2 (linear kernel)

clc
clear all

% Definición de los rangos de exploración para los parámetros C1, C2 y epsilon
C1_l = -7;     % Valor mínimo del exponente de 2 para C1
C1_h = 7;      % Valor máximo del exponente de 2 para C1

C2_l = -7;     % Valor mínimo del exponente de 2 para C2
C2_h = 7;      % Valor máximo del exponente de 2 para C2

eps_l = -7;   % Valor mínimo del exponente de 2 para epsilon
eps_h = 0;    % Valor máximo del exponente de 2 para epsilon

FunPara.kerfPara.type='lin'; % Tipo de kernel: lineal

% Nombre del archivo Excel donde se guardarán los resultados
filename_xlsx = 'CV_CPSVM_V2_lineal.xlsx';

% Nombres de los valores posibles de los parámetros C1, C2 y epsilon
C1_names = strcat("2^",string(C1_l:C1_h));
C2_names = strcat("2^",string(C2_l:C2_h));
epsilon_names = strcat("2^",string(eps_l:eps_h));

% Nombre del archivo .mat donde se guardarán los resultados
filename_mat = "CV_CPSVM_V2_lineal.mat";

% Arreglo de estructuras donde almacenarán los resultados
results = struct();

% Conjuntos de datos utilizados:
datasets = ["BreastMNIST", "DermaMNIST_0vs2", "DermaMNIST_0vs4"];

for dataset = datasets

    % Estructura que contendrá los resultados relativos a 'dataset'
    results.(dataset) = struct();

    % Inicia el temporizador para medir el tiempo de ejecución
    tic;
    
    % Muestra el nombre del conjunto de datos actual
    disp(dataset) 
    
    % Carga el archivo .mat que contiene los índices de validación cruzada y los datos de entrenamiento
    load(strcat('CV_',dataset,'.mat'))  

    folds = max(CV_indices);  % Número de iteraciones para la validación cruzada
    X = data_train;           
    Y = labels_train;

    % Inicialización de matrices para almacenar la exactitud (ACCU) y la
    % exatitud equilibrada (BAC)
    BACMATRIX=zeros(eps_h-eps_l+1,C1_h-C1_l+1, C2_h-C2_l+1);
    ACCUMATRIX=zeros(eps_h-eps_l+1, C1_h-C1_l+1, C2_h-C2_l+1);

    for j=eps_l:eps_h 
        FunPara.epsi = 2^j;
        for i1=C1_l:C1_h
             FunPara.C1 =2^i1;
            for i2=C1_l:C1_h
                FunPara.C2=2^i2;                
                fprintf('(%d, %d, %d) \n',j,i1,i2);
                for k=1:folds

                    idx_test = (CV_indices == k); % Índices de datos de prueba
                    idx_train = ~idx_test;        % Índices de datos de entrenamiento

                    X_train = X(idx_train, :); % Datos de entrenamiento
                    Y_train = Y(idx_train);    % Etiquetas de entrenamiento

                    X_test = X(idx_test, :); % Datos de prueba
                    Y_test = Y(idx_test);    % Etiquetas de prueba

                    Y_predic = cpsvm_dual_qpV2(X_train,Y_train,X_test,FunPara); % Predicciones via quadprog

                    [BAC(k),ACCU(k)]=medi_auc_accu(Y_predic,Y_test); % Calcula BAC y ACCU para dichas predicciones

                end

                % Calcula y almacena el BAC y ACCU promedio
                BACMATRIX(j-eps_l+1,i1-C1_l+1,i2-C2_l+1)=mean(BAC);
                ACCUMATRIX(j-eps_l+1, i1-C1_l+1,i2-C2_l+1)=mean(ACCU); 

            end
        end

        % Escribe los resultados en el archivo Excel

        xlswrite(filename_xlsx, {strcat("eps=",epsilon_names(j-eps_l+1))}, dataset, strcat('B',string(4+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, {"C1\C2"}, dataset, strcat('B',string(5+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, C2_names, dataset, strcat('C',string(5+(j-eps_l)*18),':Q',string(5+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, C1_names', dataset, strcat('B',string(6+(j-eps_l)*18),':B',string(20+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, squeeze(ACCUMATRIX(j-eps_l+1,:,:)), dataset, strcat('C',string(6+(j-eps_l)*18),':Q',string(20+(j-eps_l)*18)) );

        xlswrite(filename_xlsx, {strcat("eps=",epsilon_names(j-eps_l+1))}, dataset, strcat('T',string(4+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, {"C1\C2"}, dataset, strcat('T',string(5+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, C2_names, dataset, strcat('U',string(5+(j-eps_l)*18),':AI',string(5+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, C1_names', dataset, strcat('T',string(6+(j-eps_l)*18),':T',string(20+(j-eps_l)*18)) );
        xlswrite(filename_xlsx, squeeze(BACMATRIX(j-eps_l+1,:,:)), dataset, strcat('U',string(6+(j-eps_l)*18),':AI',string(20+(j-eps_l)*18)) );

    end

    % Tiempo utilizado en la validación cruzada
    tiempoTranscurrido = toc;  % Tiempo transcurrido desde tic hasta toc
    disp(['El tiempo utilizado es: ' num2str(tiempoTranscurrido) ' segundos']);

    % Almacena los resultados en la variable 'results'

    param = ["eps", "C1", "C2"]; % Nombre de los parámetros

    results.(char(dataset)).BACMATRIX = BACMATRIX;                      % Guardamos BACMATRIX  en 'results' 
    [maxBAC, linearIndex] = max(BACMATRIX(:));                          % Calculamos el máximo de BACMATRIX
    [j, i1, i2] = ind2sub(size(BACMATRIX), linearIndex);                  % Calculamos los índices del máximo de BACMATRIX
    value = [2^(j+eps_l-1), 2^(i1+C1_l-1), 2^(i2+C2_l-1)];              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([j+eps_l-1, i1+C1_l-1, i2+C2_l-1])); % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxBAC = table(param, value, valueName);    % Guardamos la informacion en una tabla

    results.(char(dataset)).ACCUMATRIX = ACCUMATRIX;                    % Guardamos ACCUMATRIX  en 'results' 
    [maxACCU, linearIndex] = max(ACCUMATRIX(:));                        % Calculamos el máximo de ACCUMATRIX
    [j, i1, i2] = ind2sub(size(ACCUMATRIX), linearIndex);               % Calculamos los índices del máximo de ACCUMATRIX
    value = [2^(j+eps_l-1), 2^(i1+C1_l-1), 2^(i2+C2_l-1)];              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([j+eps_l-1, i1+C1_l-1, i2+C2_l-1])); % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxACCU = table(param, value, valueName);   % Guardamos la informacion en una tabla
end

% Guarda 'results' en un archivo .mat 
save(filename_mat, 'results');
