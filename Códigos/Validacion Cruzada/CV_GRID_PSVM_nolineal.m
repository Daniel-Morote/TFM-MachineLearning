% Cross Validation PSVM (gaussian kernel)

clc
clear all

% Definición de los rangos de exploración para los parámetros C y epsilon
C_l = -7;     % Valor mínimo del exponente de 2 para C
C_h = 7;      % Valor máximo del exponente de 2 para C

eps_l = -7;   % Valor mínimo del exponente de 2 para epsilon
eps_h = 0;    % Valor máximo del exponente de 2 para epsilon

sigma_l=-7;    % Valor mínimo del exponente de 2 para sigma
sigma_h=7;     % Valor mínimo del exponente de 2 para sigma

FunPara.kerfPara.type='rbf'; % Tipo de kernel: radial gaussiano

% Nombre del archivo Excel donde se guardarán los resultados
filename_xlsx = 'CV_PSVM_nolineal.xlsx';

% Nombres de las columnas para el parámetro C y epsilon en el archivo Excel
C_names = strcat("2^",string(C_l:C_h));
epsilon_names = strcat("2^",string(eps_l:eps_h));
sigma_names = strcat("2^",string(sigma_l:sigma_h));

% Nombre del archivo .mat donde se guardarán los resultados
filename_mat = "CV_PSVM_nolineal.mat";

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

    folds = max(CV_indices);  % Número de iteraciones para la validación cruzada
    X = data_train;           
    Y = labels_train;        

    % Inicialización de matrices para almacenar la exactitud (ACCU) y la
    % exatitud equilibrada (BAC)
    BACMATRIX=zeros(sigma_h-sigma_l+1, C_h-C_l+1, eps_h-eps_l+1);
    ACCUMATRIX=zeros(sigma_h-sigma_l+1, C_h-C_l+1, eps_h-eps_l+1);

    for t=sigma_l:sigma_h
        FunPara.kerfPara.pars=2^t;
        for i=C_l:C_h
            FunPara.C=2^i;
            for j=eps_l:eps_h                
                fprintf('(%d, %d, %d) \n',t,i,j);
                FunPara.epsi = 2^j;
                for k=1:folds

                    idx_test = (CV_indices == k);   % Índices de datos de prueba
                    idx_train = ~idx_test;          % Índices de datos de entrenamiento

                    X_train = X(idx_train, :); % Datos de entrenamiento
                    Y_train = Y(idx_train);    % Etiquetas de entrenamiento

                    X_test = X(idx_test, :); % Datos de prueba
                    Y_test = Y(idx_test);    % Etiquetas de prueba

                    Y_predic =  PSVM_quadprog(X_train,Y_train,X_test,FunPara); % Predicciones via quadprog

                    [BAC(k),ACCU(k)]=medi_auc_accu(Y_predic,Y_test); % Calcula BAC y ACCU para dichas predicciones

                end

                % Calcula y almacena el BAC y ACCU promedio
                BACMATRIX(t-sigma_l+1,i-C_l+1,j-eps_l+1)=mean(BAC);
                ACCUMATRIX(t-sigma_l+1, i-C_l+1,j-eps_l+1)=mean(ACCU); 
            end

        end

        % Escribe los resultados en el archivo Excel

        xlswrite(filename_xlsx, {strcat("sigma=",sigma_names(t-sigma_l+1))}, dataset, strcat('B',string(4+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, {"eps\C1=C2"}, dataset, strcat('B',string(5+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, C_names, dataset, strcat('C',string(5+(t-sigma_l)*11),':Q',string(5+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, epsilon_names', dataset, strcat('B',string(6+(t-sigma_l)*11),':B',string(13+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, squeeze(ACCUMATRIX(t-sigma_l+1,:,:))', dataset, strcat('C',string(6+(t-sigma_l)*11),':Q',string(13+(t-sigma_l)*11)) );

        xlswrite(filename_xlsx, {strcat("sigma=",sigma_names(t-sigma_l+1))}, dataset, strcat('T',string(4+(t-sigma_l)*11)) );   
        xlswrite(filename_xlsx, {"eps\C1=C2"}, dataset, strcat('T',string(5+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, C_names, dataset, strcat('U',string(5+(t-sigma_l)*11),':AI',string(5+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, epsilon_names', dataset, strcat('T',string(6+(t-sigma_l)*11),':T',string(13+(t-sigma_l)*11)) );
        xlswrite(filename_xlsx, squeeze(BACMATRIX(t-sigma_l+1,:,:))', dataset, strcat('U',string(6+(t-sigma_l)*11),':AI',string(13+(t-sigma_l)*11)) );

    end

    % Tiempo utilizado en la validación cruzada
    tiempoTranscurrido = toc;  % Tiempo transcurrido desde tic hasta toc
    disp(['El tiempo utilizado es: ' num2str(tiempoTranscurrido) ' segundos']);

    % Almacena los resultados en la variable 'results'

    param = ["sigma", "C", "eps"]; % Nombre de los parámetros

    results.(char(dataset)).BACMATRIX = BACMATRIX;                      % Guardamos BACMATRIX  en 'results' 
    [maxBAC, linearIndex] = max(BACMATRIX(:));                          % Calculamos el máximo de BACMATRIX
    [t, i, j] = ind2sub(size(BACMATRIX), linearIndex);                  % Calculamos los índices del máximo de BACMATRIX
    value = [2^(t+sigma_l-1), 2^(i+C_l-1), 2^(j+eps_l-1)];              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([t+sigma_l-1, i+C_l-1, j+eps_l-1])); % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxBAC = table(param, value, valueName);    % Guardamos la informacion en una tabla

    results.(char(dataset)).ACCUMATRIX = ACCUMATRIX;                    % Guardamos ACCUMATRIX  en 'results' 
    [maxACCU, linearIndex] = max(ACCUMATRIX(:));                        % Calculamos el máximo de ACCUMATRIX
    [t, i, j] = ind2sub(size(ACCUMATRIX), linearIndex);                 % Calculamos los índices del máximo de ACCUMATRIX
    value = [2^(t+sigma_l-1), 2^(i+C_l-1), 2^(j+eps_l-1)];              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([t+sigma_l-1, i+C_l-1, j+eps_l-1])); % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxACCU = table(param, value, valueName);   % Guardamos la informacion en una tabla


end

% Guarda 'results' en un archivo .mat 
save(filename_mat, 'results');





