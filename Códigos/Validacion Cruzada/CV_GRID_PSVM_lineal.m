% Cross Validation PSVM (linear kernel)

clc
clear all

% Definición de los rangos de exploración para los parámetros C y epsilon
C_l = -7;     % Valor mínimo del exponente de 2 para C
C_h = 7;      % Valor máximo del exponente de 2 para C

eps_l = -7;   % Valor mínimo del exponente de 2 para epsilon
eps_h = 0;    % Valor máximo del exponente de 2 para epsilon

FunPara.kerfPara.type = 'lin'; % Tipo de kernel: lineal

% Nombre del archivo Excel donde se guardarán los resultados
filename_xlsx = 'CV_PSVM_lineal.xlsx';

% Nombres de las columnas para el parámetro C y epsilon en el archivo Excel
C_names = strcat('2^',string(C_l:C_h));
epsilon_names = strcat('2^',string(eps_l:eps_h));

% Nombre del archivo .mat donde se guardarán los resultados
filename_mat = "CV_PSVM_lineal.mat";

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

    % Escribe los nombres de las columnas en las celdas especificadas en el archivo Excel
    xlswrite(filename_xlsx, "Accu.", dataset, 'B3');
    xlswrite(filename_xlsx, "eps\C", dataset, 'B4');
    xlswrite(filename_xlsx, C_names, dataset, 'C4:Q4');
    xlswrite(filename_xlsx, epsilon_names', dataset, 'B5:B12');

    xlswrite(filename_xlsx, "Bal. Accu.", dataset, 'B15');
    xlswrite(filename_xlsx, "eps\C", dataset, 'B16');
    xlswrite(filename_xlsx, C_names, dataset, 'C16:Q16');
    xlswrite(filename_xlsx, epsilon_names', dataset, 'B17:B24');

    folds = max(CV_indices);  % Número de iteraciones para la validación cruzada
    X = data_train;           
    Y = labels_train;         

    % Inicialización de matrices para almacenar la exactitud (ACCU) y la
    % exatitud equilibrada (BAC)
    BACMATRIX=zeros(C_h-C_l+1, eps_h-eps_l+1);
    ACCUMATRIX=zeros(C_h-C_l+1, eps_h-eps_l+1);

    for i=C_l:C_h
        FunPara.C=2^i;
        for j=eps_l:eps_h
            fprintf('(%d, %d) \n',i,j);
            FunPara.epsi = 2^j;
            for k=1:folds
                    
                idx_test = (CV_indices == k);   % Índices de datos de prueba
                idx_train = ~idx_test;          % Índices de datos de entrenamiento
    
                X_train = X(idx_train, :); % Datos de entrenamiento
                Y_train = Y(idx_train);    % Etiquetas de entrenamiento
    
                X_test = X(idx_test, :); % Datos de prueba
                Y_test = Y(idx_test);    % Etiquetas de prueba
    
                Y_predic =  PSVM_quadprog(X_train,Y_train,X_test,FunPara);   % Predicciones via quadprog
    
                [BAC(k),ACCU(k)]=medi_auc_accu(Y_predic,Y_test); % Calcula BAC y ACCU para dichas predicciones
            
            end

            % Calcula y almacena el BAC y ACCU promedio
            BACMATRIX(i-C_l+1,j-eps_l+1)=mean(BAC);
            ACCUMATRIX(i-C_l+1,j-eps_l+1)=mean(ACCU);

        end

        % Escribe los resultados en el archivo Excel
        xlswrite(filename_xlsx, ACCUMATRIX(i-C_l+1,:)', dataset, strcat(char('C'+i-C_l),'5:',char('C'+i-C_l),'12') );
        xlswrite(filename_xlsx, BACMATRIX(i-C_l+1,:)', dataset, strcat(char('C'+i-C_l),'17:',char('C'+i-C_l),'24') );

    end

    % Tiempo utilizado en la validación cruzada
    tiempoTranscurrido = toc;  % Tiempo transcurrido desde tic hasta toc
    disp(['El tiempo utilizado es: ' num2str(tiempoTranscurrido) ' segundos']);


    % Almacena los resultados en la variable 'results'

    param = ["C", "eps"]; % Nombre de los parámetros

    results.(char(dataset)).BACMATRIX = BACMATRIX;                      % Guardamos BACMATRIX  en 'results' 
    [maxBAC, linearIndex] = max(BACMATRIX(:));                          % Calculamos el máximo de BACMATRIX
    [i, j] = ind2sub(size(BACMATRIX), linearIndex);                  % Calculamos los índices del máximo de BACMATRIX
    value = [ 2^(i+C_l-1), 2^(j+eps_l-1)];              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([i+C_l-1, j+eps_l-1])); % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxBAC = table(param, value, valueName);    % Guardamos la informacion en una tabla

    results.(char(dataset)).ACCUMATRIX = ACCUMATRIX;                    % Guardamos ACCUMATRIX  en 'results' 
    [maxACCU, linearIndex] = max(ACCUMATRIX(:));                        % Calculamos el máximo de ACCUMATRIX
    [i, j] = ind2sub(size(ACCUMATRIX), linearIndex);                 % Calculamos los índices del máximo de ACCUMATRIX
    value = [2^(i+C_l-1), 2^(j+eps_l-1)];              % Calculamos los valores correspondientes de los parámetros
    valueName = strcat("2^",string([i+C_l-1, j+eps_l-1])); % Cadena de texto con los valores correspondientes de los parámetros
    results.(char(dataset)).maxACCU = table(param, value, valueName);   % Guardamos la informacion en una tabla

    save(strcat("CV_PSVM_lineal_",dataset,".mat"))

end

% Guarda 'results' en un archivo .mat 
save(filename_mat, 'results');


