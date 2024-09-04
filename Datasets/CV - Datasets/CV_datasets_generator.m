dataset = 'BreastMNIST';
% dataset = 'DermaMNIST_0vs2';
% dataset = 'DermaMNIST_0vs4';

load(strcat(dataset,'.mat'))

CV_indices = crossvalind('Kfold', size(data_train, 1), 10); % dividir los datos en K subconjuntos

clear data_test labels_test

save(strcat('CV_',dataset,'.mat'), "CV_indices","data_train","labels_train")

