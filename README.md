# Neural Graph Collaborative Filtering-var
Proyecto final del curso Sistema de Recomendadores, de la Escuela de Ciencia de Computación, Pontifica Universidad Católica de Chile.
Autores: Mónica Cavieres, Julián García, Alexander Pinto.

## Abstract
Hoy en día los datos de cada individuo están cada vez más expuestos, y estos exigen ser tratados de manera responsable. Por otro lado, el concepto de fairness exige la igualdad en el tratamiento y en el impacto, por lo tanto los algoritmos no deben presentar discriminación en sus resultados. 
Nuestro trabajo propone un sistema recomendador denominado NGCF-var, el cual está basado en una red neuronal de grafos, que apunta a no tener un 
sesgo para los datos protegidos de edad y género. Para este propósito se introdujo en el modelo la métrica de varianza en la etapa de cálculo de pérdida, con la idea de reducir diferencias en los resultados de precisión para diferentes grupos protegidos. Se realizó una evaluación de nuestro modelo sobre el dataset de Movielens, teniendo como atributos protegidos el género y la edad de los usuarios.
Nuestros resultados preliminares no permiten dar una opinión concluyente sobre la idoneidad de la inclusión de la varianza, sin embargo nuestros resultados indican que es posible obtener resultados más estables y justos entre los grupos protegidos, además de una mejor cobertura de los ítemes recomendados.

Basada en la implementación de NGCF por:

>Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). Neural Graph Collaborative Filtering, [Paper in ACM DL](https://dl.acm.org/citation.cfm?doid=3331184.3331267) or [Paper in arXiv](https://arxiv.org/abs/1905.08108). In SIGIR'19, Paris, France, July 21-25, 2019.
Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.8.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).
* Gowalla dataset
```
python NGCF.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

* Movielens dataset
```
python NGCF.py --dataset ml100m --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 200 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

## Dataset
We provide movielens processed datasets.
* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.


## Provided notebooks

* Analisis_de_datos.ipynb: it analyses the used datasets and sees its distribution on the sensitive attributes of age and gender, to generate the graphs.
* preprocessing.py: it preprocesses the notebooks to be able to fit the required ngcf input format
* RecBole.py: it preprocesses the recbole input data and runs the metrics over the selected models Popularity and BPR to compare them to the ngcf and ngcf var models

WARNING: to change the importance of the variance loss, you must edit line 47 of NGCF.py directly, changing the number value as needed.

## Provided ablated datasets

* ml100_test_all: dataset with the complete testing set
* ml100_test_m: dataset with test set with only male users
* ml100_test_f: dataset with test set with only female users
* ml100_test_1_age: dataset with test set with only users under 18
* ml100_test_18_age: dataset with test set with only users from 18 to 24
* ml100_test_25_age: dataset with test set with only users from 25 to 34
* ml100_test_35_age: dataset with test set with only users from 35 to 44
* ml100_test_45_age: dataset with test set with only users from 45 to 49
* ml100_test_50_age: dataset with test set with only users from 50 to 55
* ml100_test_56_age: dataset with test set with only users 56+
