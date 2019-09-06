# graph_matching

## But - Ce qu'on veut faire

A partir de graphes de pits du cerveau,on cherche à établir des
 correspondances entre les pits de cerveaux de multiples patients 
(134 dans nos données). On cherche à faire un appareillage multiple mais partiel
(tous les sommets n'auront pas forcément de correspondances).

## Théorie

### Kernelized Sorting

https://papers.nips.cc/paper/3608-kernelized-sorting

Le Kernelized Sorting permet l'appariement de 2 objets (ou plus) de types 
différents n'ayant pas forcément la même structure. 
Cette méthode s'appuie sur le critère d'indépendance d'Hilbert Schmidt 
(HSIC) qui permet de mesurer la dépendance entre 2 sets de données X et Y de taille égale
 à m qui seront représentés par leurs matrices noyaux K et L. 
On cherche alors une matrice de permutation P qui permet de passer de K à L 
qui maximise ce critère.
Le problème se résume à :

![alt text](readme_image/simple_argmax.png)

Une fois la matrice P trouvée, on a P[i,j] = 1 si Y[i] correspond à X[j] 

Il existe une formule pour ce critère pour plusieurs ensembles, 
le problème revient à maximiser :

![alt text](readme_image/multi.png)

On minimisera donc son inverse.

### Convex Kernelized Sorting

http://www.dabi.temple.edu/~vucetic/documents/djuric12aaai.pdf

Le Convex Kernelized Sorting réecrit le KS en un problème équivalent, plus simple à résoudre :

![alt text](readme_image/convex.png)

Pour l'appariement de 2 objets, nous avons résolu ce problème.

### Descente du gradient 

Pour trouver des solutions aux problèmes situés plus haut, 
on utilise une descente du gradient projetée. 
Le projection met les valeurs de la matrice de permutation à 0 si ses valeurs sont négatives
. 

### Branch and bound

https://hal.inria.fr/hal-00881407/file/PermutationIcassp.pdf

Le problème de le descente du gradient est qu'on peut facilement tomber sur des optima locaux 
et non sur la solution. On trouve des résultats différents selon l'initialisation...

Le branch and bound est une méthode qui nous permet de trouver une solution optimale, mais qui a le défaut 
d'être beaucoup plus longue car dans le pire des cas, on peut se retrouver à explorer l'ensemble
des permutations.

On peut l'appliquer seulement pour des petites matrices (taille<20).


## Implémentation

#data

Contient les données (les graphes)

#hsic
- convex_simple_hsic.py contient les méthodes utiles pour la création des fonctions objectif,
gradient et la descente du gradient pour l'appariement de 2 ensembles
- convex_multi_hsic.py contient les méthodes utiles pour la création des fonctions objectif,
gradient et la descente du gradient pour l'appariement de plusieurs ensembles
- branch_and_bound.py contient les méthodes utiles pour l'algorithme du branch and bound (comparaison de 2 graphes)

#minimization
- convexminimization2.py contient les méthodes utilisées pour la descente du gradient

#multiple matching
- multiway_matching_graph.py est le programme qui charge les graphes et permet de tous les comparer à l'aide du hsic multiple puis affiche les résultats sur une sphere
- matching_on_pt_of_interests.py est le programme permettant de comparer les 134 autour de 2 points d'interet différents

#two_graph_matching
- matching_two_graphs.py est le programme qui charge les graphes et permet d'en choisir 2 à comparer puis affiche les résultats sur une sphere
- matching_2_graph_on_pt_on_interest.py  est le programme permettant de comparer 2 graphes choisis autour de 2 points d'intérêts différents

#others tests 
tests plus très utiles
#tools
- show_results_on_sphere.py contient les méthodes utiles à la visualisation des résultats
- branch_and_bound.py contient les méthodes utiles pour l'algorithme du branch and bound
- metric.py contient les méthodes utiles pour calculer les moyennes des distances des appareillages
- test_branch.py est le programme contenant un test du branch and bound pour la comparaison de 2 graphes (à un point d'intérêt)
- load_graph_and_kernel.py contient les méthodes utiles pour la récupération des données (graphes + matrices de Gram)



## Prérequis

- matplotlib
- trimesh
- numpy
- networkx
- pickle



