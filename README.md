# Coocurrences
Il s'agit d'un programme qui produit à partir d'un fichier texte une matrice de co-occurrences entre des formes. L'utilisateur doit renseigner le nom du fichier, la taille de la fenêtre pour calculer les co-occurrences, ou bien les ponctuations permettant d'établir cette fenêtre. Le programme parse le texte, isole les formes  et produit des matrices de coocurrences en python, afin de pouvoir les utiliser dans un objectif de traitement temporel.

## Les librairies utilisées 
* Scipy 1.6.2
* Numpy 1.20.3
* Pandas 1.3.0
* Scikit-Learn 0.24.2
* Regex 2021.7.6
* mpmath

## Spécifications

* **Entrées** : Répertoire de textes par dates, mais vous pouvez mettre un seul texte
* **Sorties** : Matrice dont le type est défini par l'utilisateur (dirigée, no dirigée, TF-IDF, Counter)
Métrique de spécificité définie par l'utilisateur (hypergeometrique, dice, cosine, log-likelihood, autres)
* **Autres spécifications** : Gestion des stop-words, gestion des fenêtres, optimisation des ressources mémoire et computationnelle.


## Sous Windows
Il faut installer une version de [python](https://www.python.org/downloads/windows/) supérieure à 3. Ensuite installer les bibliothèques dans shell(PowerShell par exemple) avec pip3. 
Vous pouvez par exemple installer numpy de cette façon : 
```python
pip3 install numpy
```


## Quelques éléments

* La classe **matrix_generator.py**  contient une suite d'opérations nécessaires à la construction de la matrice de coocurrence dirigée ou non. 
	- MatrixGenerator()  possède un assez grand nombre de méthodes 
		* .get_vocab()
			- > permet de déterminer les fenêtres (phrases, paragraphes, nombre de formes) dans l'espace desquelles on extrait les coocurrents 
		* .counter_full()
			- > calcul le dictionnaire des fréquences de toutes les occurrences
		* .counter_by_window() 
			- > calcul le dictionnaire des fréquences par fenêtre, les formes ont un index
		* .get_directed() 
			- > calcul la matrice dirigée et possède un paramètre permettant d'évaluer les coocurrents spécifiques à des formes pôles, il s'agit de :  option="". 
				* hypergeometrique
				* cosinus
				* dice

