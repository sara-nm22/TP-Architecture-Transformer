# TP – Architecture Transformer

TP de Pr. Mohamed Youssfi (ENSET Mohammedia) : de la prédiction de séries temporelles avec RNN / LSTM / Transformer jusqu'aux modèles de type GPT et BERT, en PyTorch.

Tout le travail est dans le notebook [`architecture_transformer.ipynb`](architecture_transformer.ipynb).

## Contenu

**Partie 1**
- Série temporelle `y = sin(x)` ; prédiction de la valeur suivante à partir de 10 valeurs
- RNN, LSTM, Transformer (encodeur) : création, entraînement, test
- GPT-2 préentraîné : génération de la suite d'un texte
- BERT multilingue préentraîné : prédiction d'un mot masqué

**Partie 2 (exercices)**
1. **Consommation électrique** : `EnergyTransformer` (fenêtre de 24 h), comparaison avec la persistance et la référence journalière, essai avec 168 h
2. **Mini-GPT** : Transformer causal entraîné depuis zéro sur des caractères (`corpus.txt`), génération avec températures 0,5 / 1,0 / 1,5, vérification du masque causal
3. **Mini-BERT** : modèle de langage masqué (15 % / 80-10-10), masque de padding, top-3 des propositions

## Lancer le projet

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook architecture_transformer.ipynb
```

La première exécution des cellules GPT / BERT télécharge les modèles `gpt2` et `bert-base-multilingual-cased` (connexion Internet nécessaire).

## Données de l'exercice 1

Le fichier `household_power_consumption.txt` (~127 Mo) **n'est pas inclus** dans le dépôt (limite GitHub de 100 Mo, il est dans le `.gitignore`).

1. Télécharger le dataset *Individual Household Electric Power Consumption* sur l'[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2. Décompresser l'archive et placer `household_power_consumption.txt` à côté du notebook.

Les sorties de l'exercice 1 s'affichent à l'exécution du notebook une fois le fichier en place.
