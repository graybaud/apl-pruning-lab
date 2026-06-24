# Contribuer a APL Pruning Lab

## Ajouter une nouvelle formule de pruning

### Niveau 1 : Formule simple (1 ligne)

Ajoutez votre méthode dans `apl_pruning/scorers.py` :

```python
METHODS.update({
    "ma_methode": {
        "formula": "|W| x log(|grad| + 1)",
        "needs_grad": True,
        "description": "Ma methode: |W| x log(|grad| + 1)",
        "variables": ["W", "grad"],
    },
})
```
Niveau 2 : Formule multi-lignes
```python
METHODS.update({
    "ma_methode_complexe": {
        "formula": """
            a <- max(|W|, dim=-1) / mean(|W|, dim=-1)
            b <- max(|grad|, dim=-1) / mean(|grad|, dim=-1)
            c <- mean(|act|, dim=0)
            a x b x c
        """,
        "needs_grad": True,
        "description": "Triple ratio W x grad x act",
        "variables": ["W", "grad", "act"],
    },
})
```
Niveau 3 : Fonction Python (si vraiment necessaire)
Si la formule ne peut pas s'exprimer en APL, ajoutez une fonction dans apl_pruning/utils.py puis referencez-la dans l'evaluateur.

Ajouter une nouvelle primitive APL
Ajoutez le token dans tokenizer.py (si nouveau symbole)

Ajoutez la regle de parsing dans grammar.py

Ajoutez l'evaluation dans ast_evaluator.py

Ajoutez l'export PyTorch dans exporter.py

Ajoutez les tests dans tests/

Structure des tests

tests/
├── test_parser.py       # Parsing de formules
├── test_scorers.py      # Evaluation des methodes du registre
├── test_exporter.py     # Export PyTorch
├── test_layers.py       # LayerScorer
└── test_robustness.py   # Edge cases (empty tensors, NaN, etc.)
Checklist pour une PR
La formule est dans scorers.py avec formula, needs_grad, description, variables

Les variables declarees dans variables sont bien celles utilisees dans la formule

needs_grad est correct (True si la formule utilise grad)

Tests ajoutes dans tests/test_scorers.py

Tous les tests passent : pytest tests/ -v

La formule parse sans erreur : python -c "from apl_pruning import MiniAPLParser; p=MiniAPLParser(); p.parse('...')"
