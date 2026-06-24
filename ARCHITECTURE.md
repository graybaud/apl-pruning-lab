# APL Pruning Lab — Architecture

## Vue d'ensemble
apl_pruning/
├── init.py # Public API
├── parser.py # MiniAPLParser (orchestration)
├── tokenizer.py # Lexer: code APL -> tokens
├── grammar.py # Parser: tokens -> AST
├── ast_evaluator.py # Evaluateur: AST -> numpy
├── exporter.py # AST -> code PyTorch
├── scorers.py # Registre de 55 methodes de pruning
├── layers.py # LayerScorer (multi-couche)
├── cache.py # LayerCache (optimisation)
└── utils.py # Safe math, broadcasting


## Flux de compilation
Code APL (string)
│
▼
tokenizer.py ──→ tokens [(TYPE, valeur), ...]
│
▼
grammar.py ──→ AST ('mul', ('abs', 'W'), ('func_with_axis', 'mean', ...))
│
├──→ ast_evaluator.py ──→ numpy array (evaluation directe)
│
└──→ exporter.py ──→ code PyTorch (export)



## Responsabilité de chaque module

### `tokenizer.py` — Lexer
Transforme une string APL en liste de tokens.

**Exemple :** `"|W| x mean(|act|)"` → `[('ABS', '|'), ('VAR', 'W'), ('ABS', '|'), ('OP', 'x'), ('FUNC', 'mean'), ...]`

**Règle :** ne contient AUCUNE logique mathématique. Juste du parsing de caractères.

### `grammar.py` — Parser
Transforme les tokens en AST (Abstract Syntax Tree).

**Exemple :** `tokens` → `('mul', ('abs', ('var', 'W')), ('func_with_axis', 'mean', ('abs', ('var', 'act')), 0))`

**Règle :** ne contient AUCUNE évaluation. Juste de la structuration.

### `ast_evaluator.py` — Évaluateur
Exécute l'AST avec des variables numpy.

**Règle :** ne dépend QUE de `numpy`. Aucune dépendance à PyTorch, transformers, etc.

### `exporter.py` — Exporteur
Traduit l'AST en code PyTorch exécutable.

**Exemple :** `ast` → `"torch.abs(W) * torch.mean(torch.abs(act), dim=0)"`

**Règle :** produit du code PyTorch sous forme de string. N'exécute rien.

### `scorers.py` — Registre de méthodes
55 formules de pruning prédéfinies, prêtes à l'emploi.

Chaque méthode a :
- `formula` : code APL (string simple ou multi-lignes)
- `needs_grad` : bool (True si nécessite des gradients)
- `description` : texte explicatif
- `variables` : liste des variables requises

**Règle :** ce fichier est un dictionnaire. Pas de logique, que des données.

### `layers.py` — LayerScorer
Applique une formule à plusieurs couches d'un modèle.

**Règle :** itère sur les couches. Délègue l'évaluation à `MiniAPLParser`.

### `cache.py` — LayerCache
Cache les résultats intermédiaires (ex: `|W|`, `mean(|act|)`) pour éviter de les recalculer.

## Principe fondamental

> **APL Pruning Lab est un LANGAGE, pas une bibliothèque de pruning.**

Il ne sait pas ce qu'est un modèle, un tokenizer, ou un GPU. Il manipule des formules mathématiques et des tableaux numpy. C'est CastNet qui lui donne un sens "métier" en lui passant des poids et des activations.
