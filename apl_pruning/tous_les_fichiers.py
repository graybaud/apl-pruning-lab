#!/usr/bin/env python3
"""
APL-Python DSL Compiler - Version Fusionnée 2.0
================================================
Un compilateur DSL inspiré d'APL pour le calcul array-oriented en Python.
Fusion des meilleures idées des deux architectures analysées.

Architecture :
    Code APL → PEG Parser → IR Syntaxique → Lowering → IR Sémantique
    → Fusion Optimizer → Backend (JAX/Numba/PyTorch/NumPy)

Auteur : Synthèse des projets A et B
Licence : MIT
"""

from __future__ import annotations

import operator
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union
)

import numpy as np

# ==============================================================================
# 0. Configuration et Dépendances Optionnelles
# ==============================================================================

_HAS_JAX = False
_HAS_NUMBA = False
_HAS_TORCH = False
_HAS_PARSIMONIOUS = False

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    pass

try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    pass

try:
    import torch
    import torch.nn.functional as F  # noqa: F401
    _HAS_TORCH = True
except ImportError:
    pass

try:
    from parsimonious.grammar import Grammar
    from parsimonious.nodes import NodeVisitor
    _HAS_PARSIMONIOUS = True
except ImportError:
    pass


# ==============================================================================
# 1. Type Fondamental : APLArray Hybride
# ==============================================================================
# Inspiré du Projet B : encapsule tableaux plats (délégués à NumPy/JAX) et
# tableaux imbriqués (gérés en interne). Supporte le buffer protocol pour
# l'interopérabilité zéro-copie.

class APLArray:
    """
    Type de données fondamental pour le DSL APL.

    Caractéristiques :
    - Tableaux plats : buffer protocol pour zéro copie avec NumPy/PyTorch
    - Tableaux imbriqués : liste d'APLArray
    - Conversion paresseuse entre représentations
    - Inférence automatique de la forme et du rang

    Examples:
        >>> APLArray(42)                     # Scalaire
        APLArray(scalar, data=42)

        >>> APLArray(np.array([1,2,3]))      # Vecteur NumPy
        APLArray(shape=(3,), data=[1 2 3])

        >>> APLArray([1, [2,3], 4])          # Imbriqué
        APLArray(nested, shape=(3,))
    """

    def __init__(
        self,
        data: Union[int, float, np.ndarray, list, 'APLArray'],
        shape: Optional[Tuple[int, ...]] = None,
        dtype=None,
    ):
        # Cas 0 : Scalaire Python → tableau 0-dimensionnel
        if isinstance(data, (int, float, np.integer, np.floating)):
            self._flat_data: Optional[np.ndarray] = np.array(data, dtype=dtype)
            self._nested_data: Optional[List[APLArray]] = None
            self._is_flat: bool = True
            self._is_jax: bool = False
            self._jax_data: Any = None

        # Cas 1 : Tableau NumPy → zéro copie (buffer protocol)
        elif isinstance(data, np.ndarray):
            self._flat_data = data if dtype is None else data.astype(dtype)
            self._nested_data = None
            self._is_flat = True
            self._is_jax = False
            self._jax_data = None

        # Cas 2 : Tableau JAX (si disponible)
        elif _HAS_JAX and isinstance(data, jnp.ndarray):
            self._flat_data = np.asarray(data)
            self._jax_data = data
            self._nested_data = None
            self._is_flat = True
            self._is_jax = True

        # Cas 3 : Liste Python → tableau imbriqué APL
        elif isinstance(data, list):
            self._flat_data = None
            self._nested_data = [APLArray(item) for item in data]
            self._is_flat = False
            self._is_jax = False
            self._jax_data = None

        # Cas 4 : Déjà un APLArray → référence partagée
        elif isinstance(data, APLArray):
            self._flat_data = data._flat_data
            self._jax_data = data._jax_data
            self._nested_data = data._nested_data
            self._is_flat = data._is_flat
            self._is_jax = data._is_jax

        else:
            raise TypeError(
                f"Cannot create APLArray from {type(data).__name__}"
            )

        # Calcul de la forme
        self._shape: Tuple[int, ...] = (
            shape if shape is not None else self._compute_shape()
        )
        self._rank: int = len(self._shape)

    def _compute_shape(self) -> Tuple[int, ...]:
        """Calcule la forme APL du tableau."""
        if self._is_flat and self._flat_data is not None:
            return self._flat_data.shape
        elif self._nested_data is not None:
            return (len(self._nested_data),)
        return ()

    # ---- Propriétés ----

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        if self._is_flat and self._flat_data is not None:
            return self._flat_data.size
        elif self._nested_data is not None:
            return len(self._nested_data)
        return 1

    @property
    def is_simple(self) -> bool:
        """Tableau simple (non imbriqué, type non objet)."""
        if not self._is_flat or self._flat_data is None:
            return False
        return self._flat_data.dtype != object

    @property
    def is_scalar(self) -> bool:
        return self._rank == 0

    # ---- Conversion vers l'écosystème Python ----

    def __array__(self, dtype=None) -> np.ndarray:
        """Pour np.asarray() - peut nécessiter une copie."""
        if self._is_flat and self._flat_data is not None:
            return np.asarray(self._flat_data, dtype=dtype)
        elif self._nested_data is not None:
            return np.array(
                [np.asarray(item) for item in self._nested_data],
                dtype=object,
            )
        raise ValueError("Empty APLArray")

    def numpy(self) -> np.ndarray:
        """Vue NumPy zéro-copie (erreur si imbriqué)."""
        if not self._is_flat or self._flat_data is None:
            raise ValueError("Cannot create flat view of nested APLArray")
        return self._flat_data

    def jax(self) -> Any:
        """Tableau JAX (copie si nécessaire)."""
        if not _HAS_JAX:
            raise ImportError("JAX is not installed")
        if self._is_jax and self._jax_data is not None:
            return self._jax_data
        return jnp.asarray(
            self._flat_data if self._is_flat else self.__array__()
        )

    def tolist(self) -> list:
        """Conversion en liste Python."""
        if self._is_flat and self._flat_data is not None:
            return self._flat_data.tolist()
        elif self._nested_data is not None:
            return [item.tolist() for item in self._nested_data]
        return []

    # ---- Buffer Protocol (zéro copie) ----

    def __buffer__(self, flags: int) -> memoryview:
        if not self._is_flat or self._flat_data is None:
            raise BufferError("Nested arrays don't support buffer protocol")
        return memoryview(self._flat_data)

    # ---- Opérations de base ----

    def __getitem__(self, idx):
        """Indexation 0-based (configurable via ⎕IO)."""
        if self._is_flat and self._flat_data is not None:
            result = self._flat_data[idx]
            if isinstance(result, np.ndarray):
                return APLArray(result)
            return APLArray(float(result))
        elif self._nested_data is not None:
            return self._nested_data[idx]
        raise IndexError("Empty array")

    def __repr__(self) -> str:
        if self.is_scalar:
            val = (
                self._flat_data.item()
                if self._flat_data is not None
                else "?"
            )
            return f"APLArray(scalar, data={val})"
        if self._is_flat and self._flat_data is not None:
            return (
                f"APLArray(shape={self._shape}, "
                f"data={self._flat_data})"
            )
        return f"APLArray(nested, shape={self._shape})"

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 0

    # ---- Opérateurs mathématiques ----

    def __neg__(self) -> 'APLArray':
        if self._is_flat and self._flat_data is not None:
            return APLArray(-self._flat_data)
        return APLArray([-x for x in self._nested_data]) if self._nested_data else self

    def __abs__(self) -> 'APLArray':
        if self._is_flat and self._flat_data is not None:
            return APLArray(np.abs(self._flat_data))
        return APLArray([abs(x) for x in self._nested_data]) if self._nested_data else self

    def __add__(self, other: 'APLArray') -> 'APLArray':
        return APLArray(self.numpy() + other.numpy())

    def __sub__(self, other: 'APLArray') -> 'APLArray':
        return APLArray(self.numpy() - other.numpy())

    def __mul__(self, other: 'APLArray') -> 'APLArray':
        return APLArray(self.numpy() * other.numpy())

    def __truediv__(self, other: 'APLArray') -> 'APLArray':
        denom = other.numpy()
        if np.any(np.abs(denom) < 1e-12):
            warnings.warn("Division by near-zero detected")
        return APLArray(self.numpy() / denom)

    def __pow__(self, other: 'APLArray') -> 'APLArray':
        return APLArray(self.numpy() ** other.numpy())


# ==============================================================================
# 2. Grammaire PEG Formelle
# ==============================================================================
# Du Projet A : grammaire non ambiguë avec précédence explicite,
# commentaires, et réductions généralisées.

APL_PEG_GRAMMAR_SOURCE = r"""
    # ============================================================
    # Top-level
    # ============================================================
    expr        = (comment / wsp)* (assignment / expression)

    # ============================================================
    # Commentaires (⍝ et //)
    # ============================================================
    comment     = lamp_comment / slash_comment
    lamp_comment = "\u235d" ~r"[^\n]*"
    slash_comment = "//" ~r"[^\n]*"

    # ============================================================
    # Assignment: name ← expression
    # ============================================================
    assignment  = identifier wsp arrow wsp expression
    arrow       = "<-" / "\u2190"

    # ============================================================
    # Expression (précédence croissante)
    # ============================================================
    expression  = sum

    # Addition / soustraction (gauche, précédence 1)
    sum         = product (wsp ("+" / "-") wsp product)*

    # Multiplication / division (gauche, précédence 2)
    product     = unary (wsp ("x" / "/") wsp unary)*

    # Puissance (droite, précédence 3)
    power       = unary (wsp "^" wsp power)?

    # ============================================================
    # Opérateurs unaires (précédence 4)
    # ============================================================
    unary       = negation / absolute / power

    negation    = "-" wsp unary
    absolute    = "|" wsp expression wsp "|"

    # ============================================================
    # Atomes
    # ============================================================
    primary     = number / function_call / reduction / variable / grouped

    grouped     = "(" wsp expression wsp ")"
    number      = ~r"\d+\.?\d*"

    # ============================================================
    # Variables avec indexation optionnelle
    # ============================================================
    variable    = identifier indexing?
    indexing    = "[" wsp index_list wsp "]"
    index_list  = index_spec (wsp "," wsp index_spec)*
    index_spec  = slice_spec / expression
    slice_spec  = expression? wsp ":" wsp expression?

    # ============================================================
    # Appels de fonction
    # ============================================================
    function_call = identifier wsp "(" wsp arg_list? wsp ")"
    arg_list    = argument (wsp "," wsp argument)*
    argument    = kwarg / expression
    kwarg       = identifier wsp "=" wsp expression

    # ============================================================
    # Réductions (généralisées)
    # ============================================================
    reduction   = reduction_op wsp unary
    reduction_op = "+/" / "x/" / "max/" / "min/"

    # ============================================================
    # Identifiants
    # ============================================================
    identifier  = ~r"[a-zA-Z_]\w*"

    # Whitespace
    wsp         = ~r"\s*"
"""

# Compiler la grammaire si parsimonious est disponible
APL_GRAMMAR = None
if _HAS_PARSIMONIOUS:
    try:
        APL_GRAMMAR = Grammar(APL_PEG_GRAMMAR_SOURCE)
    except Exception:
        pass


# ==============================================================================
# 3. IR Syntaxique (Niveau 1)
# ==============================================================================
# Du Projet A : dataclasses typées avec inférence de formes.

# Alias de type
Shape = Optional[Tuple[Optional[int], ...]]


@dataclass
class Literal:
    value: Union[int, float]
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class Variable:
    name: str
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class UnaryOp:
    op: str  # 'neg', 'abs'
    operand: 'Expr'
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class BinaryOp:
    op: str  # 'add', 'sub', 'mul', 'div', 'pow'
    left: 'Expr'
    right: 'Expr'
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class Index:
    variable: str
    index: 'Expr'
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class IndexMulti:
    variable: str
    indices: List[Union['Expr', 'SliceSpec']]
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class SliceSpec:
    start: Optional['Expr'] = None
    end: Optional['Expr'] = None


@dataclass
class FunctionCall:
    name: str
    args: List['Expr'] = field(default_factory=list)
    kwargs: Dict[str, 'Expr'] = field(default_factory=dict)
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class Reduction:
    op: str  # 'sum', 'product', 'max', 'min'
    operand: 'Expr'
    shape: Shape = field(default=None, repr=False)
    dtype: Optional[np.dtype] = field(default=None, repr=False)


@dataclass
class Assignment:
    name: str
    expression: 'Expr'


@dataclass
class Program:
    """Multi-lignes avec assignations."""
    statements: List['Expr'] = field(default_factory=list)


# Type union pour l'IR syntaxique
Expr = Union[
    Literal, Variable, UnaryOp, BinaryOp,
    Index, IndexMulti, FunctionCall, Reduction,
    Assignment, Program,
]


# ==============================================================================
# 4. IR Sémantique (Niveau 2)
# ==============================================================================
# Inspiré du Projet B : capture l'intention opérationnelle,
# pas la syntaxe. Permet la fusion de boucles.

@dataclass
class ScalarFunction:
    """Fonction scalaire primitive (+, -, ×, ÷, etc.)."""
    op: str
    _impl: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        self._impl = self._get_impl()

    def _get_impl(self) -> Callable:
        impls = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '×': lambda a, b: a * b,
            '÷': lambda a, b: a / b,
            '*': lambda a, b: a ** b,
            '⍟': lambda a, b: np.log(b) / np.log(a) if a > 0 else np.nan,
            '⌈': lambda a, b: np.maximum(a, b),
            '⌊': lambda a, b: np.minimum(a, b),
            '|': lambda a, b: np.abs(a) % np.abs(b),
        }
        return impls.get(self.op, lambda a, b: a + b)

    def __call__(self, a, b):
        return self._impl(a, b)


@dataclass
class Iota:
    """⍳N - Génération de séquence (pas un appel arange)."""
    bound: 'SExpr'
    shape: Shape = field(default=None, repr=False)


@dataclass
class Map:
    """Application élément par élément d'une fonction scalaire."""
    function: ScalarFunction
    array: 'SExpr'
    shape: Shape = field(default=None, repr=False)


@dataclass
class Reduce:
    """f/ - Réduction avec fonction et axe."""
    function: ScalarFunction
    array: 'SExpr'
    axis: int = -1
    shape: Shape = field(default=None, repr=False)


@dataclass
class Scan:
    """f\\ - Scan cumulatif."""
    function: ScalarFunction
    array: 'SExpr'
    axis: int = -1
    shape: Shape = field(default=None, repr=False)


@dataclass
class OuterProduct:
    """∘.f - Produit extérieur."""
    function: ScalarFunction
    left: 'SExpr'
    right: 'SExpr'
    shape: Shape = field(default=None, repr=False)


@dataclass
class Compose:
    """Composition de fonctions (f g h)."""
    functions: List['SExpr']
    shape: Shape = field(default=None, repr=False)


# ---- Nœuds Fusionnés (le cœur de l'optimisation) ----

@dataclass
class IotaReduce:
    """+/⍳N → boucle unique sans allocation."""
    function: ScalarFunction
    bound: 'SExpr'
    shape: Shape = field(default=None, repr=False)


@dataclass
class MapReduce:
    """f/ g¨ A → Map+Reduce en une seule passe."""
    map_function: ScalarFunction
    reduce_function: ScalarFunction
    array: 'SExpr'
    axis: int = -1
    shape: Shape = field(default=None, repr=False)


@dataclass
class IotaTable:
    """(f ⍳N) ∘.g (⍳M) → table sans allocations intermédiaires."""
    function: ScalarFunction
    rows: 'SExpr'
    cols: 'SExpr'
    shape: Shape = field(default=None, repr=False)


# Type union pour l'IR sémantique
SExpr = Union[
    Expr, Iota, Map, Reduce, Scan, OuterProduct, Compose,
    IotaReduce, MapReduce, IotaTable,
]


# ==============================================================================
# 5. Visitor PEG → IR Syntaxique
# ==============================================================================

class APLVisitor(NodeVisitor):
    """Transforme l'arbre de syntaxe concret (PEG) en IR syntaxique."""

    # ---- Top-level ----

    def visit_expr(self, node, visited_children):
        return visited_children[-1]  # Dernier enfant non-commentaire

    def visit_assignment(self, node, visited_children):
        name, _, _, _, expr = visited_children
        return Assignment(name=name, expression=expr)

    def visit_arrow(self, node, visited_children):
        return "<-"

    def visit_expression(self, node, visited_children):
        return visited_children[0]

    # ---- Binaires ----

    def visit_sum(self, node, visited_children):
        left, rest = visited_children[0], visited_children[1]
        for op, _, _, right in rest:
            left = BinaryOp(
                op='add' if op == '+' else 'sub',
                left=left, right=right,
            )
        return left

    def visit_product(self, node, visited_children):
        left, rest = visited_children[0], visited_children[1]
        for op, _, _, right in rest:
            left = BinaryOp(
                op='mul' if op == 'x' else 'div',
                left=left, right=right,
            )
        return left

    def visit_power(self, node, visited_children):
        left, rest = visited_children
        if rest:
            _, _, right = rest[0]
            return BinaryOp(op='pow', left=left, right=right)
        return left

    # ---- Unaires ----

    def visit_unary(self, node, visited_children):
        return visited_children[0]

    def visit_negation(self, node, visited_children):
        _, _, operand = visited_children
        return UnaryOp(op='neg', operand=operand)

    def visit_absolute(self, node, visited_children):
        _, _, operand, _, _ = visited_children
        return UnaryOp(op='abs', operand=operand)

    # ---- Primaires ----

    def visit_primary(self, node, visited_children):
        return visited_children[0]

    def visit_grouped(self, node, visited_children):
        _, _, expr, _, _ = visited_children
        return expr

    def visit_number(self, node, visited_children):
        text = node.text
        return Literal(
            value=float(text) if '.' in text else int(text),
        )

    # ---- Variables et indexation ----

    def visit_variable(self, node, visited_children):
        name, indexing = visited_children
        if indexing:
            indices = indexing
            if len(indices) == 1 and not isinstance(indices[0], SliceSpec):
                return Index(variable=name, index=indices[0])
            return IndexMulti(variable=name, indices=indices)
        return Variable(name=name)

    def visit_indexing(self, node, visited_children):
        _, _, indices, _, _ = visited_children
        return indices

    def visit_index_list(self, node, visited_children):
        first, rest = visited_children[0], visited_children[1]
        result = [first]
        for _, _, _, idx in rest:
            result.append(idx)
        return result

    def visit_index_spec(self, node, visited_children):
        return visited_children[0]

    def visit_slice_spec(self, node, visited_children):
        start, _, _, end = visited_children
        return SliceSpec(
            start=start[0] if start else None,
            end=end[0] if end else None,
        )

    # ---- Fonctions ----

    def visit_function_call(self, node, visited_children):
        name, _, _, _, args, _, _ = visited_children
        if args is None:
            return FunctionCall(name=name, args=[])
        pos_args = [a for a in args if not isinstance(a, dict)]
        kw_args = {}
        for a in args:
            if isinstance(a, dict):
                kw_args.update(a)
        return FunctionCall(name=name, args=pos_args, kwargs=kw_args)

    def visit_arg_list(self, node, visited_children):
        first, rest = visited_children[0], visited_children[1]
        result = [first]
        for _, _, _, arg in rest:
            result.append(arg)
        return result

    def visit_argument(self, node, visited_children):
        return visited_children[0]

    def visit_kwarg(self, node, visited_children):
        name, _, _, _, value = visited_children
        return {name: value}

    # ---- Réductions ----

    def visit_reduction(self, node, visited_children):
        op_token, _, operand = visited_children
        op_map = {'+/': 'sum', 'x/': 'product', 'max/': 'max', 'min/': 'min'}
        return Reduction(op=op_map.get(op_token, 'sum'), operand=operand)

    def visit_reduction_op(self, node, visited_children):
        return node.text

    # ---- Identifiants ----

    def visit_identifier(self, node, visited_children):
        return node.text.strip()

    # ---- Fallback ----

    def generic_visit(self, node, visited_children):
        return visited_children or node


# ==============================================================================
# 6. Optimiseur de Fusion
# ==============================================================================
# Inspiré du Projet B : réécritures pour éliminer les tableaux intermédiaires.

class FusionOptimizer:
    """
    Applique des réécritures de fusion sur l'IR sémantique.

    Règles :
    - Reduce(ScalarFn, Iota) → IotaReduce
    - Reduce(ScalarFn, Map(ScalarFn, A)) → MapReduce
    - OuterProduct(ScalarFn, Iota, Iota) → IotaTable
    """

    def __init__(self):
        self.rules = [
            self._rule_iota_reduce,
            self._rule_map_reduce,
            self._rule_iota_table,
        ]

    def optimize(self, ir: SExpr) -> SExpr:
        """Applique les règles jusqu'à point fixe."""
        changed = True
        optimized = ir
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            for rule in self.rules:
                result = rule(optimized)
                if result is not None and result is not optimized:
                    optimized = result
                    changed = True
                    break
            iterations += 1

        return optimized

    def _rule_iota_reduce(self, ir: SExpr) -> Optional[SExpr]:
        """+/⍳N → IotaReduce(+)"""
        if isinstance(ir, Reduce):
            if isinstance(ir.array, Iota) and isinstance(ir.function, ScalarFunction):
                return IotaReduce(
                    function=ir.function,
                    bound=ir.array.bound,
                    shape=(),
                )
        return None

    def _rule_map_reduce(self, ir: SExpr) -> Optional[SExpr]:
        """+/ ×¨ A → MapReduce(×, +)"""
        if isinstance(ir, Reduce):
            if isinstance(ir.array, Map) and isinstance(ir.function, ScalarFunction):
                return MapReduce(
                    map_function=ir.array.function,
                    reduce_function=ir.function,
                    array=ir.array.array,
                    axis=ir.axis,
                    shape=(),
                )
        return None

    def _rule_iota_table(self, ir: SExpr) -> Optional[SExpr]:
        """(f ⍳N) ∘.g (⍳M) → IotaTable(g)"""
        if isinstance(ir, OuterProduct):
            if isinstance(ir.left, Iota) and isinstance(ir.right, Iota):
                return IotaTable(
                    function=ir.function,
                    rows=ir.left.bound,
                    cols=ir.right.bound,
                )
        return None


# ==============================================================================
# 7. Vérificateur de Formes
# ==============================================================================
# Du Projet A : validation statique avec messages d'erreur APL.

class APLShapeError(Exception):
    """Erreur de forme avec message APL-style."""

    def __init__(
        self, message: str,
        left_shape: Shape = None,
        right_shape: Shape = None,
        axis: int = None,
        max_axis: int = None,
    ):
        self.left_shape = left_shape
        self.right_shape = right_shape
        self.axis = axis
        self.max_axis = max_axis
        super().__init__(self._format(message))

    def _format(self, message: str) -> str:
        parts = [f"APL SHAPE ERROR: {message}"]
        if self.left_shape is not None:
            parts.append(
                f"  left shape:  {self._fmt_shape(self.left_shape)}"
            )
        if self.right_shape is not None:
            parts.append(
                f"  right shape: {self._fmt_shape(self.right_shape)}"
            )
        if self.axis is not None:
            parts.append(f"  axis: {self.axis}")
            if self.max_axis is not None:
                parts.append(f"  max axis: {self.max_axis}")
        return '\n'.join(parts)

    @staticmethod
    def _fmt_shape(s: Shape) -> str:
        if s is None:
            return '?'
        return ' '.join(str(d) if d is not None else '?' for d in s)


def infer_shapes(ir: Expr, known_shapes: Dict[str, Shape]) -> Dict[str, Shape]:
    """Infère les formes de toutes les sous-expressions."""
    _infer_recursive(ir, known_shapes)
    return known_shapes


def _infer_recursive(ir: Expr, shapes: Dict[str, Shape]):
    """Inférence récursive avec validation."""
    if isinstance(ir, Literal):
        ir.shape = ()
        ir.dtype = np.float32

    elif isinstance(ir, Variable):
        if ir.name not in shapes:
            raise APLShapeError(
                f"VARIABLE '{ir.name}' has no known shape. "
                f"Did you forget to call set_variables()?"
            )
        ir.shape = shapes[ir.name]
        ir.dtype = np.float32

    elif isinstance(ir, UnaryOp):
        _infer_recursive(ir.operand, shapes)
        ir.shape = ir.operand.shape
        ir.dtype = ir.operand.dtype

    elif isinstance(ir, BinaryOp):
        _infer_recursive(ir.left, shapes)
        _infer_recursive(ir.right, shapes)
        ir.shape = _broadcast_shapes(ir.left.shape, ir.right.shape)
        ir.dtype = np.float32

    elif isinstance(ir, FunctionCall):
        for arg in ir.args:
            _infer_recursive(arg, shapes)
        for val in ir.kwargs.values():
            _infer_recursive(val, shapes)
        if ir.args:
            ir.shape = _infer_function_shape(ir.name, ir.args, ir.kwargs)
        ir.dtype = np.float32

    elif isinstance(ir, Reduction):
        _infer_recursive(ir.operand, shapes)
        ir.shape = ()
        ir.dtype = np.float32


def _broadcast_shapes(s1: Shape, s2: Shape) -> Shape:
    """Calcule la forme broadcastée (règles NumPy/APL)."""
    if s1 is None or s2 is None:
        return None
    if s1 == () and s2 == ():
        return ()
    if s1 == ():
        return s2
    if s2 == ():
        return s1

    ndim1, ndim2 = len(s1), len(s2)
    if ndim1 > ndim2:
        s2_padded = (1,) * (ndim1 - ndim2) + s2
        s1_padded = s1
    elif ndim2 > ndim1:
        s1_padded = (1,) * (ndim2 - ndim1) + s1
        s2_padded = s2
    else:
        s1_padded = s1
        s2_padded = s2

    result = []
    for i, (d1, d2) in enumerate(zip(s1_padded, s2_padded)):
        if d1 is None and d2 is None:
            result.append(None)
        elif d1 is None:
            result.append(d2)
        elif d2 is None:
            result.append(d1)
        elif d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            raise APLShapeError(
                f"Dimension mismatch at axis {i}: {d1} vs {d2}",
                left_shape=s1,
                right_shape=s2,
            )
    return tuple(result)


def _infer_function_shape(
    name: str, args: List[Expr], kwargs: Dict
) -> Shape:
    """Infère la forme de sortie d'une fonction."""
    arg_shape = args[0].shape if args else None
    if arg_shape is None:
        return None

    axis = kwargs.get('dim', kwargs.get('axis', None))

    # Fonctions qui préservent la forme
    if name in ('abs', 'sqrt', 'log', 'exp', 'softmax', 'sort', 'threshold'):
        return arg_shape

    # Fonctions qui réduisent
    if name in ('mean', 'var', 'std', 'sum', 'max', 'min', 'norm'):
        if axis is not None:
            ax = int(axis)
            ndim = len(arg_shape) if arg_shape else 0
            if ax < 0:
                ax = ndim + ax
            if ax < 0 or ax >= ndim:
                raise APLShapeError(
                    f"AXIS ERROR: axis={axis} out of bounds for rank {ndim}",
                    axis=axis,
                    max_axis=ndim - 1 if ndim > 0 else -1,
                    left_shape=arg_shape,
                )
            return arg_shape[:ax] + arg_shape[ax + 1:]
        return ()

    if name == 'topk':
        k = int(args[1]) if len(args) > 1 else 10
        return (k,)

    return None


# ==============================================================================
# 8. Évaluateur JAX
# ==============================================================================
# Du Projet A : fonctions pures pour JIT, grad, vmap.

def evaluate_ir(ir: Expr, variables: Dict[str, Any]) -> Any:
    """Évalue un nœud IR avec des variables données (backend JAX/NumPy)."""
    use_jax = _HAS_JAX and any(
        hasattr(v, 'at') for v in variables.values()
    )
    np_mod = jnp if use_jax else np

    if ir is None:
        return None

    if isinstance(ir, Literal):
        return np_mod.array(ir.value, dtype=np_mod.float32)

    if isinstance(ir, Variable):
        if ir.name not in variables:
            raise NameError(f"Variable '{ir.name}' not defined")
        val = variables[ir.name]
        if isinstance(val, APLArray):
            return val.jax() if use_jax else val.numpy()
        return np_mod.asarray(val, dtype=np_mod.float32)

    if isinstance(ir, UnaryOp):
        operand = evaluate_ir(ir.operand, variables)
        if ir.op == 'neg':
            return -operand
        if ir.op == 'abs':
            return np_mod.abs(operand)
        raise ValueError(f"Unknown unary: {ir.op}")

    if isinstance(ir, BinaryOp):
        left = evaluate_ir(ir.left, variables)
        right = evaluate_ir(ir.right, variables)
        op_map = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
            'div': lambda a, b: a / b,
            'pow': lambda a, b: a ** b,
        }
        if ir.op not in op_map:
            raise ValueError(f"Unknown binary: {ir.op}")
        result = op_map[ir.op](left, right)
        if ir.op == 'div':
            result = np_mod.where(
                np_mod.abs(right) < 1e-12,
                np_mod.full_like(left, np_mod.nan),
                result,
            )
        return result

    if isinstance(ir, FunctionCall):
        return _eval_function(ir, variables, np_mod)

    if isinstance(ir, Reduction):
        operand = evaluate_ir(ir.operand, variables)
        return _eval_reduction(ir.op, operand, np_mod)

    raise ValueError(f"Unknown IR node: {type(ir).__name__}")


def _eval_function(func: FunctionCall, variables: Dict, np_mod) -> Any:
    """Évalue un appel de fonction."""
    if not func.args:
        raise ValueError(
            f"Function '{func.name}' requires at least one argument"
        )

    name = func.name
    args = [evaluate_ir(a, variables) for a in func.args]
    kwargs = {k: evaluate_ir(v, variables) for k, v in func.kwargs.items()}

    axis = kwargs.get('dim', kwargs.get('axis', None))
    if axis is not None:
        axis = int(axis)

    # Statistiques
    if name == 'mean':
        return np_mod.mean(args[0], axis=axis)
    if name == 'var':
        return np_mod.var(args[0], axis=axis)
    if name == 'std':
        return np_mod.std(args[0], axis=axis)
    if name == 'sum':
        return np_mod.sum(args[0], axis=axis)
    if name == 'max':
        return np_mod.max(args[0], axis=axis)
    if name == 'min':
        return np_mod.min(args[0], axis=axis)

    # Norm
    if name == 'norm':
        return np_mod.linalg.norm(args[0], axis=axis)

    # Math
    if name == 'abs':
        return np_mod.abs(args[0])
    if name == 'sqrt':
        return np_mod.sqrt(np_mod.maximum(args[0], 0.0))
    if name == 'log':
        return np_mod.log(np_mod.maximum(args[0], 1e-8))
    if name == 'exp':
        return np_mod.exp(args[0])

    # Softmax
    if name == 'softmax':
        ax = axis if axis is not None else -1
        if use_jax := hasattr(np_mod, 'nn'):
            return np_mod.nn.softmax(args[0], axis=ax)
        # NumPy fallback
        e_x = np.exp(args[0] - np.max(args[0], axis=ax, keepdims=True))
        return e_x / np.sum(e_x, axis=ax, keepdims=True)

    # Spéciales
    if name == 'topk':
        k = int(args[1]) if len(args) > 1 else 10
        flat = args[0].flatten()
        return np_mod.sort(flat)[-k:]

    if name == 'threshold':
        t = args[1] if len(args) > 1 else 0.5
        return (args[0] > t).astype(np_mod.float32)

    if name == 'where':
        return np_mod.where(args[0])

    if name == 'rank':
        return np_mod.linalg.matrix_rank(args[0])

    if name == 'sort':
        return np_mod.sort(args[0], axis=axis)

    raise ValueError(f"Unknown function: {name}")


def _eval_reduction(op: str, operand: Any, np_mod) -> Any:
    """Évalue une réduction."""
    if op == 'sum':
        return np_mod.sum(operand)
    if op == 'product':
        return np_mod.prod(operand)
    if op == 'max':
        return np_mod.max(operand)
    if op == 'min':
        return np_mod.min(operand)
    raise ValueError(f"Unknown reduction: {op}")


def _extract_variables(ir: Expr) -> Set[str]:
    """Extrait tous les noms de variables d'un IR."""
    found = set()
    _extract_vars_rec(ir, found)
    return found


def _extract_vars_rec(ir, found: Set[str]):
    if isinstance(ir, Variable):
        found.add(ir.name)
    elif isinstance(ir, (Index, IndexMulti)):
        found.add(ir.variable)
    elif isinstance(ir, UnaryOp):
        _extract_vars_rec(ir.operand, found)
    elif isinstance(ir, BinaryOp):
        _extract_vars_rec(ir.left, found)
        _extract_vars_rec(ir.right, found)
    elif isinstance(ir, FunctionCall):
        for a in ir.args:
            _extract_vars_rec(a, found)
        for v in ir.kwargs.values():
            _extract_vars_rec(v, found)
    elif isinstance(ir, Reduction):
        _extract_vars_rec(ir.operand, found)


# ==============================================================================
# 9. Backend Numba
# ==============================================================================
# Inspiré du Projet B : compile les nœuds fusionnés en kernels JIT CPU.

class NumbaBackend:
    """Compile les nœuds sémantiques fusionnés en fonctions Numba."""

    def compile(self, ir: SExpr) -> Callable:
        optimizer = FusionOptimizer()
        optimized = optimizer.optimize(ir)

        if isinstance(optimized, IotaReduce):
            return self._compile_iota_reduce(optimized)
        elif isinstance(optimized, MapReduce):
            return self._compile_map_reduce(optimized)
        elif isinstance(optimized, IotaTable):
            return self._compile_iota_table(optimized)
        elif isinstance(optimized, Reduce):
            return self._compile_reduce(optimized)
        else:
            return self._compile_fallback(ir)

    def _compile_iota_reduce(self, node: IotaReduce) -> Callable:
        op = node.function.op

        if op == '+':
            if _HAS_NUMBA:
                @numba.jit(nopython=True)
                def kernel(bound):
                    acc = 0
                    for i in range(1, int(bound) + 1):
                        acc += i
                    return acc
                return kernel
            else:
                def kernel(bound):
                    return sum(range(1, int(bound) + 1))
                return kernel

        fn_impl = node.function._impl

        def kernel(bound):
            acc = 0 if op == '+' else (1 if op == '×' else 0)
            for i in range(1, int(bound) + 1):
                acc = fn_impl(acc, i)
            return acc
        return kernel

    def _compile_map_reduce(self, node: MapReduce) -> Callable:
        map_fn = node.map_function._impl
        red_fn = node.reduce_function._impl

        def kernel(arr):
            flat = np.asarray(arr).flatten()
            acc = map_fn(flat[0], flat[0])
            for i in range(1, len(flat)):
                acc = red_fn(acc, map_fn(flat[i], flat[i]))
            return acc
        return kernel

    def _compile_iota_table(self, node: IotaTable) -> Callable:
        fn = node.function._impl

        def kernel(rows, cols):
            result = np.empty((int(rows), int(cols)))
            for i in range(1, int(rows) + 1):
                for j in range(1, int(cols) + 1):
                    result[i - 1, j - 1] = fn(i, j)
            return result
        return kernel

    def _compile_reduce(self, node: Reduce) -> Callable:
        fn = node.function._impl

        def kernel(arr):
            flat = np.asarray(arr).flatten()
            acc = flat[0]
            for i in range(1, len(flat)):
                acc = fn(acc, flat[i])
            return acc
        return kernel

    def _compile_fallback(self, ir: SExpr) -> Callable:
        def kernel(**variables):
            return evaluate_ir(ir, variables)
        return kernel


# ==============================================================================
# 10. Export PyTorch
# ==============================================================================
# Du Projet A : export torch.fx natif + code string.

def ir_to_pytorch_code(ir: Expr) -> str:
    """Convertit un IR en chaîne de code PyTorch."""
    if isinstance(ir, Literal):
        return str(ir.value)
    if isinstance(ir, Variable):
        return ir.name
    if isinstance(ir, UnaryOp):
        operand = ir_to_pytorch_code(ir.operand)
        if ir.op == 'neg':
            return f"(-{operand})"
        if ir.op == 'abs':
            return f"torch.abs({operand})"
    if isinstance(ir, BinaryOp):
        left = ir_to_pytorch_code(ir.left)
        right = ir_to_pytorch_code(ir.right)
        ops = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/', 'pow': '**'}
        return f"({left} {ops.get(ir.op, '?')} {right})"
    if isinstance(ir, FunctionCall):
        args = [ir_to_pytorch_code(a) for a in ir.args]
        name = ir.name
        axis = ir.kwargs.get('dim', ir.kwargs.get('axis', None))
        torch_fn = {
            'abs': 'torch.abs', 'mean': 'torch.mean',
            'sum': 'torch.sum', 'max': 'torch.max',
            'min': 'torch.min', 'norm': 'torch.linalg.norm',
            'softmax': 'torch.nn.functional.softmax',
            'sqrt': 'torch.sqrt', 'exp': 'torch.exp',
            'log': 'torch.log', 'var': 'torch.var',
            'std': 'torch.std',
        }.get(name, name)
        if axis is not None:
            return f"{torch_fn}({args[0]}, dim={axis})"
        return f"{torch_fn}({args[0]})"
    if isinstance(ir, Reduction):
        operand = ir_to_pytorch_code(ir.operand)
        return f"torch.sum({operand})"
    return f"<{type(ir).__name__}>"


def to_pytorch(code_or_ir, variables=None) -> str:
    """Export PyTorch : code string."""
    if isinstance(code_or_ir, str):
        parser = _get_parser()
        ir = parser.parse(code_or_ir)
    else:
        ir = code_or_ir
    return ir_to_pytorch_code(ir)


def to_pytorch_function(code: str, func_name: str = None) -> str:
    """Export PyTorch : fonction complète."""
    parser = _get_parser()
    ir = parser.parse(code)
    var_names = _extract_variables(ir)
    if func_name is None:
        func_name = "pruning_score"
    body = ir_to_pytorch_code(ir)
    lines = [
        "import torch",
        "import torch.nn.functional",
        "",
        f"def {func_name}({', '.join(sorted(var_names))}):",
        f"    return {body}",
    ]
    return '\n'.join(lines)


if _HAS_TORCH:
    def to_torch_fx(code_or_ir):
        """Export torch.fx GraphModule natif."""
        from torch.fx import GraphModule, Graph

        if isinstance(code_or_ir, str):
            parser = _get_parser()
            ir = parser.parse(code_or_ir)
        else:
            ir = code_or_ir

        var_names = sorted(_extract_variables(ir))
        graph = Graph()
        node_map = {}
        counter = [0]

        def unique(prefix="t"):
            counter[0] += 1
            return f"{prefix}_{counter[0]}"

        for name in var_names:
            node_map[name] = graph.placeholder(name)

        def build(expr):
            if isinstance(expr, Variable):
                return node_map[expr.name]
            if isinstance(expr, Literal):
                return graph.call_function(
                    lambda v=expr.value: torch.tensor(float(v)),
                    {}, {}, unique("lit")
                )
            if isinstance(expr, UnaryOp):
                op = build(expr.operand)
                if expr.op == 'neg':
                    return graph.call_function(operator.neg, (op,), {}, unique("neg"))
                if expr.op == 'abs':
                    return graph.call_function(torch.abs, (op,), {}, unique("abs"))
            if isinstance(expr, BinaryOp):
                l = build(expr.left)
                r = build(expr.right)
                ops = {
                    'add': operator.add, 'sub': operator.sub,
                    'mul': operator.mul, 'div': operator.truediv,
                    'pow': operator.pow,
                }
                return graph.call_function(ops[expr.op], (l, r), {}, unique(expr.op))
            if isinstance(expr, FunctionCall):
                args = [build(a) for a in expr.args]
                name = expr.name
                axis = expr.kwargs.get('dim', expr.kwargs.get('axis', None))
                if name == 'abs':
                    return graph.call_function(torch.abs, (args[0],), {}, unique("abs"))
                if name == 'mean':
                    kw = {'dim': int(axis)} if axis is not None else {}
                    return graph.call_function(torch.mean, (args[0],), kw, unique("mean"))
                if name == 'sum':
                    kw = {'dim': int(axis)} if axis is not None else {}
                    return graph.call_function(torch.sum, (args[0],), kw, unique("sum"))
                if name == 'softmax':
                    ax = int(axis) if axis is not None else -1
                    return graph.call_function(
                        F.softmax, (args[0],), {'dim': ax}, unique("softmax")
                    )
            return graph.call_function(lambda: None, {}, {}, unique("unknown"))

        result_node = build(ir)
        graph.output(result_node)
        return GraphModule({}, graph)
else:
    def to_torch_fx(code_or_ir):
        raise ImportError("PyTorch is not installed")


# ==============================================================================
# 11. API Principale : MiniAPLParser Unifié
# ==============================================================================
# Fusion des APIs des Projets A et B.

class MiniAPLParser:
    """
    Compilateur APL unifié pour le pruning et le calcul array-oriented.

    Usage :
        parser = MiniAPLParser()
        parser.set_variables(W=weights, act=activations)
        scores = parser.evaluate("|W| x mean(|act|)")

        # JIT
        jit_fn = parser.jit("|W| x mean(|act|)")
        scores = jit_fn(W=weights, act=activations)

        # Numba (pour +/⍳N)
        fn = parser.numba_compile("+/⍳N")
        result = fn(1000)
    """

    def __init__(self, mode: str = 'auto', check_shapes: bool = True):
        self.variables: Dict[str, Any] = {}
        self._visitor = APLVisitor()
        self._optimizer = FusionOptimizer()
        self._numba_backend = NumbaBackend()
        self._mode = mode
        self._check_shapes = check_shapes

    def set_variable(self, name: str, value):
        self.variables[name] = value

    def set_variables(self, **kwargs):
        self.variables.update(kwargs)

    def parse(self, code: str) -> Expr:
        """Parse le code APL en IR syntaxique."""
        if not _HAS_PARSIMONIOUS or APL_GRAMMAR is None:
            raise ImportError(
                "parsimonious is required. Install with: pip install parsimonious"
            )
        code = self._strip_comments(code)
        if not code.strip():
            raise ValueError("Empty expression")
        tree = APL_GRAMMAR.parse(code.strip())
        return self._visitor.visit(tree)

    def evaluate(self, code: str) -> Any:
        """Parse, valide, et évalue. Retourne un tableau NumPy."""
        ir = self.parse(code)

        if self._check_shapes:
            arrays = {
                k: v for k, v in self.variables.items()
                if isinstance(v, (np.ndarray, APLArray))
            }
            if arrays:
                shapes = {
                    k: v.shape if isinstance(v, np.ndarray) else v._shape
                    for k, v in arrays.items()
                }
                infer_shapes(ir, shapes)

        result = evaluate_ir(ir, self.variables)
        if hasattr(result, 'block_until_ready'):
            return np.asarray(result)
        return np.asarray(result)

    def evaluate_jax(self, code: str):
        """Évalue et retourne un tableau JAX (pour grad)."""
        if not _HAS_JAX:
            raise ImportError("JAX is not installed")
        ir = self.parse(code)
        jax_vars = {}
        for k, v in self.variables.items():
            if isinstance(v, APLArray):
                jax_vars[k] = v.jax()
            elif isinstance(v, np.ndarray):
                jax_vars[k] = jnp.asarray(v, dtype=jnp.float32)
            else:
                jax_vars[k] = v
        return evaluate_ir(ir, jax_vars)

    def compile(self, code: str) -> Callable:
        """Compile en fonction pure (backend auto)."""
        if self._mode == 'numba' or ('⍳' in code and '/' in code):
            if _HAS_NUMBA:
                return self._numba_compile(code)
        return self._jax_compile(code)

    def jit(self, code: str) -> Callable:
        """Compile JIT pour performance maximale."""
        if not _HAS_JAX:
            return self.compile(code)

        ir = self.parse(code)
        var_names = sorted(_extract_variables(ir))

        @jax.jit
        def jit_fn(**kwargs):
            jax_vars = {
                k: jnp.asarray(v, dtype=jnp.float32)
                if isinstance(v, np.ndarray) else v
                for k, v in kwargs.items()
            }
            return evaluate_ir(ir, jax_vars)

        jit_fn._apl_code = code
        jit_fn._apl_variables = var_names
        return jit_fn

    def _jax_compile(self, code: str) -> Callable:
        ir = self.parse(code)
        var_names = sorted(_extract_variables(ir))

        def fn(**kwargs):
            return np.asarray(evaluate_ir(ir, kwargs))

        fn._apl_code = code
        fn._apl_variables = var_names
        return fn

    def numba_compile(self, code: str) -> Callable:
        """Compile avec le backend Numba (pour les motifs fusionnables)."""
        if not _HAS_NUMBA:
            warnings.warn("Numba not installed, using fallback")
        ir = self.parse(code)
        optimizer = FusionOptimizer()
        optimized = optimizer.optimize(ir)
        return self._numba_backend.compile(optimized)

    def to_pytorch(self, code: str) -> str:
        """Export en code PyTorch."""
        return to_pytorch(code)

    def to_pytorch_function(self, code: str, func_name: str = None) -> str:
        """Export en fonction PyTorch complète."""
        return to_pytorch_function(code, func_name)

    def to_torch_fx(self, code: str):
        """Export en torch.fx GraphModule."""
        return to_torch_fx(code)

    @staticmethod
    def _strip_comments(code: str) -> str:
        lines = []
        for line in code.split('\n'):
            if '\u235d' in line:
                line = line[:line.index('\u235d')]
            if '//' in line:
                line = line[:line.index('//')]
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
        return '\n'.join(lines)

    def __call__(self, code: str):
        return self.evaluate(code)


# ==============================================================================
# 12. Méthodes de Scoring (Registre)
# ==============================================================================
# Conservé du Projet A pour le cas d'usage pruning.

METHODS = {
    "magnitude": {
        "formula": "|W|",
        "needs_grad": False,
        "description": "Weight magnitude |W|",
        "variables": ["W"],
    },
    "gradient": {
        "formula": "|W| x |grad|",
        "needs_grad": True,
        "description": "Gradient-weighted |W| * |grad|",
        "variables": ["W", "grad"],
    },
    "wanda": {
        "formula": "|W| x mean(|act|)",
        "needs_grad": False,
        "description": "Wanda: |W| * mean(|act|)",
        "variables": ["W", "act"],
    },
    "wanda_per_neuron": {
        "formula": "|W| x mean(|act|, dim=-1)",
        "needs_grad": False,
        "description": "Wanda per neuron",
        "variables": ["W", "act"],
    },
    "direction": {
        "formula": "(max(|W|)) / mean(|W|)",
        "needs_grad": False,
        "description": "Direction: max(|W|) / mean(|W|)",
        "variables": ["W"],
    },
    "direction_per_neuron": {
        "formula": "max(|W|, dim=-1) / mean(|W|, dim=-1)",
        "needs_grad": False,
        "description": "Direction per neuron",
        "variables": ["W"],
    },
    "softmax_grad": {
        "formula": "softmax(|W|, dim=-1) x |grad|",
        "needs_grad": True,
        "description": "Softmax(|W|) * |grad|",
        "variables": ["W", "grad"],
    },
    "norm_ratio": {
        "formula": "norm(W, dim=-1) / (norm(grad, dim=-1) + 1e-8)",
        "needs_grad": True,
        "description": "Per-neuron norm ratio",
        "variables": ["W", "grad"],
    },
    "threshold": {
        "formula": "threshold(|W|, 0.5)",
        "needs_grad": False,
        "description": "Binary mask: |W| > 0.5",
        "variables": ["W"],
    },
}


def score_layer(method: str, **variables) -> np.ndarray:
    """Score une couche avec une méthode nommée."""
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(METHODS.keys())}"
        )
    formula = METHODS[method]["formula"]
    parser = MiniAPLParser(check_shapes=False)
    parser.set_variables(**variables)
    return parser.evaluate(formula)


def list_methods() -> List[Dict]:
    """Liste les méthodes disponibles."""
    return [
        {"name": k, "needs_grad": v["needs_grad"],
         "description": v["description"]}
        for k, v in METHODS.items()
    ]


# ==============================================================================
# 13. Cache de Sous-Expressions (backward compat)
# ==============================================================================

class LayerCache:
    """Cache par couche pour les sous-expressions communes."""

    COMMON_OPS = [
        ("abs_{name}", lambda x: np.abs(x)),
        ("mean_abs_{name}", lambda x: np.mean(np.abs(x))),
        ("norm_{name}", lambda x: np.linalg.norm(x)),
        ("max_abs_{name}", lambda x: np.max(np.abs(x))),
        ("sum_abs_{name}", lambda x: np.sum(np.abs(x))),
    ]

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0

    def precompute(self, layer_name: str, **variables):
        if layer_name not in self._cache:
            self._cache[layer_name] = {}
        for template, fn in self.COMMON_OPS:
            for var_name, value in variables.items():
                key = template.format(name=var_name)
                if key not in self._cache[layer_name]:
                    self._cache[layer_name][key] = fn(value)
        for var_name, value in variables.items():
            self._cache[layer_name][var_name] = value

    def store(self, layer_name: str, key: str, value: Any):
        if layer_name not in self._cache:
            self._cache[layer_name] = {}
        self._cache[layer_name][key] = value

    def get(self, layer_name: str, key: str):
        if layer_name in self._cache and key in self._cache[layer_name]:
            self._hits += 1
            return self._cache[layer_name][key]
        self._misses += 1
        return None

    def has(self, layer_name: str, key: str) -> bool:
        return layer_name in self._cache and key in self._cache[layer_name]

    def clear(self):
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self):
        total = self._hits + self._misses
        return {
            "hits": self._hits, "misses": self._misses,
            "total": total,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


def fused_abs_mean(x, axis=None):
    return np.mean(np.abs(x), axis=axis)


def fused_abs_max(x, axis=None):
    return np.max(np.abs(x), axis=axis)


def fused_abs_sum(x, axis=None):
    return np.sum(np.abs(x), axis=axis)


def fused_norm_ratio(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_b < 1e-12:
        raise ZeroDivisionError("Division by near-zero norm")
    return norm_a / norm_b


# ==============================================================================
# 14. LayerScorer (backward compat)
# ==============================================================================

class LayerScorer:
    """Score les couches d'un modèle avec le DSL."""

    def __init__(self, layers: dict, use_jax: bool = False):
        self.layers = layers
        self.parser = MiniAPLParser()
        self.cache = LayerCache()
        self.use_jax = use_jax

    def score_all(self, formula: str, use_cache: bool = True):
        results = {}
        for name, variables in self.layers.items():
            if use_cache:
                self.cache.precompute(name, **variables)
                cached_vars = {
                    k: self.cache.get(name, k) or v
                    for k, v in variables.items()
                }
                self.parser.set_variables(**cached_vars)
            else:
                self.parser.set_variables(**variables)
            if self.use_jax:
                results[name] = np.asarray(self.parser.evaluate_jax(formula))
            else:
                results[name] = self.parser.evaluate(formula)
        return results


# ==============================================================================
# 15. Helpers internes
# ==============================================================================

_global_parser: Optional[MiniAPLParser] = None


def _get_parser() -> MiniAPLParser:
    global _global_parser
    if _global_parser is None:
        _global_parser = MiniAPLParser(check_shapes=False)
    return _global_parser


# ==============================================================================
# 16. Démonstration
# ==============================================================================

def demo():
    """Démonstration des fonctionnalités principales."""
    print("=" * 60)
    print("APL-Python DSL Compiler v2.0 - Démonstration")
    print("=" * 60)

    # 1. Parsing basique
    print("\n--- 1. Parsing ---")
    parser = MiniAPLParser(check_shapes=False)

    W = np.random.randn(512, 256).astype(np.float32)
    act = np.random.randn(128, 256).astype(np.float32)
    parser.set_variables(W=W, act=act)

    result = parser.evaluate("|W| x mean(|act|)")
    print(f"  |W| x mean(|act|) -> shape {result.shape}")

    # 2. Formes avec vérification
    print("\n--- 2. Shape Checking ---")
    parser_check = MiniAPLParser(check_shapes=True)
    parser_check.set_variables(W=W, act=act)

    try:
        result = parser_check.evaluate("|W| x mean(|act|)")
        print(f"  ✓ Valid: result shape = {result.shape}")
    except APLShapeError as e:
        print(f"  ✗ {e}")

    # 3. Test d'erreur de forme
    print("\n--- 3. Shape Error ---")
    W_bad = np.random.randn(512, 256).astype(np.float32)
    act_bad = np.random.randn(128, 128).astype(np.float32)
    parser_check.set_variables(W=W_bad, act=act_bad)

    try:
        parser_check.evaluate("|W| x mean(|act|)")
        print("  ✗ Should have raised ShapeError!")
    except APLShapeError as e:
        print(f"  ✓ Correctly caught:\n{e}")

    # 4. JIT (si JAX disponible)
    if _HAS_JAX:
        print("\n--- 4. JIT Compilation ---")
        parser_jit = MiniAPLParser(check_shapes=False)
        parser_jit.set_variables(W=W, act=act)

        jit_fn = parser_jit.jit("|W| x mean(|act|)")
        t0 = time.time()
        for _ in range(100):
            _ = jit_fn(W=W, act=act)
        jit_time = time.time() - t0

        t0 = time.time()
        for _ in range(100):
            _ = parser.evaluate("|W| x mean(|act|)")
        no_jit_time = time.time() - t0

        print(f"  Without JIT: {no_jit_time:.4f}s")
        print(f"  With JIT:    {jit_time:.4f}s")
        if no_jit_time > 0:
            print(f"  Speedup:     {no_jit_time / jit_time:.1f}x")

    # 5. PyTorch Export
    print("\n--- 5. PyTorch Export ---")
    for formula in [
        "|W| x mean(|act|)",
        "softmax(|W|, dim=-1) x |grad|",
        "max(|W|, dim=-1) / mean(|W|, dim=-1)",
    ]:
        print(f"  APL: {formula}")
        print(f"  PyTorch: {parser.to_pytorch(formula)}")

    # 6. Méthodes de scoring
    print("\n--- 6. Scoring Methods ---")
    for m in list_methods():
        print(f"  {m['name']:<20} grad={m['needs_grad']:<5} {m['description']}")

    # 7. APLArray
    print("\n--- 7. APLArray ---")
    arr = APLArray(np.array([1, 2, 3]))
    print(f"  {arr}")
    print(f"  numpy()  -> {arr.numpy()}")
    print(f"  tolist() -> {arr.tolist()}")

    # 8. Numba (si dispo)
    if _HAS_NUMBA:
        print("\n--- 8. Numba Backend ---")
        nb_parser = MiniAPLParser(mode='numba')
        fn = nb_parser.numba_compile("+/⍳N")
        result = fn(100)
        print(f"  +/⍳100 = {result} (expected 5050)")

    print("\n" + "=" * 60)
    print("Démonstration terminée.")
    print("=" * 60)


# ==============================================================================
# 17. Point d'entrée
# ==============================================================================

if __name__ == "__main__":
    demo()



"""
apl_pruning/
├── __init__.py          # Imports publics, __all__, __version__
├── apl_array.py         # APLArray (section 1)
├── peg_grammar.py       # Grammaire PEG (section 2)
├── ir_syntaxique.py     # IR niveau 1 (section 3)
├── ir_semantique.py     # IR niveau 2 + ScalarFunction (section 4)
├── visitor.py           # APLVisitor (section 5)
├── fusion_optimizer.py  # FusionOptimizer (section 6)
├── shape_checker.py     # Inférence + APLShapeError (section 7)
├── jax_evaluator.py     # evaluate_ir + _extract_variables (section 8)
├── numba_backend.py     # NumbaBackend (section 9)
├── exporter.py          # PyTorch export (section 10)
├── parser.py            # MiniAPLParser (section 11)
├── scorers.py           # METHODS + score_layer (section 12)
├── cache.py             # LayerCache + fused_* (section 13)
├── layers.py            # LayerScorer (section 14)
└── demo.py              # Démo (section 16)


Le point d'entrée __init__.py exposerait l'API publique :

python
from .apl_array import APLArray
from .parser import MiniAPLParser
from .exporter import to_pytorch, to_pytorch_function, to_torch_fx
from .scorers import METHODS, score_layer, list_methods
from .cache import LayerCache, fused_abs_mean, fused_abs_max, fused_abs_sum, fused_norm_ratio
from .layers import LayerScorer
from .shape_checker import APLShapeError

__version__ = "2.0.0"
__all__ = [
    "APLArray",
    "MiniAPLParser",
    "to_pytorch", "to_pytorch_function", "to_torch_fx",
    "METHODS", "score_layer", "list_methods",
    "LayerCache", "fused_abs_mean", "fused_abs_max", "fused_abs_sum", "fused_norm_ratio",
    "LayerScorer",
    "APLShapeError",
]

"""