"""
    Micro-Pascal-Compiler/ast_nodes.py
============================
Abstract Syntax Tree for subset-Pascal.

Inheritance note
----------------
Python frozen dataclasses require that fields with defaults come *after*
fields without defaults, even across base-class inheritance.  The ASTNode
base is therefore a plain (non-dataclass) marker class that declares no
fields.  Each concrete node is its own frozen dataclass and places the
optional `line` / `column` source-location fields at the very end with
default 0, so call-sites that don't care about location can omit them.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum        import Enum, auto
from typing      import Optional


# ═════════════════════════════════════════════════════════════════════
#  Pascal type system
# ═══════════════════════════════════════════════════════════════

class PascalType(Enum):
    INTEGER = auto()
    BOOLEAN = auto()
    CHAR    = auto()
    STRING  = auto()   # write-only string literals (no variable type)
    UNKNOWN = auto()   # not yet resolved / error sentinel

    def __repr__(self) -> str:
        return self.name.capitalize()


# ═══════════════════════════════════════════════════════════════
#  Operator enums
# ══════════════════════════════════════════════════════════════════════

class BinOp(Enum):
    # Arithmetic
    ADD = '+';  SUB = '-';  MUL = '*';  DIV = 'div';  MOD = 'mod'
    # Relational
    EQ  = '=';  NEQ = '<>'; LT = '<';  GT  = '>';
    LTE = '<='; GTE = '>='
    # Logical
    AND = 'and'; OR = 'or'

    def __repr__(self) -> str:
        return self.value

class UnOp(Enum):
    NEG = '-'    # unary minus
    NOT = 'not'  # logical NOT

    def __repr__(self) -> str:
        return self.value


# ═════════════════════════════════════════════════════════════════
#  Marker base  (no dataclass, no fields)
# ═══════════════════════════════════════════════════════════════

class ASTNode:
    """Marker base-class for all AST nodes.  Declares no fields."""


# ══════════════════════════════════════════════════════════════════
#  Declarations
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VarDecl(ASTNode):
    """Single variable declaration entry in the symbol table."""
    name     : str
    var_type : PascalType
    offset   : int = 0    # byte offset from rbp; assigned by code-gen
    line     : int = 0
    column   : int = 0


# ═══════════════════════════════════════════════════════════════
#  Literal expression nodes
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class IntLiteral(ASTNode):
    value    : int
    var_type : PascalType = PascalType.INTEGER
    line     : int = 0
    column   : int = 0

@dataclass(frozen=True)
class BoolLiteral(ASTNode):
    value    : bool
    var_type : PascalType = PascalType.BOOLEAN
    line     : int = 0
    column   : int = 0

@dataclass(frozen=True)
class CharLiteral(ASTNode):
    value    : str          # guaranteed length-1
    var_type : PascalType = PascalType.CHAR
    line     : int = 0
    column   : int = 0

@dataclass(frozen=True)
class StringLiteral(ASTNode):
    value    : str
    var_type : PascalType = PascalType.STRING
    line     : int = 0
    column   : int = 0


# ══════════════════════════════════════════════════════════════════
#  Variable reference
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VarRef(ASTNode):
    """A use-site reference to a declared variable."""
    name     : str
    decl     : Optional[VarDecl] = None   # resolved during semantic pass
    line     : int = 0
    column   : int = 0

    @property
    def var_type(self) -> PascalType:
        return self.decl.var_type if self.decl else PascalType.UNKNOWN


# ═══════════════════════════════════════════════════════════════
#  Compound expressions
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BinaryOp(ASTNode):
    """left op right  — arithmetic, relational, or logical."""
    left     : ASTNode
    op       : BinOp
    right    : ASTNode
    var_type : PascalType = PascalType.UNKNOWN
    line     : int = 0
    column   : int = 0

@dataclass(frozen=True)
class UnaryOp(ASTNode):
    """op operand  — unary minus or logical NOT."""
    op       : UnOp
    operand  : ASTNode
    var_type : PascalType = PascalType.UNKNOWN
    line     : int = 0
    column   : int = 0


# ── Convenience type alias ───────────────────────────────────
Expr = (IntLiteral | BoolLiteral | CharLiteral | StringLiteral
        | VarRef | BinaryOp | UnaryOp)


# ═══════════════════════════════════════════════════════════════
#  Statement nodes
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Block(ASTNode):
    """BEGIN statement* END"""
    statements : tuple        # tuple[ASTNode, ...]
    line       : int = 0
    column     : int = 0

@dataclass(frozen=True)
class Assignment(ASTNode):
    """target := value"""
    target : VarRef
    value  : ASTNode
    line   : int = 0
    column : int = 0

@dataclass(frozen=True)
class IfStatement(ASTNode):
    """IF condition THEN then_branch [ELSE else_branch]"""
    condition   : ASTNode
    then_branch : ASTNode
    else_branch : Optional[ASTNode] = None
    line        : int = 0
    column      : int = 0

@dataclass(frozen=True)
class WhileStatement(ASTNode):
    """WHILE condition DO body"""
    condition : ASTNode
    body      : ASTNode
    line      : int = 0
    column    : int = 0

@dataclass(frozen=True)
class ForStatement(ASTNode):
    """FOR var := start TO|DOWNTO stop DO body"""
    var    : VarRef
    start  : ASTNode
    stop   : ASTNode
    downto : bool             # True → DOWNTO, False → TO
    body   : ASTNode
    line   : int = 0
    column : int = 0

@dataclass(frozen=True)
class WritelnStatement(ASTNode):
    """WRITELN(arg, …)  — appends newline"""
    args   : tuple            # tuple[ASTNode, ...]
    line   : int = 0
    column : int = 0

@dataclass(frozen=True)
class WriteStatement(ASTNode):
    """WRITE(arg, …)  — no newline"""
    args   : tuple
    line   : int = 0
    column : int = 0

@dataclass(frozen=True)
class ReadlnStatement(ASTNode):
    """READLN(var, …)"""
    targets : tuple           # tuple[VarRef, ...]
    line    : int = 0
    column  : int = 0

@dataclass(frozen=True)
class ReadStatement(ASTNode):
    """READ(var, …)"""
    targets : tuple
    line    : int = 0
    column  : int = 0


# ═════════════════════════════════════════════════════════════════
#  Root
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Program(ASTNode):
    """Root of the AST — the whole compilation unit."""
    name      : str
    var_decls : tuple         # tuple[VarDecl, ...]
    body      : Block
    line      : int = 0
    column    : int = 0


# ═══════════════════════════════════════════════════════════════
#  Pretty-printer (debug / test utility)
# ═════════════════════════════════════════════════════════════════

def pretty(node: ASTNode, depth: int = 0) -> str:   # noqa: C901
    """Return a human-readable, indented tree dump of the AST."""
    pad = '  ' * depth

    if isinstance(node, Program):
        decls = '\n'.join(
            f"{'  '*(depth+2)}{d.name}: {d.var_type.name}"
            for d in node.var_decls)
        return (f"{pad}Program '{node.name}'\n"
                f"{'  '*(depth+1)}Vars:\n{decls}\n"
                f"{pretty(node.body, depth+1)}")

    if isinstance(node, Block):
        inner = '\n'.join(pretty(s, depth+1) for s in node.statements)
        return f"{pad}Block\n{inner}"

    if isinstance(node, Assignment):
        return f"{pad}Assign {node.target.name} :=\n{pretty(node.value, depth+1)}"

    if isinstance(node, IfStatement):
        s = (f"{pad}If\n{pad}  cond:\n{pretty(node.condition, depth+2)}\n"
             f"{pad}  then:\n{pretty(node.then_branch, depth+2)}")
        if node.else_branch:
            s += f"\n{pad}  else:\n{pretty(node.else_branch, depth+2)}"
        return s

    if isinstance(node, WhileStatement):
        return (f"{pad}While\n{pad}  cond:\n{pretty(node.condition, depth+2)}\n"
                f"{pad}  do:\n{pretty(node.body, depth+2)}")

    if isinstance(node, ForStatement):
        dir_ = "downto" if node.downto else "to"
        return (f"{pad}For {node.var.name} :=\n{pretty(node.start, depth+2)}\n"
                f"{pad}  {dir_}:\n{pretty(node.stop, depth+2)}\n"
                f"{pad}  do:\n{pretty(node.body, depth+2)}")

    if isinstance(node, WritelnStatement):
        inner = '\n'.join(pretty(a, depth+1) for a in node.args)
        return f"{pad}Writeln\n{inner}" if inner else f"{pad}Writeln"

    if isinstance(node, WriteStatement):
        return f"{pad}Write\n" + '\n'.join(pretty(a, depth+1) for a in node.args)

    if isinstance(node, ReadlnStatement):
        return f"{pad}Readln({', '.join(t.name for t in node.targets)})"

    if isinstance(node, ReadStatement):
        return f"{pad}Read({', '.join(t.name for t in node.targets)})"

    if isinstance(node, BinaryOp):
        return (f"{pad}BinOp({node.op.value})\n"
                f"{pretty(node.left, depth+1)}\n"
                f"{pretty(node.right, depth+1)}")

    if isinstance(node, UnaryOp):
        return f"{pad}UnOp({node.op.value})\n{pretty(node.operand, depth+1)}"

    if isinstance(node, IntLiteral):   return f"{pad}Int({node.value})"
    if isinstance(node, BoolLiteral):  return f"{pad}Bool({node.value})"
    if isinstance(node, CharLiteral):  return f"{pad}Char('{node.value}')"
    if isinstance(node, StringLiteral):return f"{pad}Str({node.value!r})"

    if isinstance(node, VarRef):
        t = node.var_type.name
        return f"{pad}Var({node.name}:{t})"

    return f"{pad}<unknown {type(node).__name__}>"
