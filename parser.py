"""
Micro-Pascal-Compiler/parser.py
=========================
Recursive-descent parser + symbol table for subset-Pascal.

Grammar (informal BNF)
----------------------
program       = 'program' IDENT ';' var_block? block '.'
var_block     = 'var' (ident_list ':' type_spec ';')+
block         = 'begin' statement_list 'end'
statement_list= statement (';' statement)*
statement     = assignment | if_stmt | while_stmt | for_stmt
              | writeln_stmt | write_stmt | readln_stmt | read_stmt
              | block | ε
assignment    = IDENT ':=' expression
if_stmt       = 'if' expression 'then' statement ['else' statement]
while_stmt    = 'while' expression 'do' statement
for_stmt      = 'for' IDENT ':=' expression ('to'|'downto') expression 'do' statement
writeln_stmt  = 'writeln' ['(' expr_list ')']
write_stmt    = 'write' '(' expr_list ')'
readln_stmt   = 'readln' ['(' ident_list ')']
read_stmt     = 'read'   '(' ident_list ')'
expression    = simple_expr [rel_op simple_expr]
simple_expr   = ['+' | '-'] term {('+' | '-' | 'or') term}
term          = factor {('*' | 'div' | 'mod' | 'and') factor}
factor        = INTEGER_LITERAL | BOOLEAN_LITERAL | CHAR_LITERAL | STRING_LITERAL
              | IDENT | '(' expression ')' | 'not' factor | '-' factor
rel_op        = '=' | '<>' | '<' | '>' | '<=' | '>='
type_spec     = 'integer' | 'boolean' | 'char'

Operator precedence (highest → lowest)
---------------------------------------
1.  not, unary -
2.  *  div  mod  and
3.  +  -  or
4.  =  <>  <  >  <=  >=
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional

from lexer import Lexer, Token, TT
from ast_nodes import (
    ASTNode, PascalType, BinOp, UnOp,
    VarDecl, Program, Block,
    Assignment, IfStatement, WhileStatement, ForStatement,
    WritelnStatement, WriteStatement, ReadlnStatement, ReadStatement,
    BinaryOp, UnaryOp, VarRef,
    IntLiteral, BoolLiteral, CharLiteral, StringLiteral,
)


# ═══════════════════════════════════════════════════════════════
#  Errors
# ═══════════════════════════════════════════════════════════════

class ParseError(Exception):
    def __init__(self, msg: str, tok: Token):
        super().__init__(f"[Parser] {msg}  (line {tok.line}, col {tok.column})")
        self.token = tok

class SemanticError(Exception):
    def __init__(self, msg: str, line: int = 0, col: int = 0):
        super().__init__(f"[Semantic] {msg}  (line {line}, col {col})")


# ═══════════════════════════════════════════════════════════════
#  Symbol table
# ═══════════════════════════════════════════════════════════════

class SymbolTable:
    """
    Flat symbol table for single-scope Pascal programs.
    (Nested scopes / procedures are not in scope for this compiler.)
    """

    def __init__(self):
        self._table: dict[str, VarDecl] = {}

    def define(self, decl: VarDecl) -> None:
        key = decl.name.lower()
        if key in self._table:
            raise SemanticError(
                f"Variable '{decl.name}' already declared",
                decl.line, decl.column
            )
        self._table[key] = decl

    def lookup(self, name: str, line: int = 0, col: int = 0) -> VarDecl:
        key = name.lower()
        if key not in self._table:
            raise SemanticError(f"Undeclared variable '{name}'", line, col)
        return self._table[key]

    def all_decls(self) -> list[VarDecl]:
        return list(self._table.values())

    def assign_offsets(self) -> int:
        """
        Assign rbp-relative byte offsets to every variable.
        All variables are stored as 8-byte QWORD on the stack
        (simplest uniform layout).
        Returns total bytes needed.
        """
        offset = 0
        new_table: dict[str, VarDecl] = {}
        for key, decl in self._table.items():
            offset += 8
            new_table[key] = replace(decl, offset=offset)
        self._table = new_table
        return offset


# ═══════════════════════════════════════════════════════════════
#  Type-checker helpers
# ═══════════════════════════════════════════════════════════════

_NUMERIC = {PascalType.INTEGER}
_BOOL    = {PascalType.BOOLEAN}
_CHAR    = {PascalType.CHAR}

def _arith_type(op: BinOp, lt: PascalType, rt: PascalType,
                line: int, col: int) -> PascalType:
    if lt == PascalType.INTEGER and rt == PascalType.INTEGER:
        return PascalType.INTEGER
    raise SemanticError(
        f"Operator '{op.value}' requires Integer operands, got {lt.name} and {rt.name}",
        line, col
    )

def _logic_type(op: BinOp, lt: PascalType, rt: PascalType,
                line: int, col: int) -> PascalType:
    if lt == PascalType.BOOLEAN and rt == PascalType.BOOLEAN:
        return PascalType.BOOLEAN
    raise SemanticError(
        f"Operator '{op.value}' requires Boolean operands, got {lt.name} and {rt.name}",
        line, col
    )

def _relational_type(op: BinOp, lt: PascalType, rt: PascalType,
                     line: int, col: int) -> PascalType:
    if lt != rt:
        raise SemanticError(
            f"Operator '{op.value}' requires matching types, got {lt.name} and {rt.name}",
            line, col
        )
    return PascalType.BOOLEAN


# ═══════════════════════════════════════════════════════════════
#  Parser
# ═══════════════════════════════════════════════════════════════

class Parser:
    """
    Recursive-descent parser.  One call to parse() returns
    a fully annotated Program AST node (all VarRefs resolved).

    Usage
    -----
    ast = Parser(source_code).parse()
    """

    def __init__(self, source: str):
        self._tokens: list[Token] = Lexer(source).tokenize()
        self._pos   : int         = 0
        self._syms  : SymbolTable = SymbolTable()

    # ── token navigation ──────────────────────────────────────

    @property
    def _cur(self) -> Token:
        return self._tokens[self._pos]

    def _peek(self, offset: int = 1) -> Token:
        i = min(self._pos + offset, len(self._tokens) - 1)
        return self._tokens[i]

    def _advance(self) -> Token:
        tok = self._cur
        if tok.type != TT.EOF:
            self._pos += 1
        return tok

    def _expect(self, tt: TT) -> Token:
        if self._cur.type != tt:
            raise ParseError(
                f"Expected {tt.name}, got {self._cur.type.name} ({self._cur.value!r})",
                self._cur
            )
        return self._advance()

    def _match(self, *types: TT) -> bool:
        return self._cur.type in types

    def _consume(self, *types: TT) -> Optional[Token]:
        if self._cur.type in types:
            return self._advance()
        return None

    # ── type keyword → PascalType ─────────────────────────────

    def _parse_type(self) -> PascalType:
        tok = self._cur
        mapping = {
            TT.INTEGER_TYPE: PascalType.INTEGER,
            TT.BOOLEAN_TYPE: PascalType.BOOLEAN,
            TT.CHAR_TYPE:    PascalType.CHAR,
        }
        if tok.type not in mapping:
            raise ParseError(
                f"Expected type name (Integer/Boolean/Char), got {tok.value!r}", tok
            )
        self._advance()
        return mapping[tok.type]

    # ── VAR block ─────────────────────────────────────────────

    def _parse_var_block(self) -> None:
        """Parse zero or more  ident, ident : Type;  declarations."""
        self._expect(TT.VAR)
        while self._match(TT.IDENTIFIER):
            # Collect comma-separated names
            names: list[Token] = [self._advance()]
            while self._consume(TT.COMMA):
                names.append(self._expect(TT.IDENTIFIER))
            self._expect(TT.COLON)
            ptype = self._parse_type()
            self._expect(TT.SEMICOLON)
            for name_tok in names:
                decl = VarDecl(name_tok.value, ptype,
                               line=name_tok.line, column=name_tok.column)
                self._syms.define(decl)

    # ══════════════════════════════════════════════════════════
    #  Statements
    # ══════════════════════════════════════════════════════════

    def _parse_block(self) -> Block:
        tok = self._expect(TT.BEGIN)
        stmts: list[ASTNode] = []
        while not self._match(TT.END, TT.EOF):
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
            # Semicolon is a separator between statements (not a terminator)
            if not self._match(TT.END, TT.EOF):
                self._expect(TT.SEMICOLON)
            else:
                self._consume(TT.SEMICOLON)   # optional trailing semi
        self._expect(TT.END)
        return Block(tuple(stmts), line=tok.line, column=tok.column)

    def _parse_statement(self) -> Optional[ASTNode]:
        tok = self._cur

        if tok.type == TT.BEGIN:
            return self._parse_block()

        if tok.type == TT.IDENTIFIER:
            # Peek-ahead: if next is ':=' → assignment
            if self._peek().type == TT.ASSIGN:
                return self._parse_assignment()

        if tok.type == TT.IF:
            return self._parse_if()

        if tok.type == TT.WHILE:
            return self._parse_while()

        if tok.type == TT.FOR:
            return self._parse_for()

        if tok.type == TT.WRITELN:
            return self._parse_writeln()

        if tok.type == TT.WRITE:
            return self._parse_write()

        if tok.type == TT.READLN:
            return self._parse_readln()

        if tok.type == TT.READ:
            return self._parse_read()

        # Empty statement (bare semicolon between BEGIN/END)
        return None

    def _parse_assignment(self) -> Assignment:
        name_tok = self._expect(TT.IDENTIFIER)
        decl     = self._syms.lookup(name_tok.value, name_tok.line, name_tok.column)
        ref      = VarRef(name_tok.value, decl,
                          line=name_tok.line, column=name_tok.column)
        self._expect(TT.ASSIGN)
        value = self._parse_expression()

        # Type-check assignment
        vtype = value.var_type if hasattr(value, 'var_type') else PascalType.UNKNOWN
        if vtype not in (decl.var_type, PascalType.UNKNOWN):
            raise SemanticError(
                f"Cannot assign {vtype.name} to variable '{decl.name}' of type {decl.var_type.name}",
                name_tok.line, name_tok.column
            )
        return Assignment(ref, value, line=name_tok.line, column=name_tok.column)

    def _parse_if(self) -> IfStatement:
        tok = self._expect(TT.IF)
        cond = self._parse_expression()
        self._expect(TT.THEN)
        then_br = self._parse_statement()
        else_br = None
        if self._consume(TT.ELSE):
            else_br = self._parse_statement()
        return IfStatement(cond, then_br, else_br,
                           line=tok.line, column=tok.column)

    def _parse_while(self) -> WhileStatement:
        tok = self._expect(TT.WHILE)
        cond = self._parse_expression()
        self._expect(TT.DO)
        body = self._parse_statement()
        return WhileStatement(cond, body, line=tok.line, column=tok.column)

    def _parse_for(self) -> ForStatement:
        tok      = self._expect(TT.FOR)
        name_tok = self._expect(TT.IDENTIFIER)
        decl     = self._syms.lookup(name_tok.value, name_tok.line, name_tok.column)
        if decl.var_type != PascalType.INTEGER:
            raise SemanticError(
                f"FOR loop variable '{decl.name}' must be Integer", name_tok.line
            )
        ref = VarRef(name_tok.value, decl,
                     line=name_tok.line, column=name_tok.column)
        self._expect(TT.ASSIGN)
        start  = self._parse_expression()
        downto = False
        if self._consume(TT.DOWNTO):
            downto = True
        else:
            self._expect(TT.TO)
        stop = self._parse_expression()
        self._expect(TT.DO)
        body = self._parse_statement()
        return ForStatement(ref, start, stop, downto, body,
                            line=tok.line, column=tok.column)

    def _parse_writeln(self) -> WritelnStatement:
        tok  = self._expect(TT.WRITELN)
        args = self._parse_optional_arg_list()
        return WritelnStatement(tuple(args), line=tok.line, column=tok.column)

    def _parse_write(self) -> WriteStatement:
        tok  = self._expect(TT.WRITE)
        self._expect(TT.LPAREN)
        args = self._parse_expr_list()
        self._expect(TT.RPAREN)
        return WriteStatement(tuple(args), line=tok.line, column=tok.column)

    def _parse_readln(self) -> ReadlnStatement:
        tok     = self._expect(TT.READLN)
        targets = self._parse_optional_var_list()
        return ReadlnStatement(tuple(targets), line=tok.line, column=tok.column)

    def _parse_read(self) -> ReadStatement:
        tok = self._expect(TT.READ)
        self._expect(TT.LPAREN)
        targets = self._parse_var_list()
        self._expect(TT.RPAREN)
        return ReadStatement(tuple(targets), line=tok.line, column=tok.column)

    # ── argument / variable list helpers ─────────────────────

    def _parse_optional_arg_list(self) -> list[ASTNode]:
        if self._consume(TT.LPAREN):
            args = self._parse_expr_list()
            self._expect(TT.RPAREN)
            return args
        return []

    def _parse_expr_list(self) -> list[ASTNode]:
        items = [self._parse_expression()]
        while self._consume(TT.COMMA):
            items.append(self._parse_expression())
        return items

    def _parse_optional_var_list(self) -> list[VarRef]:
        if self._consume(TT.LPAREN):
            refs = self._parse_var_list()
            self._expect(TT.RPAREN)
            return refs
        return []

    def _parse_var_list(self) -> list[VarRef]:
        def one() -> VarRef:
            t = self._expect(TT.IDENTIFIER)
            d = self._syms.lookup(t.value, t.line, t.column)
            return VarRef(t.value, d, line=t.line, column=t.column)
        items = [one()]
        while self._consume(TT.COMMA):
            items.append(one())
        return items

    # ══════════════════════════════════════════════════════════
    #  Expressions  (precedence climbing)
    # ══════════════════════════════════════════════════════════

    _REL_OPS: dict[TT, BinOp] = {
        TT.EQ:  BinOp.EQ,  TT.NEQ: BinOp.NEQ,
        TT.LT:  BinOp.LT,  TT.GT:  BinOp.GT,
        TT.LTE: BinOp.LTE, TT.GTE: BinOp.GTE,
    }
    _ADD_OPS: dict[TT, BinOp] = {
        TT.PLUS: BinOp.ADD, TT.MINUS: BinOp.SUB, TT.OR: BinOp.OR,
    }
    _MUL_OPS: dict[TT, BinOp] = {
        TT.STAR: BinOp.MUL, TT.DIV: BinOp.DIV,
        TT.MOD:  BinOp.MOD, TT.AND: BinOp.AND,
    }

    def _parse_expression(self) -> ASTNode:
        """expression = simple_expr [rel_op simple_expr]"""
        left = self._parse_simple_expr()
        if self._cur.type in self._REL_OPS:
            op_tok = self._advance()
            op     = self._REL_OPS[op_tok.type]
            right  = self._parse_simple_expr()
            lt = getattr(left,  'var_type', PascalType.UNKNOWN)
            rt = getattr(right, 'var_type', PascalType.UNKNOWN)
            vtype = _relational_type(op, lt, rt, op_tok.line, op_tok.column)
            return BinaryOp(left, op, right, vtype,
                            line=op_tok.line, column=op_tok.column)
        return left

    def _parse_simple_expr(self) -> ASTNode:
        """simple_expr = ['+' | '-'] term {('+' | '-' | 'or') term}"""
        # Optional leading sign
        neg_tok = None
        if self._match(TT.PLUS):
            self._advance()
        elif self._match(TT.MINUS):
            neg_tok = self._advance()

        node = self._parse_term()

        if neg_tok is not None:
            node = UnaryOp(UnOp.NEG, node, PascalType.INTEGER,
                           line=neg_tok.line, column=neg_tok.column)

        while self._cur.type in self._ADD_OPS:
            op_tok = self._advance()
            op     = self._ADD_OPS[op_tok.type]
            right  = self._parse_term()
            lt = getattr(node,  'var_type', PascalType.UNKNOWN)
            rt = getattr(right, 'var_type', PascalType.UNKNOWN)
            if op == BinOp.OR:
                vtype = _logic_type(op, lt, rt, op_tok.line, op_tok.column)
            else:
                vtype = _arith_type(op, lt, rt, op_tok.line, op_tok.column)
            node = BinaryOp(node, op, right, vtype,
                            line=op_tok.line, column=op_tok.column)
        return node

    def _parse_term(self) -> ASTNode:
        """term = factor {('*' | 'div' | 'mod' | 'and') factor}"""
        node = self._parse_factor()
        while self._cur.type in self._MUL_OPS:
            op_tok = self._advance()
            op     = self._MUL_OPS[op_tok.type]
            right  = self._parse_factor()
            lt = getattr(node,  'var_type', PascalType.UNKNOWN)
            rt = getattr(right, 'var_type', PascalType.UNKNOWN)
            if op == BinOp.AND:
                vtype = _logic_type(op, lt, rt, op_tok.line, op_tok.column)
            else:
                vtype = _arith_type(op, lt, rt, op_tok.line, op_tok.column)
            node = BinaryOp(node, op, right, vtype,
                            line=op_tok.line, column=op_tok.column)
        return node

    def _parse_factor(self) -> ASTNode:
        """factor = literal | IDENT | '(' expr ')' | 'not' factor | '-' factor"""
        tok = self._cur

        if tok.type == TT.INTEGER_LITERAL:
            self._advance()
            return IntLiteral(tok.value, line=tok.line, column=tok.column)

        if tok.type == TT.BOOLEAN_LITERAL:
            self._advance()
            return BoolLiteral(tok.value, line=tok.line, column=tok.column)

        if tok.type == TT.CHAR_LITERAL:
            self._advance()
            return CharLiteral(tok.value, line=tok.line, column=tok.column)

        if tok.type == TT.STRING_LITERAL:
            self._advance()
            return StringLiteral(tok.value, line=tok.line, column=tok.column)

        if tok.type == TT.IDENTIFIER:
            self._advance()
            decl = self._syms.lookup(tok.value, tok.line, tok.column)
            return VarRef(tok.value, decl, line=tok.line, column=tok.column)

        if tok.type == TT.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TT.RPAREN)
            return expr

        if tok.type == TT.NOT:
            self._advance()
            operand = self._parse_factor()
            ot = getattr(operand, 'var_type', PascalType.UNKNOWN)
            if ot not in (PascalType.BOOLEAN, PascalType.UNKNOWN):
                raise SemanticError(
                    f"'not' requires Boolean operand, got {ot.name}", tok.line
                )
            return UnaryOp(UnOp.NOT, operand, PascalType.BOOLEAN,
                           line=tok.line, column=tok.column)

        if tok.type == TT.MINUS:
            self._advance()
            operand = self._parse_factor()
            return UnaryOp(UnOp.NEG, operand, PascalType.INTEGER,
                           line=tok.line, column=tok.column)

        raise ParseError(
            f"Unexpected token '{tok.value}' in expression", tok
        )

    # ══════════════════════════════════════════════════════════
    #  Entry point
    # ══════════════════════════════════════════════════════════

    def parse(self) -> Program:
        """
        Parse the full program and return the annotated AST.
        Also assigns stack offsets to all variables.
        """
        tok = self._expect(TT.PROGRAM)
        name_tok = self._expect(TT.IDENTIFIER)
        self._expect(TT.SEMICOLON)

        if self._match(TT.VAR):
            self._parse_var_block()

        self._syms.assign_offsets()

        body = self._parse_block()
        self._expect(TT.DOT)

        if self._cur.type != TT.EOF:
            raise ParseError(
                f"Unexpected tokens after program end", self._cur
            )

        return Program(
            name      = name_tok.value,
            var_decls = tuple(self._syms.all_decls()),
            body      = body,
            line      = tok.line,
            column    = tok.column,
        )
