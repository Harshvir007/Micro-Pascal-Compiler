"""
pascal_compiler/lexer.py
========================
Single-pass lexer for subset-Pascal.

Supported features
------------------
* Case-insensitive keywords  (VAR == var == Var)
* Comment styles:  { ... }  and  (* ... *)
* Integer, Boolean (true/false), Char ('x'), String ('hello') literals
* All Pascal operators and delimiters including  :=  <>  ..
* Rich source-position info (line, column) on every token
"""

from __future__ import annotations
from enum      import Enum, auto
from dataclasses import dataclass
from typing    import Optional


# ═══════════════════════════════════════════════════════════════
#  1.  Token Types
# ═══════════════════════════════════════════════════════════════

class TT(Enum):
    # ── Literals ─────────────────────────────────────────────
    INTEGER_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    CHAR_LITERAL    = auto()
    STRING_LITERAL  = auto()

    # ── Names ────────────────────────────────────────────────
    IDENTIFIER      = auto()

    # ── Keywords ─────────────────────────────────────────────
    PROGRAM         = auto()
    VAR             = auto()
    BEGIN           = auto()
    END             = auto()
    IF              = auto()
    THEN            = auto()
    ELSE            = auto()
    WHILE           = auto()
    DO              = auto()
    FOR             = auto()
    TO              = auto()
    DOWNTO          = auto()
    WRITELN         = auto()
    WRITE           = auto()
    READLN          = auto()
    READ            = auto()
    AND             = auto()
    OR              = auto()
    NOT             = auto()
    DIV             = auto()
    MOD             = auto()
    # ── Type keywords ────────────────────────────────────────
    INTEGER_TYPE    = auto()   # 'Integer'
    BOOLEAN_TYPE    = auto()   # 'Boolean'
    CHAR_TYPE       = auto()   # 'Char'

    # ── Arithmetic operators ──────────────────────────────────
    PLUS            = auto()   # +
    MINUS           = auto()   # -
    STAR            = auto()   # *

    # ── Relational operators ──────────────────────────────────
    EQ              = auto()   # =
    NEQ             = auto()   # <>
    LT              = auto()   # <
    GT              = auto()   # >
    LTE             = auto()   # <=
    GTE             = auto()   # >=

    # ── Assignment ───────────────────────────────────────────
    ASSIGN          = auto()   # :=

    # ── Delimiters ────────────────────────────────────────────
    LPAREN          = auto()   # (
    RPAREN          = auto()   # )
    LBRACKET        = auto()   # [
    RBRACKET        = auto()   # ]
    SEMICOLON       = auto()   # ;
    COLON           = auto()   # :
    COMMA           = auto()   # ,
    DOT             = auto()   # .
    DOTDOT          = auto()   # ..

    # ── End-of-file ───────────────────────────────────────────
    EOF             = auto()


# ═══════════════════════════════════════════════════════════════
#  2.  Keyword map  (lower-case string → TT)
# ═══════════════════════════════════════════════════════════════

KEYWORDS: dict[str, TT] = {
    "program":  TT.PROGRAM,
    "var":      TT.VAR,
    "begin":    TT.BEGIN,
    "end":      TT.END,
    "if":       TT.IF,
    "then":     TT.THEN,
    "else":     TT.ELSE,
    "while":    TT.WHILE,
    "do":       TT.DO,
    "for":      TT.FOR,
    "to":       TT.TO,
    "downto":   TT.DOWNTO,
    "writeln":  TT.WRITELN,
    "write":    TT.WRITE,
    "readln":   TT.READLN,
    "read":     TT.READ,
    "true":     TT.BOOLEAN_LITERAL,   # resolved directly to literal
    "false":    TT.BOOLEAN_LITERAL,
    "and":      TT.AND,
    "or":       TT.OR,
    "not":      TT.NOT,
    "div":      TT.DIV,
    "mod":      TT.MOD,
    "integer":  TT.INTEGER_TYPE,
    "boolean":  TT.BOOLEAN_TYPE,
    "char":     TT.CHAR_TYPE,
}


# ═══════════════════════════════════════════════════════════════
#  3.  Token dataclass
# ═══════════════════════════════════════════════════════════════

@dataclass(slots=True)
class Token:
    type   : TT
    value  : object    # int | bool | str | None
    line   : int
    column : int

    def __repr__(self) -> str:
        return f"Token({self.type.name:<18} {self.value!r:<12} @ {self.line}:{self.column})"


# ═══════════════════════════════════════════════════════════════
#  4.  Lexer error
# ═══════════════════════════════════════════════════════════════

class LexerError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"[Lexer] {msg}  (line {line}, col {col})")
        self.line = line
        self.col  = col


# ═══════════════════════════════════════════════════════════════
#  5.  Lexer
# ═══════════════════════════════════════════════════════════════

class Lexer:
    """
    Convert a Pascal source string into a flat list of Tokens.

    Example
    -------
    >>> tokens = Lexer(source).tokenize()
    """

    def __init__(self, source: str):
        self._src  = source
        self._pos  = 0
        self._line = 1
        self._col  = 1

    # ── character-level helpers ───────────────────────────────

    @property
    def _ch(self) -> Optional[str]:
        """Current character (None at EOF)."""
        return self._src[self._pos] if self._pos < len(self._src) else None

    @property
    def _peek(self) -> Optional[str]:
        """One character ahead."""
        p = self._pos + 1
        return self._src[p] if p < len(self._src) else None

    def _advance(self) -> str:
        """Consume current character, update position tracking."""
        ch = self._src[self._pos]
        self._pos += 1
        if ch == '\n':
            self._line += 1
            self._col   = 1
        else:
            self._col  += 1
        return ch

    def _match(self, expected: str) -> bool:
        """Consume and return True if current character == expected."""
        if self._ch == expected:
            self._advance()
            return True
        return False

    def _err(self, msg: str) -> LexerError:
        return LexerError(msg, self._line, self._col)

    # ── whitespace / comment skipping ────────────────────────

    def _skip(self) -> None:
        while self._ch is not None:
            # Plain whitespace
            if self._ch in ' \t\r\n':
                self._advance()

            # Curly-brace comment  { ... }
            elif self._ch == '{':
                self._advance()
                while self._ch is not None and self._ch != '}':
                    self._advance()
                if self._ch is None:
                    raise self._err("Unterminated comment '{ ... }'")
                self._advance()   # consume '}'

            # Old-style comment  (* ... *)
            elif self._ch == '(' and self._peek == '*':
                self._advance(); self._advance()
                while self._ch is not None:
                    if self._ch == '*' and self._peek == ')':
                        self._advance(); self._advance()
                        break
                    self._advance()
                else:
                    raise self._err("Unterminated comment '(* ... *)'")

            else:
                break

    # ── token readers ─────────────────────────────────────────

    def _read_number(self) -> Token:
        line, col = self._line, self._col
        buf: list[str] = []
        while self._ch is not None and self._ch.isdigit():
            buf.append(self._advance())
        return Token(TT.INTEGER_LITERAL, int(''.join(buf)), line, col)

    def _read_string(self) -> Token:
        """
        Single-quoted literal.  Two consecutive apostrophes '' mean one '.
        Yields CHAR_LITERAL for length-1 strings, STRING_LITERAL otherwise.
        """
        line, col = self._line, self._col
        self._advance()   # opening '
        buf: list[str] = []
        while True:
            if self._ch is None:
                raise self._err("Unterminated string literal")
            if self._ch == "'":
                self._advance()           # consume ' or first of ''
                if self._ch == "'":       # escaped apostrophe
                    buf.append("'")
                    self._advance()
                else:
                    break                 # end of string
            else:
                buf.append(self._advance())
        text = ''.join(buf)
        tt   = TT.CHAR_LITERAL if len(text) == 1 else TT.STRING_LITERAL
        return Token(tt, text, line, col)

    def _read_word(self) -> Token:
        """Identifier or keyword (case-insensitive)."""
        line, col = self._line, self._col
        buf: list[str] = []
        while self._ch is not None and (self._ch.isalnum() or self._ch == '_'):
            buf.append(self._advance())
        raw   = ''.join(buf)
        lower = raw.lower()
        tt    = KEYWORDS.get(lower)

        if tt == TT.BOOLEAN_LITERAL:
            return Token(TT.BOOLEAN_LITERAL, lower == 'true', line, col)
        if tt is not None:
            return Token(tt, lower, line, col)
        return Token(TT.IDENTIFIER, raw, line, col)

    # ── main dispatch ─────────────────────────────────────────

    def _next(self) -> Token:
        self._skip()
        line, col = self._line, self._col

        if self._ch is None:
            return Token(TT.EOF, None, line, col)

        ch = self._ch

        if ch.isdigit():             return self._read_number()
        if ch == "'":                return self._read_string()
        if ch.isalpha() or ch == '_': return self._read_word()

        self._advance()   # consume the character before building token

        match ch:
            case '+': return Token(TT.PLUS,      '+',  line, col)
            case '-': return Token(TT.MINUS,     '-',  line, col)
            case '*': return Token(TT.STAR,      '*',  line, col)
            case '=': return Token(TT.EQ,        '=',  line, col)
            case ';': return Token(TT.SEMICOLON, ';',  line, col)
            case ',': return Token(TT.COMMA,     ',',  line, col)
            case '(': return Token(TT.LPAREN,    '(',  line, col)
            case ')': return Token(TT.RPAREN,    ')',  line, col)
            case '[': return Token(TT.LBRACKET,  '[',  line, col)
            case ']': return Token(TT.RBRACKET,  ']',  line, col)
            case '.':
                if self._match('.'):
                    return Token(TT.DOTDOT, '..', line, col)
                return Token(TT.DOT, '.', line, col)
            case ':':
                if self._match('='):
                    return Token(TT.ASSIGN, ':=', line, col)
                return Token(TT.COLON, ':', line, col)
            case '<':
                if self._match('>'):
                    return Token(TT.NEQ,  '<>', line, col)
                if self._match('='):
                    return Token(TT.LTE,  '<=', line, col)
                return Token(TT.LT, '<', line, col)
            case '>':
                if self._match('='):
                    return Token(TT.GTE,  '>=', line, col)
                return Token(TT.GT, '>', line, col)
            case _:
                raise self._err(f"Unexpected character '{ch}'")

    def tokenize(self) -> list[Token]:
        """Return all tokens, ending with EOF."""
        tokens: list[Token] = []
        while True:
            tok = self._next()
            tokens.append(tok)
            if tok.type == TT.EOF:
                break
        return tokens
