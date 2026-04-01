#!/usr/bin/env python3
"""
Micro-Pascal-Compiler/compile.py
==========================
Command-line driver.  Glues Lexer → Parser → CodeGenerator.

Usage
-----
    python compile.py source.pas              # prints ASM to stdout
    python compile.py source.pas -o out.asm   # writes to file
    python compile.py source.pas --ast        # pretty-prints AST only

End-to-end test (run without arguments)
----------------------------------------
    python compile.py
Compiles a built-in mini Pascal program and prints the resulting assembly.
"""

import sys
import argparse
from parser   import Parser
from codegen  import CodeGenerator
from ast_nodes import pretty


# ═══════════════════════════════════════════════════════════════
#  Built-in demo program (simpler than FizzBuzz — good for tracing)
# ═══════════════════════════════════════════════════════════════

DEMO_SOURCE = """\
{ Sum of 1..N using a while loop, with if/else output }

program SumDemo;

var
  n     : Integer;
  i     : Integer;
  total : Integer;
  ok    : Boolean;

begin
  n     := 10;
  total := 0;
  i     := 1;
  ok    := true;

  { While loop: accumulate sum }
  while i <= n do
  begin
    total := total + i;
    i     := i + 1
  end;

  { Write the result }
  writeln('Sum 1..10 =');
  writeln(total);

  { For loop: print squares }
  writeln('Squares:');
  for i := 1 to 5 do
    writeln(i * i);

  { Boolean branch }
  ok := (total > 50) and not (total = 100);
  if ok then
    writeln('total is in range')
  else
    writeln('total out of range')
end.
"""

EXPECTED_FRAGMENT = """\
; Expected assembly fragments (abbreviated):
;   main:
;     push   rbp
;     mov    rbp, rsp
;     sub    rsp, 48            ; 6 vars × 8 bytes → aligned to 48
;
;     mov    QWORD [rbp-8],  10   ; n := 10
;     mov    QWORD [rbp-24], 0    ; total := 0
;     mov    QWORD [rbp-16], 1    ; i := 1
;     mov    QWORD [rbp-32], 1    ; ok := true
;
;   .L1:                          ; while i <= n
;     mov    rax, QWORD [rbp-16]
;     cmp    rax, QWORD [rbp-8]
;     jg     .L2
;       total := total + i
;     jmp    .L1
;   .L2:
;
;     writeln(total)  → printf("%lld\n", rax)
;
;   FOR loop generates:
;     mov    QWORD [rbp-16], 1     ; i := 1
;     push   r12
;     mov    r12, 5                ; stop value
;   .L3:
;     cmp    rax, r12
;     jg     .L4
;     ...body (writeln(i*i))...
;     inc    rax
;     jmp    .L3
;   .L4:
;     pop    r12
"""


# ═══════════════════════════════════════════════════════════════
#  Compiler pipeline
# ═══════════════════════════════════════════════════════════════

def compile_source(source: str, show_ast: bool = False) -> str:
    # Phase 1+2: Parse (includes lexing + symbol table)
    ast = Parser(source).parse()

    if show_ast:
        return pretty(ast)

    # Phase 3: Code generation
    return CodeGenerator(ast).generate()


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description='Subset-Pascal → x86-64 NASM compiler'
    )
    ap.add_argument('source', nargs='?', help='Pascal source file')
    ap.add_argument('-o', '--output', help='Output file (default: stdout)')
    ap.add_argument('--ast', action='store_true',
                    help='Print AST instead of assembly')
    args = ap.parse_args()

    if args.source:
        with open(args.source) as f:
            source = f.read()
    else:
        # Self-test mode: use the built-in demo
        source = DEMO_SOURCE

    try:
        result = compile_source(source, show_ast=args.ast)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(1)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(result)
        print(f'Written to {args.output}')
    else:
        print(result)

    if not args.source:
        # Also show the expected output guide
        print(EXPECTED_FRAGMENT)


# ═══════════════════════════════════════════════════════════════
#  Inline self-test (run during demo)
# ═══════════════════════════════════════════════════════════════

def _self_test() -> None:
    """Quick regression: compile the demo, spot-check assembly."""
    asm = compile_source(DEMO_SOURCE)
    lines = asm.splitlines()

    checks = [
        ('global  main',               'global main declaration'),
        ('extern  printf',             'extern printf'),
        ('section .text',              '.text section'),
        ('section .data',              '.data section'),
        ('_fmt_intln',                 'integer format string (with newline)'),
        ('push     rbp',               'function prologue'),
        ('leave',                      'function epilogue'),
        ('printf WRT ..plt',           'printf PLT call'),
        ('idiv',                       'integer division'),
        ('imul',                       'integer multiplication'),
        ('.L',                         'generated labels'),
        ('jz',                         'conditional jump (if/while)'),
        ('setle',                      'relational operator (<=)'),
    ]

    print('=' * 60)
    print('  SELF-TEST: assembly spot-checks')
    print('=' * 60)
    all_ok = True
    for needle, desc in checks:
        found = any(needle in line for line in lines)
        status = 'PASS' if found else 'FAIL'
        if not found:
            all_ok = False
        print(f'  [{status}] {desc}')

    print()
    if all_ok:
        print('  All checks passed.')
    else:
        print('  Some checks FAILED.')
    print()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # Demo mode: show AST, then assembly, then self-test
        print('=' * 60)
        print('  INPUT PASCAL PROGRAM')
        print('=' * 60)
        print(DEMO_SOURCE)

        print('=' * 60)
        print('  AST')
        print('=' * 60)
        print(compile_source(DEMO_SOURCE, show_ast=True))

        print()
        print('=' * 60)
        print('  GENERATED x86-64 ASSEMBLY (NASM)')
        print('=' * 60)
        asm = compile_source(DEMO_SOURCE)
        print(asm)

        print(EXPECTED_FRAGMENT)
        _self_test()
    else:
        main()
