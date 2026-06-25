"""Tokenizer: converts APL source code into tokens."""


def tokenize(code: str) -> list:
    """Split an APL expression into a list of (type, value) tokens."""
    tokens = []
    i = 0
    
    while i < len(code):
        c = code[i]
        
        # Whitespace
        if c.isspace():
            i += 1
            continue
        
        # Comments
        if c == '\u235d':  # lamp
            while i < len(code) and code[i] != '\n':
                i += 1
            continue
        if code[i:i+2] == '//':
            while i < len(code) and code[i] != '\n':
                i += 1
            continue
        
        # Multi-character symbols
        if code[i:i+2] == '+/':
            tokens.append(('REDUCE_SUM', '+/'))
            i += 2
            continue
        if code[i:i+2] == '<-':
            tokens.append(('ASSIGN', '<-'))
            i += 2
            continue
        
        # Single-character symbols with dedicated token types
        if c == ':':
            tokens.append(('COLON', c))
            i += 1
            continue
        if c == ',':
            tokens.append(('COMMA', c))
            i += 1
            continue
        if c == '=':
            tokens.append(('EQUALS', c))
            i += 1
            continue
        
        # Binary operators: + - x / ^
        if c in '+x/^':
            tokens.append(('OP', c))
            i += 1
            continue
        
        if c == '-':
            prev_types = [t[0] for t in tokens]
            if not tokens or prev_types[-1] in (
                'OP', 'LPAREN', 'ASSIGN', 'COMMA', 'LBRACKET', 'EQUALS', 'COLON'
            ):
                tokens.append(('NEG', c))
            else:
                tokens.append(('OP', c))
            i += 1
            continue
        
        # Brackets and bars
        if c == '|':
            tokens.append(('ABS', c))
            i += 1
            continue
        if c == '(':
            tokens.append(('LPAREN', c))
            i += 1
            continue
        if c == ')':
            tokens.append(('RPAREN', c))
            i += 1
            continue
        if c == '[':
            tokens.append(('LBRACKET', c))
            i += 1
            continue
        if c == ']':
            tokens.append(('RBRACKET', c))
            i += 1
            continue
        
        # Unicode left arrow
        if c == '\u2190':
            tokens.append(('ASSIGN', '<-'))
            i += 1
            continue
        
        # Words: functions, variables, kwargs
        if c.isalpha() or c == '_':
            start = i
            while i < len(code) and (code[i].isalnum() or code[i] == '_'):
                i += 1
            word = code[start:i]
            
            if word in (
                'mean', 'var', 'norm', 'sum', 'max', 'min',
                'sqrt', 'log', 'exp', 'abs', 'std',
                'softmax', 'topk', 'threshold', 'where', 'rank', 'sort', 'count'
            ):
                tokens.append(('FUNC', word))
            elif word in ('dim', 'axis'):
                tokens.append(('KWARG', word))
            else:
                tokens.append(('VAR', word))
            continue
        
        # Numbers
        if c.isdigit() or (c == '.' and i+1 < len(code) and code[i+1].isdigit()):
            start = i
            while i < len(code) and (code[i].isdigit() or code[i] == '.'):
                i += 1
            num_str = code[start:i]
            tokens.append(('NUM', float(num_str) if '.' in num_str else int(num_str)))
            continue
        
        i += 1
    
    return tokens
