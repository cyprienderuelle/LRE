import re
import random

class samples:
    def __init__(self, name, function, type=-1):
        self.name = name
        self.function = function
        self.type = type
        self.variants = []

def maybe(prob=0.5):
    return random.random() < prob

def insert_useless_casts(code, prob=0.5):
    if not maybe(prob):
        return code
    code = re.sub(r'\b(\d+)\b', lambda m: f"(int){m.group(1)}" if maybe(prob) else m.group(1), code)
    code = re.sub(r"'(.)'", lambda m: f"(char)\'{m.group(1)}\'" if maybe(prob) else m.group(0), code)
    return code

def replace_simple_arithmetic(code, prob=0.6):
    if not maybe(prob):
        return code
        
    # Transformation 1: x + 1 => x + 2 - 1
    code = re.sub(r'(\b\w+\b)\s*\+\s*1', lambda m: f"{m.group(1)} + 2 - 1" if maybe(0.5) else m.group(0), code)
    
    # Transformation 2: x * 2 => x << 1
    code = re.sub(r'(\b\w+\b)\s*\*\s*2', lambda m: f"{m.group(1)} << 1" if maybe(0.5) else m.group(0), code)
    
    # Transformation 3: x - 1 => x - (y / y) (requires complex parsing, simplified here)
    code = re.sub(r'(\b\w+\b)\s*-\s*1', lambda m: f"{m.group(1)} - 5 / 5" if maybe(0.2) else m.group(0), code)
    
    # Transformation 4: (condition) => (condition || 0)
    code = re.sub(r'(if|while|for)\s*\((.*?)\)', 
                  lambda m: f"{m.group(1)} ({m.group(2)} || 0)" if maybe(0.3) else m.group(0), code)

    return code

def replace_shorthand_ops(code, prob=0.5):
    if not maybe(prob):
        return code
    code = re.sub(r'(\b\w+\b)\s*\+\+', lambda m: f"{m.group(1)} = {m.group(1)} + 1" if maybe(prob) else m.group(0), code)
    code = re.sub(r'(\b\w+\b)\s*--', lambda m: f"{m.group(1)} = {m.group(1)} - 1" if maybe(prob) else m.group(0), code)
    code = re.sub(r'(\b\w+\b)\s*\+=\s*(\w+)', lambda m: f"{m.group(1)} = {m.group(1)} + {m.group(2)}" if maybe(prob) else m.group(0), code)
    code = re.sub(r'(\b\w+\b)\s*-=\s*(\w+)', lambda m: f"{m.group(1)} = {m.group(1)} - {m.group(2)}" if maybe(prob) else m.group(0), code)
    return code

def brace_newline_style(code, prob=0.5):
    if not maybe(prob):
        return code
    return re.sub(r'(\b(if|for|while|else|switch)\b[^\n{]*)\s*{', r'\1\n{', code)


def rename_identifiers(code):

    prefixes = [
        "var", "tmp", "foo", "bar", "baz", "qux", "val", "tmpvar", "temp", "data",
        "node", "item", "elem", "cnt", "idx", "num", "flag", "res", "ptr", "obj",
        "valeur", "comp", "test", "buff", "buf", "aux", "param", "arg", "val1", "val2",
        "var1", "var2", "tmp1", "tmp2", "result", "counter", "index", "element", "item1"
    ]

    pattern = re.compile(r'\b(int|char|float|double|long|short|void)\s+(\**)(\w+)\b')
    replacements = {}
    counter = 1
    
    def replacer(match):
        nonlocal counter
        type_, stars, name = match.groups()
        if name not in ["main", "return"]:
            prefix = random.choice(prefixes)
            new_name = f"{prefix}_{counter}"
            replacements[name] = new_name
            counter += 1
            return f"{type_} {stars}{new_name}"
        return match.group(0)
    
    code = pattern.sub(replacer, code)
    
    for old, new in replacements.items():
        code = re.sub(rf'\b{old}\b', new, code)
        
    return code


def shuffle_declarations(code):
    lines = code.splitlines()
    new_lines = []
    decls = []
    
    for line in lines:
        stripped = line.strip()
        if re.match(r'(int|char|float|double|long|short)\s+[^=;]+(=.+)?;', stripped):
            decls.append(line)
        else:
            if decls:
                random.shuffle(decls)
                new_lines.extend(decls)
                decls = []
            new_lines.append(line)
    
    if decls:
        random.shuffle(decls)
        new_lines.extend(decls)
    
    return "\n".join(new_lines)

def insert_dead_code(code):
    dead_lines = [
    "int __dead_var = 0;",
    "if (0) { printf(\"never\"); }",
    "/* noop */",
    "volatile int __unused = 42;",
    "do { } while (0);",
    "if (0) { /* unreachable */ }",
    "int __dummy = (0);",
    "while (0) {}",
    "(void)0;",
    "/* dead code */",
    "for (int __i = 0; __i < 0; __i++) {}",
    "if (0) return;",
    "switch(0) { default: break; }",
    "int __zero = 0;",
    "asm(\"\");",
    "((void)0);",
    ]

    lines = code.splitlines() # Plus propre que split("\n")
    
    # Sécurité : On n'insère que s'il y a assez de lignes pour choisir un milieu
    if len(lines) > 2:
        insert_at = random.randint(1, len(lines) - 2)
        lines.insert(insert_at, random.choice(dead_lines))
    else:
        # Pour les fonctions très courtes, on ajoute juste à la fin avant l'accolade
        lines.insert(max(0, len(lines)-1), random.choice(dead_lines))
        
    return "\n".join(lines)

def normalize_blocks(code):
    code = re.sub(r'(if\s*\(.*?\))\s*([^\s{][^\n;]*;)', r'\1 { \2 }', code)
    code = re.sub(r'(for\s*\(.*?\))\s*([^\s{][^\n;]*;)', r'\1 { \2 }', code)
    return code

def AugmentCode(code: samples):
    
    transformations = [
        (rename_identifiers, "RENAME IDENTIFIERS"),
        (shuffle_declarations, "SHUFFLE DECLARATIONS"),
        (insert_dead_code, "INSERT DEAD CODE"),
        (normalize_blocks, "NORMALIZE BLOCKS"),
        (insert_useless_casts, "INSERT USELESS CASTS"),
        (replace_shorthand_ops, "REPLACE SHORTHAND OPS"),
        (brace_newline_style, "BRACE NEWLINE STYLE"),
        (replace_simple_arithmetic, "REPLACE SIMPLE ARITHMETIC (x+1)")
    ]

    num_transforms = random.randint(6, len(transformations))
    selected_transforms = random.sample(transformations, num_transforms)

    current_code = code.function

    for i, (transform_func, transform_name) in enumerate(selected_transforms, 1):
        current_code = transform_func(current_code)

    res = samples(code.name, current_code, code.type)
    return res