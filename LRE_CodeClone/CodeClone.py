import numpy as np
import torch

class samples:
    def __init__(self, name, function, type=-1):
        self.name = name
        self.function = function
        self.type = type
        self.variants = []

class transform1:
    def __init__(self, window_size: int):
        self.window_size = window_size

    def __call__(self, sample):
        text = sample.function
        lines = text.split("\n")

        if len(lines) <= 2:
            return sample

        first_idx, last_idx = 1, max(2, len(lines) - 1 - self.window_size)

        if last_idx <= first_idx:
            window_start = first_idx
        else:
            window_start = np.random.randint(first_idx, last_idx)

        window_end = min(len(lines), window_start + self.window_size)
        sample.function = "\n".join(lines[window_start:window_end])
        return sample

class transform2:
    def __init__(self):
        pass

    def __call__(self, sample):
        text = sample.function
        lines = text.split("\n")

        list_lines = []
        is_inside_comment = False

        for line in lines:
            line_strip = line.strip()

            if "/*" in line_strip:
                is_inside_comment = True
            if not is_inside_comment:
                list_lines.append(line)
            if "*/" in line_strip:
                is_inside_comment = False

        sample.function = "\n".join(list_lines)
        return sample

class tranform3:
    def __init__(self):
        self.type_mapping = {
            'int': 'long',
            'float': 'double',
            'char': 'unsigned char',
        }

    def __call__(self, sample):
        text = sample.function
        for old_type, new_type in self.type_mapping.items():
            text = text.replace(old_type, new_type)
        sample.function = text
        return sample

class transform4:
    def __init__(self, prefix="my_"):

        self.prefix = prefix

    def __call__(self, sample):
        text = sample.function
        lines = text.split("\n")

        reserved_words = [
            "int", "float", "char", "double", "long", "short", "void", "return",
            "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
            "default", "sizeof", "typedef", "struct", "union", "enum", "const",
            "static", "extern", "register", "volatile", "goto", "signed", "unsigned",
            "main", "printf", "scanf", "puts", "gets", "fopen", "fclose", "fread",
            "fwrite", "malloc", "calloc", "free", "NULL", "true", "false", "include"
        ]

        prefixes = [
            "custom_",
            "user_",
            "gen_",
            "mod_",
            "safe_",
            "secure_",
            "alt_",
            "dev_",
            "build_",
            "v1_",
            "v2_",
            "tmp_",
            "test_",
            "try_",
            "draft_",
            "wip_",
            "sandbox_",
            "scratch_",
            "project_",
            "myproj_",
            "clone_",
            "user42_",
            "ai_",
            "lab_",
            "zz_",
            "x_",
            "yo_",
            "lol_",
            "neo_",
            "hax_",
            "glitch_",
            "alex_",
            "exp2025_",
            "code_",
            "bot_",
            "alpha_",
            "beta_"
        ]

        new_lines = []
        for line in lines:
            words = line.split()
            for i, word in enumerate(words):
                if word.isidentifier() and word not in reserved_words:
                    words[i] = np.random.choice(prefixes) + word
            new_lines.append(" ".join(words))

        sample.function = "\n".join(new_lines)
        return sample

class transform5:
    def __init__(self):
        pass

    def __call__(self, sample):
        text = sample.function
        lines = text.split("\n")
        formatted_lines = [line.strip() for line in lines]
        sample.function = "\n".join(formatted_lines)
        return sample