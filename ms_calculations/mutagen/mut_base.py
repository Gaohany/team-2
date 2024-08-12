import ast
from contextlib import suppress
from typing import Dict


class MutationTransformer(ast.NodeTransformer):
    mutation_typ: str = "other"

    def __init__(self, config: Dict):
        self.current_function = "_global"
        self.config = config

        for k, v in config.items():
            setattr(self, k, v)

    def mutate(self, input_ast: ast.AST):
        return self.visit(input_ast)

    def visit(self, node):
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)

        if node.__class__.__name__ == "FunctionDef":
            self.current_function = node.name

        with suppress(AssertionError, StopIteration):
            r = visitor(node)
            if r is None:
                r = self.generic_visit(node)
            return r
        return node


class TrainingMutation(MutationTransformer):
    mutation_typ: str = "train"


class ModelMutation(MutationTransformer):
    mutation_typ: str = "model"


class EvalMutation(MutationTransformer):
    mutation_typ: str = "eval"
