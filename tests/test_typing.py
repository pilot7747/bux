"""Arrays say their dtype and shape, with jaxtyping, and the tests check them.

The tests run with every function that handles arrays checked on every call
(tests/conftest.py), so that an array of the wrong dtype or shape fails where it goes
in, not somewhere downstream. That checks the annotations there are; this checks that
every array has one.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import jax.numpy as jnp
import jaxtyping
import pytest

from bux import config as cfg
from bux.architecture.rope import rotate
from bux.train.objective import z_loss

ROOT = Path(__file__).parents[1]
FILES = sorted([*ROOT.glob("src/bux/**/*.py"), *ROOT.glob("tests/*.py")])
# Array types, which annotations put in a dtype of jaxtyping's, Float[Array, "b t d"]:
# by name, or as attributes of these modules (jax.Array, np.ndarray, ...).
ARRAYS = {"Array", "ArrayLike", "ndarray", "NDArray"}
MODULES = {"jax", "jnp", "jaxtyping", "np", "numpy", "npt"}
DTYPES = {
    name
    for name, value in vars(jaxtyping).items()
    if isinstance(value, type) and issubclass(value, jaxtyping.AbstractDtype)
}


def _name(node: ast.AST) -> str | None:
    """The last name of `Float` or `jaxtyping.Float`."""
    match node:
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return name
    return None


def _is_array(node: ast.AST) -> bool:
    """Whether `node` names an array type: `Array`, `jax.Array`, `np.ndarray`, ..."""
    match node:
        case ast.Name(id=name):
            return name in ARRAYS
        case ast.Attribute(value=module, attr=name):
            while isinstance(module, ast.Attribute):
                module = module.value
            return (
                name in ARRAYS and isinstance(module, ast.Name) and module.id in MODULES
            )
    return False


def _annotations(tree: ast.Module) -> Iterator[ast.expr]:
    """Every annotation in `tree`, and what its module-level names alias."""
    for node in ast.walk(tree):
        match node:
            case ast.FunctionDef(args=a) | ast.AsyncFunctionDef(args=a):
                for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]:
                    if arg is not None and arg.annotation is not None:
                        yield arg.annotation
                if node.returns is not None:
                    yield node.returns
            case ast.AnnAssign(annotation=annotation):
                yield annotation
            case ast.TypeAlias(value=value):
                yield value
    for node in tree.body:
        if isinstance(node, ast.Assign) and not isinstance(node.value, ast.Constant):
            yield node.value


def _bare(annotation: ast.expr) -> Iterator[ast.AST]:
    """The array types in `annotation` that no dtype wraps."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        annotation = ast.parse(annotation.value, mode="eval").body
    wrapped: set[int] = set()
    for node in ast.walk(annotation):
        match node:
            case ast.Subscript(value=dtype, slice=ast.Tuple(elts=[array, *_])) if (
                _name(dtype) in DTYPES
            ):
                wrapped.update(id(n) for n in ast.walk(array))
    return (n for n in ast.walk(annotation) if _is_array(n) and id(n) not in wrapped)


def test_every_array_says_its_dtype_and_shape():
    bare = [
        f"{path.relative_to(ROOT)}:{annotation.lineno}: {ast.unparse(node)}"
        for path in FILES
        for annotation in _annotations(ast.parse(path.read_text(encoding="utf-8")))
        for node in _bare(annotation)
    ]
    assert not bare, (
        'arrays without a dtype and shape; say them as Float[Array, "b t d"]:\n'
        + "\n".join(bare)
    )


def test_a_call_checks_dtypes_and_shapes():
    assert z_loss(jnp.zeros((2, 3, 5))).shape == ()
    with pytest.raises(jaxtyping.TypeCheckError):
        z_loss(jnp.zeros((2, 3)))  # without the vocabulary's axis
    with pytest.raises(jaxtyping.TypeCheckError):
        z_loss(jnp.zeros((2, 3, 5), jnp.int32))  # ids, not logits


def test_a_call_checks_that_its_arrays_agree():
    x, rope = jnp.zeros((2, 8, 4, 16)), cfg.RoPEConfig()
    assert rotate(x, rope, jnp.zeros((2, 8), jnp.int32)).shape == x.shape
    with pytest.raises(jaxtyping.TypeCheckError):
        rotate(x, rope, jnp.zeros((2, 9), jnp.int32))  # a position too many
