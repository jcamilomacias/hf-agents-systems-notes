# Tools & Actions

## What is a tool?

A **tool** is a function the agent can call to interact with the external world. It has:

- A **name** (used by the LLM to reference it)
- A **description** (tells the LLM when and how to use it)
- A typed **signature** (arguments and return type)

## Defining a tool with smolagents

```python
from smolagents import tool

@tool
def get_weather(city: str) -> str:
    """Return current weather for a given city.

    Args:
        city: Name of the city, e.g. 'Paris'.
    """
    # ... call a weather API ...
    return f"Sunny, 22°C in {city}"
```

The `@tool` decorator automatically extracts the JSON schema from the docstring and type hints, so the LLM knows exactly how to call it.

## Tool categories

| Category | Examples |
|----------|---------|
| Information retrieval | Web search, Wikipedia lookup, RAG |
| Computation | Python REPL, calculator |
| I/O | File read/write, email, calendar |
| External services | APIs, databases |
| Perception | Image captioning, OCR |

## Notes


## Python decorators

A **decorator** is a function that wraps another function to extend or modify its behavior without changing its source code. The `@decorator` syntax is shorthand for `func = decorator(func)`.

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before the call")
        result = func(*args, **kwargs)
        print("After the call")
        return result
    return wrapper

@my_decorator
def greet(name: str) -> str:
    return f"Hello, {name}!"

greet("Alice")
# Before the call
# After the call
```

### Why decorators matter for tools

The `@tool` decorator in `smolagents` inspects the wrapped function's **type hints** and **docstring** at decoration time, builds a JSON schema from them, and registers the function as a callable tool. When the LLM decides to use the tool, the framework validates arguments against that schema before calling the underlying Python function.

This pattern — extract metadata at decoration time, validate at call time — is common in Python frameworks (FastAPI routes, pytest fixtures, click commands, etc.)

### Key built-in decorators

| Decorator | Purpose |
|-----------|---------|
| `@staticmethod` | Method that receives no implicit first argument |
| `@classmethod` | Method that receives the class (`cls`) as first argument |
| `@property` | Expose a method as a read-only attribute |
| `@functools.wraps` | Preserve the wrapped function's `__name__` and `__doc__` |

Always apply `@functools.wraps(func)` inside your decorator so that introspection tools (and frameworks like `smolagents`) still see the original function's metadata:

```python
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

## Let's break the code a little bit

`final_answer.py`

In `final_answer.py` we have a `FinalAnswerTool` class that inherits from smolagents' `Tool`. Its `forward` method simply returns its input unchanged — it is the terminal tool in the agent loop: when the agent calls it, execution stops and the value is surfaced as the final answer.

```python
class FinalAnswerTool(Tool):
    name = "final_answer"
    description = "Provides a final answer to the given problem."
    inputs = {'answer': {'type': 'any', 'description': 'The final answer to the problem'}}
    output_type = "any"

    def forward(self, answer: Any) -> Any:
        return answer
```

### Class attributes vs. instance attributes

`name`, `description`, `inputs`, and `output_type` are **class attributes** — defined directly in the class body and shared by all instances. The smolagents framework reads them at class-definition time to register and use the tool; it's a convention the `Tool` base class expects.

Compare with instance attributes (set inside `__init__`):

```python
def __init__(self):
    self.is_initialized = False  # instance attribute — each object has its own copy
```

| | Class attribute | Instance attribute |
|---|---|---|
| Defined | In class body | Inside `__init__` via `self` |
| Shared | By all instances | Unique per instance |
| Accessed | `FinalAnswerTool.name` or `self.name` | Only via `self.name` |

`name`, `description`, `inputs`, and `output_type` are metadata the framework reads at class-definition time to understand what the tool is and what arguments it accepts — before any instance is even created.
