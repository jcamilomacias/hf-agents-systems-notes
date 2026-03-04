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
