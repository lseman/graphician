"""Receiver type hints and cross-library disambiguation for call resolution.

Maps variable/function names to their likely implementation types
across all supported languages. Also provides method-level disambiguation
when the same method name exists in multiple stub types.
"""

from __future__ import annotations

# ── Receiver hints: variable/function names → stub types ────────────────────

_RECEIVER_HINTS: dict[str, str] = {
    # ── Rust patterns ───────────────────────────────────────────────────

    "graph": "Graph", "main_graph": "Graph", "app_graph": "Graph", "g": "Graph",
    "motif": "MotifBuilder", "mb": "MotifBuilder", "motif_builder": "MotifBuilder",
    "store": "Store", "db": "Store", "database": "Store",
    "vec": "Vec", "vecdeque": "VecDeque", "deque": "VecDeque",
    "v": "Vec", "vs": "Vec", "elements": "Vec",
    "hashmap": "HashMap", "map": "HashMap", "dict": "HashMap",
    "hm": "HashMap", "m": "HashMap", "counts": "HashMap",
    "btmap": "BTreeMap", "btree": "BTreeMap", "btree_map": "BTreeMap",
    "btm": "BTreeMap",
    "set": "HashSet", "hs": "HashSet",
    "seen": "HashSet", "visited": "HashSet",
    "string": "String", "s": "String", "text": "String", "buf": "String",
    "builder": "String", "result": "String",
    "path": "Path", "pathbuf": "PathBuf", "p": "PathBuf",
    "opt": "Option", "maybe": "Option", "val": "Option",
    "res": "Result", "ret": "Result", "output": "Result",
    "iter": "Iterator", "it": "Iterator",
    "entry": "HashMap", "e": "HashMap",
    "config": "Config", "cfg": "Config", "options": "Options", "opts": "Options",
    "ctx": "Context", "context": "Context",
    "conn": "Connection", "sqlite": "Connection",
    "rx": "Receiver", "sender": "Sender", "tx": "Sender",
    "pool": "Pool", "client": "Client", "handle": "Handle",

    # ── Python patterns ─────────────────────────────────────────────────

    "items": "list", "values": "dict", "keys": "dict",
    "result": "dict", "data": "dict", "args": "list", "kwargs": "dict",
    "lines": "list", "entries": "list", "records": "list",
    "files": "list", "dirs": "list", "paths": "list",
    "nodes": "list", "edges": "list", "edges_list": "list",
    "config": "dict", "settings": "dict", "options": "dict",
    "cache": "dict", "registry": "dict", "index": "dict",
    "session": "Session", "request": "Request", "response": "Response",
    "ctx": "Context", "context": "Context",
    "logger": "Logger", "log": "Logger",
    "conn": "Connection", "cursor": "Cursor", "db": "Database",
    "client": "Client", "server": "Server", "app": "App",
    "fs": "Path", "path": "Path", "p": "Path",

    # ── Exception receiver hints ────────────────────────────────────────

    "ValueError": "ValueError", "KeyError": "KeyError", "TypeError": "TypeError",
    "IndexError": "IndexError", "AttributeError": "AttributeError",
    "ImportError": "ImportError", "ModuleNotFoundError": "ModuleNotFoundError",
    "FileNotFoundError": "FileNotFoundError", "PermissionError": "PermissionError",
    "OSError": "OSError", "RuntimeError": "RuntimeError", "StopIteration": "StopIteration",
    "SystemExit": "SystemExit", "KeyboardInterrupt": "KeyboardInterrupt",
    "RecursionError": "RecursionError", "OverflowError": "OverflowError",
    "ZeroDivisionError": "ZeroDivisionError", "FloatingPointError": "FloatingPointError",
    "NotImplementedError": "NotImplementedError", "EnvironmentError": "EnvironmentError",
    "IOError": "IOError", "BufferError": "BufferError", "ArithmeticError": "ArithmeticError",
    "LookupError": "LookupError", "AssertionError": "AssertionError",
    "EOFError": "EOFError", "ConnectionError": "ConnectionError",
    "BrokenPipeError": "BrokenPipeError", "ConnectionAbortedError": "ConnectionAbortedError",
    "ConnectionRefusedError": "ConnectionRefusedError", "ConnectionResetError": "ConnectionResetError",
    "TimeoutError": "TimeoutError", "NotADirectoryError": "NotADirectoryError",
    "IsADirectoryError": "IsADirectoryError", "ProcessLookupError": "ProcessLookupError",
    "ChildProcessError": "ChildProcessError", "SyntaxError": "SyntaxError",
    "SystemError": "SystemError", "ReferenceError": "ReferenceError",
    "MemoryError": "MemoryError", "Warning": "Warning", "UserWarning": "UserWarning",
    "DeprecationWarning": "DeprecationWarning", "FutureWarning": "FutureWarning",
    "PendingDeprecationWarning": "PendingDeprecationWarning",
    "ImportWarning": "ImportWarning", "SyntaxWarning": "SyntaxWarning",
    "ResourceWarning": "ResourceWarning", "BytesWarning": "BytesWarning",
    "UnicodeWarning": "UnicodeWarning", "UnicodeError": "UnicodeError",
    "UnicodeDecodeError": "UnicodeDecodeError", "UnicodeEncodeError": "UnicodeEncodeError",
    "UnicodeTranslateError": "UnicodeTranslateError", "TabError": "TabError",
    "IndentationError": "IndentationError",

    # ── External class receiver hints ───────────────────────────────────

    "SentenceTransformer": "SentenceTransformer", "HTMLParser": "HTMLParser",
    "HTTPServer": "HTTPServer", "Popen": "Popen", "ThreadPoolExecutor": "ThreadPoolExecutor",
    "MagicMock": "MagicMock", "Query": "Query", "QueryCursor": "QueryCursor",
    "TreeSitterLanguage": "TreeSitterLanguage", "RustAnalyzerOptions": "RustAnalyzerOptions",
    "WeightedPath": "WeightedPath", "PythonFilter": "PythonFilter",
    "Client": "Client", "Confidence": "Confidence", "PathQuery": "PathQuery",
    "ImpactQuery": "ImpactQuery", "LanguageRegistry": "LanguageRegistry",
    "ExtractionPipeline": "ExtractionPipeline", "GraphStore": "GraphStore",
    "CommunityOptions": "CommunityOptions", "EmbeddingIndex": "EmbeddingIndex",
    "ExternalEmbeddingConfig": "ExternalEmbeddingConfig", "NetworkX": "NetworkX",

    # ── NumPy patterns ──────────────────────────────────────────────────

    "np": "numpy", "num": "numpy",
    "arr": "ndarray", "a": "ndarray", "x": "ndarray", "y": "ndarray",
    "matrix": "ndarray", "M": "ndarray", "v": "ndarray", "w": "ndarray",
    "data": "ndarray", "values": "ndarray", "tensor": "ndarray",
    "t": "ndarray", "feature": "ndarray", "features": "ndarray",
    "row": "ndarray", "col": "ndarray", "index": "ndarray",
    "rng": "random", "generator": "random", "rs": "random",
    "state": "random", "bitgen": "random",
    "zeros": "numpy", "ones": "numpy", "empty": "numpy", "full": "numpy",
    "array": "numpy", "arange": "numpy", "linspace": "numpy", "logspace": "numpy",
    "eye": "numpy", "identity": "numpy", "diag": "numpy",
    "sort": "numpy", "argsort": "numpy", "partition": "numpy",
    "unique": "numpy", "intersect1d": "numpy", "union1d": "numpy",
    "any": "numpy", "all": "numpy", "count_nonzero": "numpy",
    "sum": "numpy", "mean": "numpy", "std": "numpy", "var": "numpy",
    "min": "numpy", "max": "numpy", "argmin": "numpy", "argmax": "numpy",
    "median": "numpy", "percentile": "numpy", "quantile": "numpy",
    "diff": "numpy", "gradient": "numpy", "cross": "numpy",
    "cumsum": "numpy", "cumprod": "numpy", "clip": "numpy",
    "repeat": "numpy", "tile": "numpy", "rot90": "numpy",
    "insert": "numpy", "delete": "numpy", "resize": "numpy",
    "searchsorted": "numpy", "extract": "numpy", "put": "numpy",
    "round": "numpy", "floor": "numpy", "ceil": "numpy",
    "log": "numpy", "log10": "numpy", "log2": "numpy", "exp": "numpy",
    "sqrt": "numpy", "power": "numpy", "multiply": "numpy",
    "divide": "numpy", "subtract": "numpy", "add": "numpy",
    "equal": "numpy", "not_equal": "numpy", "greater": "numpy",
    "less": "numpy", "logical_and": "numpy", "logical_or": "numpy",
    "where": "numpy", "select": "numpy", "copy": "numpy",
    "zeros_like": "numpy", "ones_like": "numpy", "empty_like": "numpy",
    "full_like": "numpy", "astype": "numpy", "reshape": "numpy",
    "intp": "numpy", "int64": "numpy", "int32": "numpy", "float64": "numpy",
    "float32": "numpy", "bool_": "numpy", "uint8": "numpy", "uint64": "numpy",

    # ── JavaScript patterns ─────────────────────────────────────────────

    "arr": "Array", "list": "Array", "items": "Array", "elements": "Array",
    "map": "Map", "dict": "Map", "obj_map": "Map",
    "set": "Set", "unique": "Set", "distinct": "Set",
    "promise": "Promise",
    "str": "String", "text": "String", "msg": "String",
    "err": "Error", "error": "Error",
    "url": "URL", "search": "URLSearchParams",
}


# ── Cross-library method disambiguation ──────────────────────────────────────
# Maps method names → preferred stub type when the same method exists in
# multiple languages. Uses Rust-specific method names to disambiguate first,
# then Python patterns, then C++ patterns.

_METHOD_DISAMBIGUATION: dict[str, str] = {
    # ── Python-specific methods (unique or most common in Python) ─────────
    "setdefault": "dict",
    "fromkeys": "dict",
    "get_or_default": "dict",
    "popitem": "dict",
    "popitem": "defaultdict",
    "default_factory": "defaultdict",
    "most_common": "Counter",
    "elements": "Counter",
    "total": "Counter",
    "counter": "Counter",
    "counters": "Counter",
    "move_to_end": "OrderedDict",
    "mappingproxy": "Mapping",
    "__getitem__": "dict",
    "__setitem__": "dict",
    "__delitem__": "dict",
    "__len__": "dict",
    "__contains__": "dict",
    "__iter__": "dict",
    "__reversed__": "dict",
    "get_or_insert": "Vec",  # Rust HashMap method but also common in Vec
    "get_or_insert_with": "Vec",
    
    # ── Rust-specific collection methods ─────────────────────────────────
    "or_insert": "HashMap",
    "or_insert_with": "HashMap",
    "entry": "HashMap",
    "remove_entry": "HashMap",
    "into_keys": "HashMap",
    "into_values": "HashMap",
    "get_mut": "HashMap",
    "extract_if": "Vec",
    "drain": "Vec",
    "dedup": "Vec",
    "dedup_by": "Vec",
    "dedup_by_key": "Vec",
    "retain": "Vec",
    "try_push": "Vec",
    "shrink_to": "Vec",
    "resize_with": "Vec",
    "into_boxed_slice": "Vec",
    "as_slice": "Vec",
    "as_mut_slice": "Vec",
    "as_ptr": "Vec",
    "as_mut_ptr": "Vec",
    "to_vec": "Vec",
    "swap_remove": "Vec",
    "rotate_left": "VecDeque",
    "rotate_right": "VecDeque",
    "append": "Vec",
    "extend": "Vec",
    "and_then": "Option",
    "unwrap_or": "Option",
    "unwrap_or_else": "Option",
    "map_or": "Option",
    "map_or_else": "Option",
    "ok_or": "Option",
    "ok_or_else": "Option",
    "is_some": "Option",
    "is_none": "Option",
    "unwrap": "Result",
    "unwrap_or": "Result",
    "unwrap_or_else": "Result",
    "map": "Result",
    "and_then": "Result",
    "or": "Result",
    "expect": "Result",
    "expect_err": "Result",
    "map_err": "Result",
    "is_ok": "Result",
    "is_err": "Result",
    "iter_mut": "Option",
    "as_ref": "Option",
    "as_mut": "Option",
    "deref": "Box",
    "as_ref": "Box",
    "as_mut": "Box",
    "into_inner": "Box",
    "clone_from": "Box",
    
    # ── C++ vector-specific methods ──────────────────────────────────────
    "push_back": "vector",
    "pop_back": "vector",
    "push_front": "vector",
    "pop_front": "vector",
    "emplace_back": "vector",
    "emplace_front": "vector",
    "resize": "vector",
    "clear": "vector",
    "front": "vector",
    "back": "vector",
    "data": "vector",
    "begin": "vector",
    "end": "vector",
    "rbegin": "vector",
    "rend": "vector",
    "insert": "vector",
    "erase": "vector",
    "assign": "vector",
    "swap": "vector",
    "reserve": "vector",
    "capacity": "vector",
    "shrink_to_fit": "vector",
    "fill": "vector",
    "emplace": "vector",
    "at": "vector",
    
    # ── C++ deque-specific methods ───────────────────────────────────────
    "push_front": "deque",
    "pop_front": "deque",
    "emplace_front": "deque",
    "front": "deque",
    "back": "deque",
    "rbegin": "deque",
    "rend": "deque",
    "resize": "deque",
    
    # ── C++ string-specific methods ──────────────────────────────────────
    "c_str": "string",
    "data": "string",
    "substr": "string",
    "find": "string",
    "rfind": "string",
    "find_first_of": "string",
    "find_last_of": "string",
    "find_first_not_of": "string",
    "find_last_not_of": "string",
    "compare": "string",
    "append": "string",
    "prepend": "string",
    "length": "string",
    "insert": "string",
    "erase": "string",
    "replace": "string",
    "copy": "string",
    "getline": "string",
    
    # ── C++ map/unordered_map methods ────────────────────────────────────
    "lower_bound": "map",
    "upper_bound": "map",
    "equal_range": "map",
    "merge": "map",
    "extract": "map",
    "contains": "map",
    "try_emplace": "map",
    "bucket_count": "unordered_map",
    "bucket": "unordered_map",
    "load_factor": "unordered_map",
    "max_load_factor": "unordered_map",
    "rehash": "unordered_map",
    "bucket_count": "unordered_set",
    "bucket": "unordered_set",
    
    # ── C++ set/unordered_set methods ────────────────────────────────────
    # Note: lower_bound, upper_bound, equal_range defined above for map
    "merge": "set",
    "extract": "set",
    "contains": "set",
    
    # ── Common methods with cross-library ambiguity ──────────────────────
    # Prefer Python for most common cases, override where Rust/C++ dominate
    
    # "get" is ambiguous: Python dict.get, Rust HashMap.get, JS Map.get
    # Python dict is most common in codebases
    "get": "dict",
    "keys": "dict",
    "values": "dict",
    "items": "dict",
    "clear": "dict",
    "pop": "dict",
    "update": "dict",
    
    # "find" is ambiguous: C++ string.find, Rust Iterator.find, JS Array.find
    # In Python context, Array.find is most common
    "find": "Array",
    
    # "replace" is ambiguous: C++ string.replace, Python str.replace
    "replace": "str",
    
    # "count" is ambiguous: C++ map/set count, Python list.count, Rust Iterator.count
    "count": "list",
    
    # "size" is ambiguous: C++ container.size, Python len()
    "size": "list",
    
    # "empty" is ambiguous: C++ container.empty, Python not container
    "empty": "list",
    
    # "begin/end" are C++ specific
    "begin": "vector",
    "end": "vector",
    "rbegin": "vector",
    "rend": "vector",
    
    # "at" is ambiguous: C++ at(), Python []
    "at": "vector",
    
    # "operator[]" is C++ specific
    "operator[]": "vector",
    
    # "contains" is ambiguous: Python 'in', Rust HashSet.contains, C++ set.contains
    "contains": "list",
    
    # "insert" is ambiguous: Python list.insert, C++ vector.insert, Rust Vec::insert
    "insert": "list",
    
    # "remove" is ambiguous: Python list.remove, Rust Vec::remove, C++ container.erase
    "remove": "list",
    
    # "append" is ambiguous: Python list.append, Rust Vec::append, C++ string.append
    "append": "list",
    
    # "push" is ambiguous: Python list.push (rare), Rust Vec::push, JS Array.push
    "push": "Array",
    
    # "sort" is ambiguous: Python list.sort, Rust Vec::sort, JS Array.sort
    "sort": "list",
    
    # "map" is ambiguous: Python map(), Rust Iterator::map, JS Array::map
    "map": "Array",
    
    # "filter" is ambiguous: Python filter(), Rust Iterator::filter, JS Array::filter
    "filter": "Array",
    
    # "len" is ambiguous: Rust len(), C++ size(), Python len()
    "len": "list",
    
    # "new" is ambiguous: Rust Vec::new, JS new Array(), Python list()
    "new": "list",
    
    # "copy" is ambiguous: Python list.copy, Rust clone(), C++ copy()
    "copy": "list",
    
    # "close" is ambiguous: File.close, connection.close
    "close": "File",
    
    # "read" is ambiguous: File.read, socket.read
    "read": "File",
    
    # "write" is ambiguous: File.write, socket.write
    "write": "File",
}


def get_method_disambiguation(method_name: str) -> str | None:
    """Return the preferred stub type for a method name when it's ambiguous
    across languages, or None if no disambiguation is available.
    """
    return _METHOD_DISAMBIGUATION.get(method_name)


def _get_stub_lookup() -> dict[str, list[tuple[str, str]]]:
    """Build a lookup of all stub types that have a given method.
    
    Returns a dict mapping method names to list of (stub_type, language) tuples.
    """
    from graphician.extraction.library_stubs._python_stubs import _PYTHON_STUBS
    from graphician.extraction.library_stubs._rust_stubs import _RUST_STUBS
    from graphician.extraction.library_stubs._javascript_stubs import _JAVASCRIPT_STUBS
    from graphician.extraction.library_stubs._cpp_stubs import _CPP_STUBS
    
    lookup: dict[str, list[tuple[str, str]]] = {}
    
    for stub_type in _RUST_STUBS:
        stub = _RUST_STUBS[stub_type]
        for method in stub.methods:
            lookup.setdefault(method, []).append((stub_type, "rust"))
    
    for stub_type in _PYTHON_STUBS:
        stub = _PYTHON_STUBS[stub_type]
        for method in stub.methods:
            lookup.setdefault(method, []).append((stub_type, "python"))
    
    for stub_type in _JAVASCRIPT_STUBS:
        stub = _JAVASCRIPT_STUBS[stub_type]
        for method in stub.methods:
            lookup.setdefault(method, []).append((stub_type, "javascript"))
    
    for stub_type in _CPP_STUBS:
        stub = _CPP_STUBS[stub_type]
        for method in stub.methods:
            lookup.setdefault(method, []).append((stub_type, "cpp"))
    
    return lookup
