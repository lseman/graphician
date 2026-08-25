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
    "ctx": "Context", "context": "Context", "handle": "Handle",
    "writer": "Writer", "reader": "Reader", "builder": "Builder",
    "factory": "Factory", "handler": "Handler", "middleware": "Middleware",
    "router": "Router", "controller": "Controller", "repository": "Repository",
    "service": "Service", "manager": "Manager", "driver": "Driver",
    "executor": "Executor", "scheduler": "Scheduler", "provider": "Provider",
    "observer": "Observer", "listener": "Listener", "subscriber": "Subscriber",
    "cache": "Cache", "pool": "Pool", "connection": "Connection",
    "session": "Session", "request": "Request", "response": "Response",
    "event": "Event", "task": "Task", "job": "Job", "thread": "Thread",
    "process": "Process", "timer": "Timer", "counter": "Counter",
    "signal": "Signal", "bus": "Bus", "channel": "Channel",
    "queue": "Queue", "stack": "Stack", "heap": "Heap", "tree": "Tree",
    "node": "Node", "edge": "Edge", "matrix": "Matrix",
    "vector": "Vector", "scalar": "Scalar", "tensor": "Tensor",
    "layer": "Layer", "model": "Model", "dataset": "Dataset", "batch": "Batch",
    "epoch": "Epoch", "step": "Step", "iteration": "Iteration", "flags": "Flags",
    "params": "Params", "hyperparams": "HyperParams", "args": "Args", "kwargs": "Kwargs",
    "env": "Env", "environment": "Env", "settings": "Settings",
    "metadata": "Metadata", "arguments": "Arguments",
    "conn": "Connection", "sqlite": "Connection",
    "rx": "Receiver", "sender": "Sender", "tx": "Sender", "client": "Client",

    # ── Python patterns ─────────────────────────────────────────────────

    "items": "list", "values": "dict", "keys": "dict",
    "result": "dict", "data": "dict", "args": "list", "kwargs": "dict",
    "lines": "list", "entries": "list", "records": "list",
    "files": "list", "dirs": "list", "paths": "list",
    "nodes": "list", "edges": "list", "edges_list": "list",
    "config": "dict", "settings": "dict", "options": "dict",
    "cache": "dict", "registry": "dict", "index": "dict",
    "logger": "Logger", "log": "Logger", "cursor": "Cursor", "db": "Database", "server": "Server", "app": "App",
    "fs": "Path", "p": "Path",
    "filepath": "Path", "filename": "Path", "dirpath": "Path", "dirname": "Path",
    "source_path": "Path", "target_path": "Path", "output_path": "Path",
    "input_path": "Path", "data_path": "Path", "cache_path": "Path",
    "log_path": "Path", "config_path": "Path", "home": "Path", "cwd": "Path",
    # DataFrame / pandas
    "df": "DataFrame", "frame": "DataFrame", "tbl": "DataFrame", "table": "DataFrame",
    "records": "DataFrame", "data_frame": "DataFrame",
    "s": "Series", "ser": "Series", "col": "Series", "column": "Series",
    "idx": "Index", "index": "Index", "labels": "Index", "names": "Index",
    "mi": "MultiIndex", "midx": "MultiIndex", "level": "MultiIndex", "db_session": "Session", "sess": "Session",
    "engine": "Engine", "db_engine": "Engine", "txn": "Transaction", "transaction": "Transaction",
    "table": "Table", "tbl": "Table", "meta": "MetaData", "metadata": "MetaData",
    "col": "Column", "field": "Column", "schema": "Table",
    "query": "Query", "q": "Query", "stmt": "Statement", "statement": "Statement",
    # Pydantic
    "model": "BaseModel", "m": "BaseModel", "obj": "BaseModel", "record": "BaseModel",
    "dto": "BaseModel", "schema": "BaseModel", "spec": "BaseModel",
    # FastAPI
    "app": "FastAPI", "api": "FastAPI", "router": "APIRouter",
    "req": "Request", "r": "Request", "res": "Response", "resp": "Response",
    # asyncio
    "loop": "EventLoop", "event_loop": "EventLoop", "coro": "Task",
    "fut": "Future", "future": "Future", "asyncio_task": "Task", "asyncio_future": "Future", "http_resp": "Response", "http_session": "Session", "http_req": "Request", "payload": "Request",
    # numpy
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

    # ── JavaScript patterns ─────────────────────────────────────────────

    "arr": "Array", "list": "Array", "items": "Array", "elements": "Array",
    "map": "Map", "dict": "Map", "obj_map": "Map",
    "set": "Set", "unique": "Set", "distinct": "Set",
    "promise": "Promise", "prom": "Promise", "async_result": "Promise",
    "str": "String", "msg": "String", "message": "String",
    "err": "Error", "error": "Error", "e": "Error", "exception": "Error",
    "url": "URL", "search": "URLSearchParams", "params": "URLSearchParams",
    "query": "URLSearchParams", "qs": "URLSearchParams",
    "regex": "RegExp", "re": "RegExp", "pattern": "RegExp",
    "date": "Date", "time": "Date", "timestamp": "Date",
    "buffer": "Uint8Array", "buf": "Uint8Array", "bytes": "Uint8Array",
    "blob": "Blob", "file": "File", "form": "FormData", "fd": "FormData", "express": "Application",
    "fs": "fs", "path": "path", "os": "os", "http": "http",
    "https": "https", "net": "net", "url": "url", "querystring": "querystring",
    # ── lodash patterns ─────────────────────────────────────────────
    "_": "_", "lodash": "_", "utils": "_", "helpers": "_",
    # ── rxjs patterns ───────────────────────────────────────────────
    "obs": "Observable", "observable": "Observable", "stream": "Observable",
    "src": "Observable", "subject": "Subject", "behavior": "BehaviorSubject",
    "replay": "ReplaySubject", "sub": "Subscription", "subscrip": "Subscription",
    # ── dayjs patterns ──────────────────────────────────────────────
    "d": "dayjs", "date": "dayjs", "dt": "dayjs", "time": "dayjs",
    "timestamp": "dayjs", "now": "dayjs", "today": "dayjs",
    # ── axios patterns ──────────────────────────────────────────────
    "http": "axios", "client": "axios", "api": "axios", "fetch": "axios",
    # ── commander patterns ──────────────────────────────────────────
    "cmd": "Command", "command": "Command", "program": "Command", "cli": "Command",
    # ── zod patterns ────────────────────────────────────────────────
    "schema": "ZodType", "zod": "ZodType", "spec": "ZodType", "pino": "Logger", "winston": "Logger",
    # ── dotenv patterns ─────────────────────────────────────────────
    "env": "dotenv", "config": "dotenv", "env_config": "dotenv",
    # ── uuid patterns ───────────────────────────────────────────────
    "uid": "uuid", "uuid": "uuid", "guid": "uuid", "id": "uuid", "next": "NextFunction", "app": "Express", "list": "Vec", "dq": "VecDeque", "sb": "String", "file": "Path", "maybe_val": "Option", "r": "Result", "chain": "Iterator", "cache_store": "Store", "cx": "Context", "db_conn": "Connection", "channel": "Sender", "rt": "Handle",
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

    # ── DataFrame / pandas methods ─────────────────────────────────────
    "head": "DataFrame", "tail": "DataFrame", "iloc": "DataFrame",
    "loc": "DataFrame", "copy": "DataFrame", "assign": "DataFrame",
    "drop": "DataFrame", "dropna": "DataFrame", "fillna": "DataFrame",
    "replace": "DataFrame", "rename": "DataFrame", "reset_index": "DataFrame",
    "set_index": "DataFrame", "sort_values": "DataFrame", "sort_index": "DataFrame",
    "merge": "DataFrame", "join": "DataFrame", "groupby": "DataFrame",
    "pivot": "DataFrame", "pivot_table": "DataFrame", "transpose": "DataFrame",
    "melt": "DataFrame", "stack": "DataFrame", "unstack": "DataFrame",
    "explode": "DataFrame", "sample": "DataFrame", "describe": "DataFrame",
    "to_csv": "DataFrame", "to_excel": "DataFrame", "to_json": "DataFrame",
    "to_html": "DataFrame", "to_pickle": "DataFrame", "to_parquet": "DataFrame",
    "isnull": "DataFrame", "notnull": "DataFrame", "empty": "DataFrame",
    "shape": "DataFrame", "size": "DataFrame", "dtypes": "DataFrame",
    "columns": "DataFrame", "index": "DataFrame", "values": "DataFrame",
    "T": "DataFrame", "nbytes": "DataFrame", "memory_usage": "DataFrame",
    "select_dtypes": "DataFrame", "eval": "DataFrame", "query": "DataFrame",
    "pipe": "DataFrame", "apply": "DataFrame", "applymap": "DataFrame",
    "map": "DataFrame", "agg": "DataFrame", "aggregate": "DataFrame",
    "transform": "DataFrame", "corr": "DataFrame", "cov": "DataFrame",
    "dot": "DataFrame", "prod": "DataFrame", "sum": "DataFrame",
    "mean": "DataFrame", "median": "DataFrame", "min": "DataFrame",
    "max": "DataFrame", "idxmin": "DataFrame", "idxmax": "DataFrame",
    "argmin": "DataFrame", "argmax": "DataFrame", "pct_change": "DataFrame",
    "cumsum": "DataFrame", "cumprod": "DataFrame", "cummin": "DataFrame",
    "cummax": "DataFrame", "diff": "DataFrame", "rank": "DataFrame",
    "rolling": "DataFrame", "expanding": "DataFrame", "ewm": "DataFrame",
    "shift": "DataFrame", "between": "DataFrame", "clip": "DataFrame",
    "drop_duplicates": "DataFrame", "duplicated": "DataFrame",
    "reindex": "DataFrame", "align": "DataFrame", "compare": "DataFrame",
    "filter": "DataFrame", "first": "DataFrame", "last": "DataFrame",
    "truncate": "DataFrame", "swaplevel": "DataFrame", "droplevel": "DataFrame",
    "is_monotonic_increasing": "DataFrame", "is_monotonic_decreasing": "DataFrame",
    "is_unique": "DataFrame", "hasnans": "DataFrame", "isna": "DataFrame",
    "__len__": "DataFrame", "__getitem__": "DataFrame", "__setitem__": "DataFrame",
    "__iter__": "DataFrame", "__contains__": "DataFrame",
    # Series methods
    "head": "Series", "tail": "Series", "copy": "Series", "drop": "Series",
    "dropna": "Series", "fillna": "Series", "replace": "Series", "rename": "Series",
    "sort_values": "Series", "sort_index": "Series", "astype": "Series",
    "convert_dtypes": "Series", "unique": "Series", "value_counts": "Series",
    "isnull": "Series", "notnull": "Series", "empty": "Series",
    "shape": "Series", "size": "Series", "dtype": "Series", "dtypes": "Series",
    "index": "Series", "values": "Series", "ndim": "Series", "name": "Series",
    "nbytes": "Series", "memory_usage": "Series", "to_frame": "Series",
    "to_dict": "Series", "to_list": "Series", "to_numpy": "Series",
    "to_csv": "Series", "to_json": "Series", "to_excel": "Series",
    "to_pickle": "Series", "isna": "Series",
    "notna": "Series", "abs": "Series",
    "clip": "Series", "between": "Series", "eq": "Series", "ne": "Series",
    "lt": "Series", "le": "Series", "gt": "Series", "ge": "Series",
    "add": "Series", "radd": "Series", "sub": "Series", "rsub": "Series",
    "mul": "Series", "rmul": "Series", "truediv": "Series", "rtruediv": "Series",
    "floordiv": "Series", "rfloordiv": "Series", "mod": "Series", "rmod": "Series",
    "pow": "Series", "rpow": "Series", "sum": "Series", "mean": "Series",
    "median": "Series", "min": "Series", "max": "Series", "idxmin": "Series",
    "idxmax": "Series", "argmin": "Series", "argmax": "Series",
    "pct_change": "Series", "cumsum": "Series", "cumprod": "Series",
    "cummin": "Series", "cummax": "Series", "diff": "Series", "rank": "Series",
    "rolling": "Series", "expanding": "Series", "ewm": "Series", "shift": "Series",
    "corr": "Series", "cov": "Series", "count": "Series", "nunique": "Series",
    "std": "Series", "var": "Series", "sample": "Series", "quantile": "Series",
    "skew": "Series", "kurt": "Series", "drop_duplicates": "Series",
    "duplicated": "Series", "reindex": "Series", "align": "Series",
    "truncate": "Series", "__len__": "Series", "__getitem__": "Series",
    "__setitem__": "Series", "__iter__": "Series", "__contains__": "Series",
    # Index methods
    "name": "Index", "names": "Index", "dtype": "Index", "inferred_type": "Index",
    "is_unique": "Index", "is_monotonic_increasing": "Index",
    "is_monotonic_decreasing": "Index", "is_lexsorted": "Index",
    "nlevels": "Index", "ndim": "Index", "size": "Index", "length": "Index",
    "empty": "Index", "shape": "Index", "values": "Index", "data": "Index",
    "min": "Index", "max": "Index", "argmin": "Index", "argmax": "Index",
    "unique": "Index", "nunique": "Index", "value_counts": "Index",
    "drop_duplicates": "Index", "duplicated": "Index", "reindex": "Index",
    "equals": "Index", "union": "Index", "intersection": "Index",
    "difference": "Index", "symmetric_difference": "Index",
    "join": "Index", "map": "Index", "rename": "Index", "rename_axis": "Index",
    "set_name": "Index", "copy": "Index", "astype": "Index", "view": "Index",
    "to_frame": "Index", "to_series": "Index", "to_list": "Index",
    "to_numpy": "Index", "sort_values": "Index", "__len__": "Index",
    "__getitem__": "Index", "__iter__": "Index", "__contains__": "Index",

    # ── HTTP / REST methods ───────────────────────────────────────────
    "status_code": "Response", "ok": "Response", "reason": "Response",
    "url": "Response", "text": "Response", "content": "Response",
    "json": "Response", "encoding": "Response", "headers": "Response",
    "cookies": "Response", "history": "Response", "elapsed": "Response",
    "request": "Response", "links": "Response", "raise_for_status": "Response",
    "iter_lines": "Response", "iter_content": "Response", "raw": "Response",
    "apparent_encoding": "Response", "__bool__": "Response",
    "get": "Session", "post": "Session", "put": "Session", "patch": "Session",
    "delete": "Session", "head": "Session", "options": "Session",
    "request": "Session", "send": "Session", "merge_environment_settings": "Session",
    "build_request": "Session", "prepare_request": "Session",
    "resolve_redirects": "Session", "adapt_adapter": "Session",
    "get_adapter": "Session", "trust_env": "Session", "hooks": "Session",
    "auth": "Session", "cookies": "Session", "headers": "Session",
    "params": "Session", "verify": "Session", "cert": "Session",
    "proxies": "Session", "stream": "Session", "timeout": "Session",
    "allow_redirects": "Session", "max_redirects": "Session",
    "mount": "Session", "close": "Session", "__enter__": "Session",
    "__exit__": "Session",

    # ── Graph methods ─────────────────────────────────────────────────
    "add_node": "Graph", "add_nodes_from": "Graph", "remove_node": "Graph",
    "remove_nodes_from": "Graph", "add_edge": "Graph", "add_edges_from": "Graph",
    "remove_edge": "Graph", "remove_edges_from": "Graph", "edges": "Graph",
    "nodes": "Graph", "has_node": "Graph", "has_edge": "Graph",
    "neighbors": "Graph", "predecessors": "Graph", "successors": "Graph",
    "adj": "Graph", "adjacency": "Graph", "degree": "Graph",
    "in_degree": "Graph", "out_degree": "Graph", "subgraph": "Graph",
    "copy": "Graph", "to_directed": "Graph", "to_undirected": "Graph",
    "clear": "Graph", "clear_edges": "Graph", "update": "Graph",
    "compose": "Graph", "union": "Graph", "intersection": "Graph",
    "difference": "Graph", "reverse": "Graph", "density": "Graph",
    "is_directed": "Graph", "is_connected": "Graph", "order": "Graph",
    "size": "Graph", "degree_view": "Graph",
    "add_weighted_edges_from": "Graph", "add_path": "Graph", "add_cycle": "Graph",
    "add_star": "Graph", "get_edge_data": "Graph", "pred": "Graph",
    "succ": "Graph", "is_empty": "Graph", "is_multigraph": "Graph",

    # ── Path methods ──────────────────────────────────────────────────
    "exists": "Path", "is_file": "Path", "is_dir": "Path", "is_symlink": "Path",
    "read_text": "Path", "read_bytes": "Path", "write_text": "Path",
    "write_bytes": "Path", "iterdir": "Path", "parent": "Path", "name": "Path",
    "stem": "Path", "suffix": "Path", "parts": "Path", "match": "Path",
    "resolve": "Path", "absolute": "Path", "relative_to": "Path",
    "with_suffix": "Path", "with_name": "Path", "joinpath": "Path",
    "glob": "Path", "rglob": "Path", "stat": "Path", "mkdir": "Path",
    "unlink": "Path", "rename": "Path", "replace": "Path", "touch": "Path",
    "chmod": "Path", "samefile": "Path", "as_posix": "Path", "as_uri": "Path",
    "readlink": "Path", "cwd": "Path", "expanduser": "Path", "group": "Path",
    "hardlink_to": "Path", "home": "Path", "is_absolute": "Path", "lchmod": "Path",
    "link_to": "Path", "lstat": "Path", "owner": "Path", "parents": "Path",
    "rmdir": "Path", "symlink_to": "Path", "suffixes": "Path",
    "is_reserved": "Path", "walk": "Path", "is_socket": "Path",
    "is_block_device": "Path", "is_char_device": "Path", "is_fifo": "Path",
    "__truediv__": "Path", "__fspath__": "Path", "__repr__": "Path",
    "__str__": "Path", "__lt__": "Path", "__le__": "Path",
    "__gt__": "Path", "__ge__": "Path", "__hash__": "Path", "__eq__": "Path",
}


def get_method_disambiguation(method_name: str) -> str | None:
    """Return the preferred stub type for a method name when it's ambiguous
    across languages, or None if no disambiguation is available.
    """
    return _METHOD_DISAMBIGUATION.get(method_name)
