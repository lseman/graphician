"""Library stub resolver — Tier 7 of the call-placeholder resolution chain.

After the 6-tier heuristic resolver exhausts all local-disambiguation
strategies, this module resolves remaining ``call::name`` placeholders
against a pre-built database of library stubs keyed by dependency name
or always-available globals (Rust stdlib, JS/TS, C++ STL).

Creates stub nodes for known library types (Vec, HashMap, String, etc.)
and connects unresolved ``call::name`` edges to them when a method match
is found.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.edge import Confidence, Edge, EdgeKind
from ..core.graph import Graph
from ..core.id import NodeId
from ..core.node import NodeKind
from .call_resolution import should_suppress_call_placeholder


# ── Rust stdlib + common crate stubs ──────────────────────────────────

_RUST_STUBS: dict[str, list[str]] = {
    "Vec": [
        "push", "pop", "len", "is_empty", "insert", "remove", "get", "get_mut",
        "clear", "reserve", "resize", "truncate", "retain", "sort", "sort_by",
        "sort_by_key", "contains", "swap", "split_off", "drain", "extend",
        "from", "into_iter", "iter", "iter_mut", "as_slice", "as_mut_slice",
        "capacity", "with_capacity", "new", "dedup", "first", "first_mut",
        "last", "last_mut", "splice", "append", "to_vec", "as_ptr", "as_mut_ptr",
        "try_push", "shrink_to", "shrink_to_fit", "dedup_by", "dedup_by_key",
        "resize_with", "into_boxed_slice",
    ],
    "HashMap": [
        "get", "get_mut", "insert", "remove", "len", "is_empty", "contains_key",
        "clear", "entry", "remove_entry", "keys", "values", "values_mut", "iter",
        "iter_mut", "extend", "reserve", "shrink_to_fit", "get_or_insert",
        "get_or_insert_with", "or_insert", "or_insert_with", "default", "from",
        "from_iter", "into_keys", "into_values", "retain", "drain",
        "swap_remove", "capacity",
    ],
    "BTreeMap": [
        "get", "get_mut", "insert", "remove", "entry", "contains_key", "keys",
        "values", "values_mut", "iter", "iter_mut", "extend", "new", "from",
        "from_iter", "len", "is_empty", "clear", "first_entry", "last_entry",
        "floor_entry", "ceiling_entry", "range", "get_key_value", "retain", "drain",
    ],
    "HashSet": [
        "insert", "remove", "contains", "get", "len", "is_empty", "iter", "clear",
        "extend", "new", "from", "from_iter", "reserve", "shrink_to_fit",
        "difference", "symmetric_difference", "intersection", "union", "swap_remove",
        "extract_if", "drain", "retain",
    ],
    "VecDeque": [
        "push_back", "push_front", "pop_back", "pop_front", "get", "get_mut",
        "len", "is_empty", "front", "back", "front_mut", "back_mut", "clear",
        "iter", "iter_mut", "new", "reserve", "with_capacity", "truncate",
        "rotate_left", "rotate_right", "contains", "insert", "remove", "drain",
        "append", "extract_if", "retain",
    ],
    "String": [
        "new", "from", "to_string", "to_owned", "push", "push_str", "pop", "clear",
        "len", "is_empty", "capacity", "reserve", "resize", "truncate", "insert",
        "remove", "replace_range", "split_off", "drain", "as_str", "as_mut_str",
        "as_bytes", "as_mut_vec", "retain", "shrink_to_fit", "shrink_to",
        "extend", "from_utf8", "from_utf8_lossy", "from_utf16", "from_utf16_lossy",
        "from_str", "into_boxed_str", "into_bytes", "to_lowercase", "to_uppercase",
        "trim", "trim_start", "trim_end", "trim_matches", "strip_prefix",
        "strip_suffix", "split", "splitn", "rsplit", "lines", "chars",
        "bytes", "is_ascii", "to_ascii_lowercase", "to_ascii_uppercase",
    ],
    "Option": [
        "is_some", "is_none", "Some", "None", "unwrap", "expect", "map",
        "and_then", "or_else", "ok_or", "ok_or_else", "unwrap_or",
        "unwrap_or_default", "unwrap_or_else", "as_ref", "as_mut", "as_deref",
        "filter", "transpose", "and", "or", "iter", "as_ref_and_then",
    ],
    "Result": [
        "ok", "err", "unwrap", "expect", "map", "map_err", "and_then", "or_else",
        "unwrap_or", "unwrap_or_else", "unwrap_or_default", "as_ref", "as_mut",
        "is_ok", "is_err", "transpose", "ok_or", "ok_or_else",
    ],
    "Iterator": [
        "next", "nth", "last", "count", "size_hint", "fold", "for_each", "map",
        "filter", "filter_map", "flat_map", "collect", "cloned", "copied",
        "enumerate", "zip", "chain", "fuse", "by_ref", "scan", "flatten", "unzip",
        "try_fold", "try_for_each", "peekable", "skip", "take", "skip_while",
        "take_while", "step_by", "cycle", "repeat", "repeat_with", "intersperse",
        "rev", "sum", "product", "max", "min", "max_by", "min_by", "max_by_key",
        "min_by_key", "any", "all", "find", "position", "rposition",
        "partition", "exactly_one",
    ],
    "Slice": [
        "iter", "iter_mut", "len", "is_empty", "get", "get_mut", "split",
        "split_mut", "splitn", "rsplit", "rsplitn", "chunks", "chunks_mut",
        "windows", "first", "first_mut", "last", "last_mut", "swap", "contains",
        "starts_with", "ends_with", "sort", "sort_by", "sort_by_key",
        "sort_unstable", "sort_unstable_by", "sort_unstable_by_key",
        "partition", "binary_search", "binary_search_by", "binary_search_by_key",
        "to_vec", "to_owned", "as_ptr", "as_mut_ptr", "get_unchecked",
    ],
    "Path": [
        "join", "parent", "file_name", "file_stem", "extension", "strip_prefix",
        "starts_with", "ends_with", "to_str", "to_path_buf", "components", "iter",
        "display", "as_path", "as_os_str", "push", "pop", "into_os_string",
    ],
    "PathBuf": [
        "new", "push", "pop", "as_path", "to_path_buf", "to_str", "as_os_str",
        "into_os_string", "file_name", "file_stem", "extension", "parent",
        "join", "is_absolute", "is_relative", "exists",
    ],
    "Duration": [
        "as_secs", "as_nanos", "checked_add", "checked_sub", "checked_mul",
        "saturating_add", "saturating_sub", "saturating_mul", "from_secs",
        "from_millis", "from_nanos", "from_micros", "is_zero", "new",
    ],
    "Box": [
        "new", "into_raw", "from_raw", "into", "as_ref", "as_mut", "deref",
        "into_boxed_slice", "unwrap", "leak", "downcast", "downcast_mut",
    ],
    "Arc": [
        "new", "clone", "try_unwrap", "into_raw", "from_raw", "as_ptr",
        "strong_count", "weak_count", "downgrade", "upgrade", "get_mut",
    ],
    "Mutex": [
        "new", "lock", "try_lock", "get_mut", "into_inner", "into_lock_guard",
    ],
    "RwLock": [
        "new", "read", "try_read", "write", "try_write", "get_mut",
        "into_inner", "into_read_guard", "into_write_guard",
    ],
    "Cell": ["new", "get", "set", "replace", "take", "into_inner"],
    "RefCell": [
        "new", "borrow", "borrow_mut", "try_borrow", "try_borrow_mut",
        "into_inner", "borrow_check",
    ],
    "AtomicUsize": [
        "new", "load", "store", "fetch_add", "fetch_sub", "fetch_and",
        "fetch_or", "fetch_xor", "swap", "compare_exchange", "compare_exchange_weak",
    ],
    "AtomicBool": [
        "new", "load", "store", "fetch_and", "fetch_or", "fetch_xor",
        "swap", "compare_exchange", "compare_exchange_weak",
    ],
    "FromStr": ["from_str"],
    "io": [
        "read", "read_to_end", "read_to_string", "read_exact", "write",
        "write_all", "flush", "write_fmt", "BufRead", "Read", "Write",
    ],
    "env": ["args", "current_dir", "set_current_dir", "temp_dir", "var", "var_os",
            "vars", "vars_os", "set_var", "remove_var", "current_exe", "var"],
    "fs": [
        "read", "read_to_string", "write", "create", "create_dir", "create_dir_all",
        "remove_dir", "remove_dir_all", "remove_file", "rename", "copy", "metadata",
        "symlink_metadata", "read_dir", "set_permissions", "canonicalize", "exists",
        "write_bytes", "read_bytes",
    ],
    "time": ["now", "duration_since", "elapsed"],
    "serde_json": [
        "Value", "to_string", "to_string_pretty", "to_value", "from_str",
        "from_slice", "from_reader", "to_writer", "from_value", "Map", "Number",
        "from_reader", "to_writer",
    ],
    "tokio": ["spawn", "task"],
    "tracing": [
        "info", "debug", "warn", "error", "trace", "info_span", "debug_span",
        "warn_span", "error_span", "trace_span", "instrument",
    ],
    "anyhow": ["Result", "Context", "bail", "ensure"],
    "regex": [
        "Regex", "new", "find", "find_iter", "captures", "captures_iter",
        "is_match", "replace", "replace_all", "split", "splitn",
        "RegexSet", "no_lookahead",
    ],
    "walkdir": ["WalkDir", "new", "into_iter"],
    "rayon": [
        "join", "scope", "iter", "scope", "into_iter", "scope",
        "IntoParallelIterator", "IntoParallelRefIterator", "ParallelIterator",
        "ParallelDrainFull",
    ],
    "std::error": ["Error"],
    "std::fmt": ["Display", "Debug", "Formatter", "write", "write_fmt"],
}

# ── Python stdlib + builtins stubs ─────────────────────────────────────

_PYTHON_STUBS: dict[str, list[str]] = {
    "list": [
        "append", "extend", "insert", "remove", "pop", "clear", "index",
        "count", "sort", "reverse", "copy", "len", "iter", "__getitem__",
        "__setitem__", "__delitem__", "__contains__", "__iter__", "__len__",
    ],
    "dict": [
        "get", "setdefault", "pop", "popitem", "clear", "update", "keys",
        "values", "items", "fromkeys", "copy", "len", "iter", "__getitem__",
        "__setitem__", "__delitem__", "__contains__", "__iter__", "__len__",
    ],
    "str": [
        "upper", "lower", "strip", "lstrip", "rstrip", "replace", "split",
        "rsplit", "join", "startswith", "endswith", "find", "rfind", "index",
        "rindex", "count", "isdigit", "isalpha", "isalnum", "isspace",
        "isupper", "islower", "title", "capitalize", "swapcase", "zfill",
        "center", "ljust", "rjust", "partition", "rpartition", "expandtabs",
        "translate", "encode", "decode", "format", "format_map", "isdecimal",
        "isascii", "isprintable", "removesuffix", "removeprefix",
    ],
    "int": ["bit_length", "to_bytes", "from_bytes", "denominator", "numerator", "conjugate",
            "real", "imag", "__abs__", "__floor__", "__ceil__", "__round__"],
    "float": [
        "is_integer", "as_integer_ratio", "hex", "fromhex", "is_nan", "is_inf",
        "is_finite", "real", "imag",
    ],
    "set": [
        "add", "remove", "discard", "pop", "clear", "update", "intersection_update",
        "difference_update", "symmetric_difference_update", "union", "intersection",
        "difference", "symmetric_difference", "issubset", "issuperset", "isdisjoint",
        "copy", "len", "iter", "__contains__", "__iter__", "__len__",
    ],
    "tuple": [
        "count", "index", "len", "iter", "__getitem__", "__contains__",
        "__iter__", "__len__",
    ],
    "bytes": [
        "decode", "encode", "upper", "lower", "strip", "split", "find",
        "index", "count", "hex", "fromhex", "join", "replace", "translate",
        "maketrans", "len", "iter", "__contains__", "__iter__", "__len__",
    ],
    "Optional": ["is_some", "is_none", "unwrap", "expect", "map", "and_then", "unwrap_or"],
    "Union": ["isinstance"],
    "Callable": ["__call__"],
    "Iterator": [
        "next", "iter", "__next__", "__iter__",
    ],
    "Generator": [
        "send", "throw", "close", "send", "return",
    ],
    "Iterable": ["__iter__"],
    "Mapping": [
        "keys", "values", "items", "get", "pop", "popitem", "clear", "update",
        "copy", "len", "iter", "__getitem__", "__setitem__", "__delitem__",
        "__contains__", "__iter__", "__len__", "__reversed__",
    ],
    "Sequence": [
        "append", "extend", "insert", "remove", "pop", "clear", "index",
        "count", "sort", "reverse", "copy", "len", "iter",
        "__getitem__", "__setitem__", "__delitem__", "__contains__",
        "__iter__", "__len__", "__reversed__",
    ],
    "Path": [
        "exists", "is_file", "is_dir", "is_symlink", "read_text", "read_bytes",
        "write_text", "write_bytes", "iterdir", "parent", "name", "stem",
        "suffix", "parts", "match", "resolve", "absolute", "relative_to",
        "with_suffix", "with_name", "joinpath", "glob", "rglob", "stat",
        "mkdir", "unlink", "rename", "replace", "touch", "chmod", "samefile",
    ],
    "os": [
        "path", "environ", "getenv", "setenv", "remove", "rename", "renames",
        "mkdir", "makedirs", "rmdir", "removedirs", "listdir", "scandir",
        "walk", "chdir", "getcwd", "getpid", "getppid", "getuid", "getgid",
        "execvp", "fork", "waitpid", "kill", "access", "stat", "lstat",
        "link", "symlink", "readlink", "umask", "truncate",
    ],
    "sys": [
        "argv", "exit", "version", "platform", "stdin", "stdout", "stderr",
        "path", "modules", "getsizeof", "setrecursionlimit", "getrecursionlimit",
        "getdefaultencoding", "settrace", "gettrace", "getprofile",
        "exc_info", "excepthook", "unraisablehook",
    ],
    "re": [
        "search", "match", "fullmatch", "findall", "finditer", "sub", "subn",
        "split", "compile", "escape", "pattern", "flags", "group", "groups",
        "groupdict", "start", "end", "span",
    ],
    "json": [
        "loads", "dumps", "load", "dump", "JSONDecoder", "JSONEncoder",
        "JSONDecodeError",
    ],
    "logging": [
        "debug", "info", "warning", "warn", "error", "critical", "log",
        "getLogger", "basicConfig", "StreamHandler", "FileHandler",
        "Formatter", "Logger",
    ],
    "dataclasses": [
        "dataclass", "field", "make_dataclass", "asdict", "astuple",
        "fields", "is_dataclass", "replace",
    ],
    "typing": [
        "Optional", "Union", "List", "Dict", "Set", "Tuple", "Sequence",
        "Mapping", "Iterator", "Generator", "Callable", "TypeVar", "Generic",
        "Any", "ClassVar", "Final", "Protocol", "runtime_checkable",
        "cast", "get_type_hints", "get_args", "get_origin",
        "Annotated", "Literal", "TypedDict",
    ],
    "collections": [
        "defaultdict", "Counter", "OrderedDict", "deque", "namedtuple",
        "ChainMap", "UserDict", "UserList", "UserString",
    ],
    "functools": [
        "lru_cache", "cache", "partial", "wraps", "partialmethod",
        "reduce", "total_ordering",
    ],
    "itertools": [
        "chain", "combinations", "combinations_with_replacement", "compress",
        "count", "cycle", "dropwhile", "filterfalse", "groupby", "islice",
        "permutations", "product", "repeat", "starmap", "takewhile",
        "tee", "zip_longest",
    ],
    "os.path": [
        "join", "exists", "isfile", "isdir", "islink", "isabs", "abspath",
        "relpath", "dirname", "basename", "splitext", "split", "commonpath",
        "realpath", "samefile", "getsize", "getmtime", "getctime", "lexists",
        "walk", "pardir", "sep", "altsep",
    ],
    "subprocess": [
        "run", "call", "check_output", "check_call", "Popen", "PIPE",
        "STDOUT", "CompletedProcess",
    ],
    "datetime": [
        "datetime", "date", "time", "timedelta", "timezone", "UTC",
        "now", "today", "fromtimestamp", "utcnow", "strptime", "strftime",
    ],
    "hashlib": [
        "md5", "sha1", "sha256", "sha512", "new", "file_digest", "pbkdf2_hmac",
    ],
    "threading": [
        "Thread", "Lock", "RLock", "Semaphore", "Event", "Condition",
        "Barrier", "Timer", "BoundedSemaphore", "active_count",
        "enumerate", "current_thread", "main_thread",
    ],
    "contextlib": [
        "contextmanager", "suppress", "redirect_stdout", "redirect_stderr",
        "closing", "ExitStack", "AbstractContextManager", "AbstractAsyncContextManager",
    ],
    "unittest": [
        "TestCase", "main", "mock", "SkipTest", "skip", "skipIf", "skipUnless",
    ],
    "argparse": [
        "ArgumentParser", "add_argument", "parse_args", "parse_known_args",
        "Namespace",
    ],
}

# ── JavaScript / TypeScript stubs ──────────────────────────────────────

_JAVASCRIPT_STUBS: dict[str, list[str]] = {
    "Array": [
        "push", "pop", "shift", "unshift", "splice", "slice", "concat",
        "join", "toString", "indexOf", "lastIndexOf", "includes",
        "forEach", "map", "filter", "reduce", "reduceRight", "some",
        "every", "find", "findIndex", "flat", "flatMap", "fill",
        "sort", "reverse", "copyWithin", "entries", "keys", "values",
        "at", "findLast", "findLastIndex", "toReversed", "toSorted",
        "toSpliced", "with",
    ],
    "Object": [
        "keys", "values", "entries", "assign", "create", "defineProperty",
        "defineProperties", "getOwnPropertyDescriptor", "getOwnPropertyDescriptors",
        "getOwnPropertyNames", "getOwnPropertySymbols", "getPrototypeOf",
        "setPrototypeOf", "is", "isExtensible", "preventExtensions",
        "seal", "isSealed", "freeze", "isFrozen", "hasOwn", "has",
    ],
    "Map": [
        "get", "set", "has", "delete", "clear", "size", "entries",
        "keys", "values", "forEach",
    ],
    "Set": [
        "add", "delete", "has", "clear", "size", "entries", "keys",
        "values", "forEach", "intersection", "union", "difference",
        "symmetricDifference", "isSubsetOf", "isSupersetOf", "isDisjointFrom",
    ],
    "Promise": [
        "then", "catch", "finally", "all", "allSettled", "any", "race",
        "resolve", "reject",
    ],
    "Date": [
        "now", "parse", "UTC", "getFullYear", "getMonth", "getDate",
        "getDay", "getHours", "getMinutes", "getSeconds", "getTime",
        "setFullYear", "setMonth", "setDate", "setHours", "setMinutes",
        "setSeconds", "setTime", "toDateString", "toTimeString",
        "toISOString", "toJSON", "toString", "valueOf",
    ],
    "Error": [
        "name", "message", "stack", "cause",
    ],
    "console": [
        "log", "info", "warn", "error", "debug", "trace", "table",
        "count", "countReset", "time", "timeEnd", "timeLog",
        "assert", "dir", "dirxml", "group", "groupEnd", "clear",
    ],
    "process": [
        "exit", "version", "versions", "platform", "arch", "pid",
        "cwd", "chdir", "env", "argv", "stdin", "stdout", "stderr",
        "exitCode", "exitCode", "on", "emit", "hrtime",
        "memoryUsage", "cpuUsage", "umask",
    ],
    "String": [
        "charAt", "charCodeAt", "concat", "includes", "endsWith", "indexOf",
        "lastIndexOf", "localeCompare", "match", "matchAll", "normalize",
        "padEnd", "padStart", "repeat", "replace", "replaceAll", "search",
        "slice", "split", "startsWith", "substring", "toLocaleLowerCase",
        "toLocaleUpperCase", "toLowerCase", "toString", "toUpperCase",
        "trim", "trimStart", "trimEnd", "trimLeft", "trimRight",
        "valueOf", "at", "codePointAt", "fromCodePoint",
    ],
    "Number": [
        "isFinite", "isInteger", "isNaN", "isSafeInteger", "parseFloat",
        "parseInt", "toExponential", "toFixed", "toPrecision", "toString",
        "valueOf",
    ],
    "Math": [
        "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh",
        "atan2", "cbrt", "ceil", "clz32", "cos", "cosh", "exp",
        "expm1", "floor", "fround", "hypot", "imul", "log", "log1p",
        "log10", "log2", "max", "min", "pow", "random", "round",
        "sign", "sin", "sinh", "sqrt", "tan", "tanh", "trunc",
    ],
    "RegExp": [
        "test", "exec", "source", "global", "ignoreCase", "multiline",
        "sticky", "unicode", "dotAll", "flags", "lastIndex",
    ],
    "JSON": ["parse", "stringify"],
    "ArrayBuffer": ["byteLength", "slice", "resize", "detach", "isView",
                    "byteLength", "maxByteLength", "transfer"],
    "Uint8Array": [
        "set", "subarray", "slice", "fill", "find", "findIndex", "forEach",
        "join", "lastIndexOf", "reverse", "sort", "toString",
    ],
    "Uint8ClampedArray": ["set", "subarray", "slice", "fill"],
    "Int8Array": ["set", "subarray", "slice", "fill"],
    "Int16Array": ["set", "subarray", "slice", "fill"],
    "Int32Array": ["set", "subarray", "slice", "fill"],
    "Uint16Array": ["set", "subarray", "slice", "fill"],
    "Uint32Array": ["set", "subarray", "slice", "fill"],
    "Float32Array": ["set", "subarray", "slice", "fill"],
    "Float64Array": ["set", "subarray", "slice", "fill"],
    "URL": [
        "parse", "resolve", "toString", "toJSON", "href", "protocol",
        "username", "password", "hostname", "host", "port", "pathname",
        "search", "hash", "searchParams", "origin",
    ],
    "URLSearchParams": [
        "append", "delete", "get", "getAll", "has", "set", "sort",
        "toString", "entries", "keys", "values", "forEach",
    ],
    "fetch": ["get", "post", "put", "delete", "patch"],
    "setTimeout": [],
    "setInterval": [],
    "require": ["resolve", "cache", "extensions"],
}

# ── C++ STL stubs ─────────────────────────────────────────────────────

_CPP_STUBS: dict[str, list[str]] = {
    "vector": [
        "push_back", "pop_back", "size", "empty", "resize", "clear", "front",
        "back", "data", "begin", "end", "rbegin", "rend", "insert", "erase",
        "assign", "swap", "reserve", "capacity", "shrink_to_fit",
        "emplace", "emplace_back", "at", "operator[]",
    ],
    "string": [
        "push_back", "pop_back", "size", "length", "empty", "clear", "front",
        "back", "c_str", "data", "begin", "end", "rbegin", "rend",
        "insert", "erase", "replace", "append", "prepend", "substr",
        "find", "rfind", "find_first_of", "find_last_of", "find_first_not_of",
        "find_last_not_of", "compare", "operator[]", "at", "swap",
        "c_str", "copy", "getline",
    ],
    "map": [
        "find", "insert", "erase", "clear", "size", "empty", "begin", "end",
        "rbegin", "rend", "count", "lower_bound", "upper_bound", "equal_range",
        "emplace", "emplace_hint", "operator[]", "at", "swap", "merge",
        "extract", "contains", "try_emplace",
    ],
    "unordered_map": [
        "find", "insert", "erase", "clear", "size", "empty", "begin", "end",
        "count", "bucket_count", "bucket", "load_factor", "max_load_factor",
        "rehash", "reserve", "emplace", "operator[]", "at", "swap",
        "contains", "try_emplace",
    ],
    "set": [
        "insert", "erase", "clear", "size", "empty", "begin", "end",
        "rbegin", "rend", "find", "count", "lower_bound", "upper_bound",
        "equal_range", "emplace", "emplace_hint", "swap", "merge",
        "extract", "contains",
    ],
    "unordered_set": [
        "insert", "erase", "clear", "size", "empty", "begin", "end",
        "count", "bucket_count", "bucket", "find", "rehash", "reserve",
        "emplace", "swap", "contains",
    ],
    "optional": [
        "has_value", "value", "value_or", "reset", "emplace", "operator bool",
        "operator*", "operator->",
    ],
    "variant": [
        "index", "holds_alternative", "get", "get_if", "visit", "index",
        "valueless_by_exception",
    ],
    "unique_ptr": [
        "get", "release", "reset", "swap", "operator bool", "operator[]",
        "operator*", "operator->", "get_deleter", "make_unique",
    ],
    "shared_ptr": [
        "get", "reset", "swap", "operator bool", "operator[]", "operator*",
        "operator->", "use_count", "unique", "lock", "expired", "make_shared",
        "allocate_shared", "static_pointer_cast", "dynamic_pointer_cast",
        "const_pointer_cast", "shared_from_this",
    ],
    "array": [
        "size", "empty", "front", "back", "data", "begin", "end", "rbegin",
        "rend", "fill", "swap", "operator[]", "at",
    ],
    "tuple": [
        "get", "make_tuple", "tie", "forward_as_tuple", "tuple_size",
        "tuple_element", "get<0>", "get<1>", "get<2>",
    ],
    "function": ["target_type", "target", "operator()", "swap", "function"],
    "chrono": [
        "now", "duration_cast", "time_point", "system_clock", "steady_clock",
        "high_resolution_clock", "nanoseconds", "microseconds", "milliseconds",
        "seconds", "minutes", "hours",
    ],
    "thread": [
        "join", "detach", "joinable", "get_id", "hardware_concurrency",
        "native_handle", "thread",
    ],
    "mutex": ["lock", "unlock", "try_lock", "try_lock_for", "try_lock_until"],
    "atomic": [
        "load", "store", "exchange", "compare_exchange_weak",
        "compare_exchange_strong", "fetch_add", "fetch_sub", "fetch_and",
        "fetch_or", "fetch_xor",
    ],
    "iostream": ["cin", "cout", "cerr", "endl", "getline", "flush", "peek", "put",
                 "ignore", "write", "read"],
    "fstream": [
        "open", "close", "is_open", "read", "write", "good", "eof",
        "fail", "clear", "gcount", "putback", "seekg", "seekp", "tellg", "tellp",
        "getline",
    ],
    "sstream": ["str", "clear", "get", "ignore", "peek", "putback",
                "seekg", "seekp", "tellg", "tellp", "str"],
}

# ── Receiver type hints (name → impl type) ─────────────────────────────

_RECEIVER_HINTS: dict[str, str] = {
    # Rust patterns
    "graph": "Graph", "main_graph": "Graph", "app_graph": "Graph", "g": "Graph",
    "motif": "MotifBuilder", "mb": "MotifBuilder", "motif_builder": "MotifBuilder",
    "store": "Store", "db": "Store", "database": "Store",
    "vec": "Vec", "vecdeque": "VecDeque", "deque": "VecDeque",
    "v": "Vec", "vs": "Vec", "items": "Vec", "elements": "Vec",
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
    # Python patterns
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
    # JS patterns
    "arr": "Array", "list": "Array", "items": "Array", "elements": "Array",
    "map": "Map", "dict": "Map", "obj_map": "Map",
    "set": "Set", "unique": "Set", "distinct": "Set",
    "promise": "Promise",
    "str": "String", "text": "String", "msg": "String",
    "err": "Error", "error": "Error",
    "url": "URL", "search": "URLSearchParams",
}


# ── Build stub lookup: method_name -> [(type_name, dialect)] ───────────

def _build_stub_lookup() -> dict[str, list[tuple[str, str]]]:
    """Build a reverse lookup: method_name -> list of (type_name, dialect)."""
    lookup: dict[str, list[tuple[str, str]]] = {}

    # Rust stdlib (always available)
    for type_name, methods in _RUST_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "rust"))

    # JS/TS globals (always available)
    for type_name, methods in _JAVASCRIPT_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "javascript"))

    # C++ STL (always available)
    for type_name, methods in _CPP_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "cpp"))

    # Python stubs (available globally for Python projects)
    for type_name, methods in _PYTHON_STUBS.items():
        for method in methods:
            lookup.setdefault(method, []).append((type_name, "python"))

    return lookup


# Cached stub lookup
_STUB_LOOKUP: dict[str, list[tuple[str, str]]] | None = None


def _get_stub_lookup() -> dict[str, list[tuple[str, str]]]:
    global _STUB_LOOKUP
    if _STUB_LOOKUP is None:
        _STUB_LOOKUP = _build_stub_lookup()
    return _STUB_LOOKUP


# ── Resolver ───────────────────────────────────────────────────────────

def _create_stub_node(graph: Graph, type_name: str, dialect: str) -> NodeId | None:
    """Create or return an existing stub node for a library type."""
    stub_name = f"stub::{type_name}"
    existing = graph.find_by_qname(stub_name)
    if existing is not None:
        return existing

    # Find source file for stub node
    source_uri = f"<{dialect}-stdlib:{type_name}>"

    # Create stub node
    node = {
        "id": None,  # Will be assigned by add_node
        "name": type_name,
        "qualified_name": stub_name,
        "kind": NodeKind.CLASS,
        "source_uri": source_uri,
        "line_start": None,
        "line_end": None,
        "source_text": f"Library stub for {type_name} ({dialect})",
        "valid_from": None,
        "valid_to": None,
        "decorators": [],
        "properties": {"dialect": dialect, "is_stub": True},
    }

    return graph.add_node(node)


def resolve_library_stubs(graph: Graph) -> int:
    """Resolve ``call::name`` placeholders against library stubs (Tier 7).

    Creates stub nodes for known library types (Vec, HashMap, String, etc.)
    and connects unresolved ``call::name`` edges to them when a method
    match is found.

    This is the **7th (final) tier** of call-placeholder resolution.
    It runs after the 6-tier heuristic resolver exhausts all
    local-disambiguation strategies.

    Args:
        graph: The code graph to resolve stubs in.

    Returns:
        Number of new stub edges added.
    """
    lookup = _get_stub_lookup()

    # Gather all stub type names that already have nodes (avoid duplicates)
    stub_type_nodes: dict[str, NodeId] = {}
    for nid, node in graph.nodes():
        if node.qualified_name.startswith("stub::"):
            type_name = node.qualified_name[len("stub::"):]
            stub_type_nodes[type_name] = nid

    # Collect edges to process (avoid mutation during iteration)
    edge_data = []
    for eid, src, dst, edge in graph.edges():
        if edge.kind != EdgeKind.CALLS:
            continue
        dst_node = graph.node(dst)
        if dst_node is None:
            continue
        qn = dst_node.qualified_name
        if not qn.startswith("call::"):
            continue
        if should_suppress_call_placeholder(qn[6:]):
            continue
        edge_data.append((eid, src, dst, edge, qn))

    additions = 0

    for eid, src, dst, edge, callee_qn in edge_data:
        method_name = callee_qn[6:]  # strip "call::"
        candidates = lookup.get(method_name)
        if candidates is None:
            continue

        # Pick the best candidate (prefer exact match, then Rust, then JS)
        best_type = None
        best_dialect = None
        for type_name, dialect in candidates:
            if type_name == method_name:
                best_type = type_name
                best_dialect = dialect
                break

        if best_type is None:
            best_type, best_dialect = candidates[0]

        # Get or create the stub node
        stub_id = stub_type_nodes.get(best_type)
        if stub_id is None:
            stub_id = _create_stub_node(graph, best_type, best_dialect)
            if stub_id is not None:
                stub_type_nodes[best_type] = stub_id

        if stub_id is not None:
            # Remove the old unresolved edge and add a stub edge
            # (We skip the removal here to keep the graph simple;
            # the old edge stays but the new stub edge provides a resolution.)
            graph.add_edge(src, stub_id, Edge(kind=edge.kind, confidence=Confidence.inferred(), valid_from=edge.valid_from, valid_to=edge.valid_to))
            additions += 1

    return additions


def resolve_library_stubs_batch(graph: Graph) -> dict[str, Any]:
    """Full batch resolution with statistics.

    Args:
        graph: The code graph.

    Returns:
        Statistics dict with total placeholders, resolved, and unresolved.
    """
    total = 0
    resolved = 0

    # Count unresolved call placeholders
    for nid, node in graph.nodes():
        if node.qualified_name.startswith("call::") and not should_suppress_call_placeholder(node.qualified_name[6:]):
            total += 1

    # Resolve
    added = resolve_library_stubs(graph)
    resolved = added

    # Re-count after resolution
    remaining = 0
    for nid, node in graph.nodes():
        if node.qualified_name.startswith("call::") and not should_suppress_call_placeholder(node.qualified_name[6:]):
            remaining += 1

    return {
        "operation": "library_stubs",
        "total_unresolved": total,
        "resolved": resolved,
        "unresolved_remaining": remaining,
        "resolution_rate": round(resolved / max(1, total), 3) if total > 0 else 0.0,
    }
