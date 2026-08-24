"""Rust stdlib and common crate stubs for call resolution.

Contains stub definitions for Rust standard library types and commonly
used crates (tokio, serde_json, tracing, etc.).
"""

from __future__ import annotations

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
