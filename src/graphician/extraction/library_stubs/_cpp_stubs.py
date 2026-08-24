"""C++ STL stubs for call resolution.

Contains stub definitions for C++ standard library components including
containers, algorithms, I/O, threading, and smart pointers.
"""

from __future__ import annotations

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
