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

    # ── Additional STL containers ─────────────────────────────────────

    "deque": [
        "push_back", "push_front", "pop_back", "pop_front", "size", "empty",
        "clear", "front", "back", "begin", "end", "rbegin", "rend",
        "insert", "erase", "assign", "swap", "reserve", "emplace",
        "emplace_back", "emplace_front", "at", "operator[]",
        "resize", "shrink_to_fit", "get_allocator",
    ],
    "list": [
        "push_back", "push_front", "pop_back", "pop_front", "size", "empty",
        "clear", "front", "back", "begin", "end", "rbegin", "rend",
        "insert", "erase", "assign", "swap", "emplace", "emplace_back",
        "emplace_front", "resize", "remove", "remove_if", "unique",
        "sort", "merge", "splice", "reverse", "shuffle",
    ],
    "forward_list": [
        "push_front", "pop_front", "size", "empty", "clear", "before_begin",
        "begin", "end", "cbefore_begin", "cbegin", "cend", "insert_after",
        "emplace_after", "erase_after", "erase", "resize", "remove",
        "remove_if", "unique", "sort", "merge", "splice_after",
        "splice_after", "reverse", "swap",
    ],
    "stack": [
        "push", "pop", "top", "empty", "size", "emplace", "swap",
    ],
    "queue": [
        "push", "pop", "front", "back", "empty", "size", "emplace",
    ],
    "priority_queue": [
        "push", "pop", "top", "empty", "size", "emplace", "swap",
    ],
    "bitset": [
        "set", "reset", "flip", "test", "count", "size", "any", "none",
        "all", "to_string", "to_ulong", "to_ullong", "operator[]",
        "reset", "flip", "test", "count", "size", "any", "none",
        "all", "to_string", "to_ulong", "to_ullong", "operator[]",
    ],

    # ── STL algorithms ────────────────────────────────────────────────

    "algorithm": [
        "sort", "stable_sort", "partial_sort", "nth_element", "partition",
        "stable_partition", "lower_bound", "upper_bound", "equal_range",
        "binary_search", "find", "find_if", "find_if_not", "find_end",
        "find_first_of", "search", "search_n", "count", "count_if",
        "mismatch", "equal", "lexicographical_compare",
        "max", "min", "minmax", "max_element", "min_element",
        "swap", "iter_swap", "swap_ranges", "move", "move_backward",
        "fill", "fill_n", "generate", "generate_n", "iota",
        "transform", "transform_if", "replace", "replace_if",
        "replace_copy", "replace_copy_if", "remove", "remove_if",
        "remove_copy", "remove_copy_if", "unique", "unique_copy",
        "reverse", "reverse_copy", "rotate", "rotate_copy",
        "next_permutation", "prev_permutation", "random_shuffle",
        "shuffle", "sample", "partial_sort_copy", "inplace_merge",
        "merge", "set_union", "set_intersection", "set_difference",
        "set_symmetric_difference", "includes", "includes", "copy",
        "copy_if", "copy_n", "copy_backward", "move", "move_backward",
        "iter_swap", "iter_swap", "iter_swap", "swap_ranges",
        "swap_ranges", "swap_ranges", "swap_ranges",
        "for_each", "for_each_n", "reduce", "transform_reduce",
        "transform_reduce", "reduce", "transform_reduce",
        "find_first_of", "search", "search_n", "count", "count_if",
        "mismatch", "equal", "lexicographical_compare",
        "max", "min", "minmax", "max_element", "min_element",
    ],

    # ── STL filesystem ────────────────────────────────────────────────

    "filesystem": [
        "path", "directory_iterator", "recursive_directory_iterator",
        "directory_entry", "file_status", "file_type", "space_info",
        "file_size", "hard_link_count", "last_write_time",
        "exists", "is_regular_file", "is_directory", "is_symlink",
        "is_block_file", "is_character_file", "is_fifo", "is_socket",
        "is_other", "is_empty", "is_complete", "is_relative",
        "is_absolute", "has_extension", "has_filename", "has_parent_path",
        "has_stem", "has_suffix", "has_root_name", "has_root_directory",
        "has_root_path", "lexically_normal", "lexically_relative",
        "lexically_proximate", "native", "string", "u8string",
        "generic_string", "generic_u8string", "c_str", "data",
        "operator/", "operator/=", "append", "replace_filename",
        "replace_extension", "remove_filename", "stem", "extension",
        "suffix", "parent_path", "root_name", "root_directory",
        "root_path", "filename", "relative_path", "path",
        "create_directory", "create_directories", "remove",
        "remove_all", "rename", "rename_copy", "copy",
        "copy_file", "copy_symlink", "hard_link_count",
        "create_hard_link", "create_symlink", "create_directory_symlink",
        "read_symlink", "canonical", "weakly_canonical",
        "relative", "proximate", "space", "last_write_time",
        "set_last_write_time", "file_size", "permissions",
        "status", "symlink_status", "file_status", "status_known",
        "permissions::none", "permissions::owner_read",
        "permissions::owner_write", "permissions::owner_exec",
        "permissions::group_read", "permissions::group_write",
        "permissions::group_exec", "permissions::others_read",
        "permissions::others_write", "permissions::others_exec",
        "permissions::add", "permissions::remove",
        "permissions::set", "permissions::nofollow",
    ],

    # ── STL regex ─────────────────────────────────────────────────────

    "regex": [
        "regex", "regex_search", "regex_match", "regex_replace",
        "regex_iterator", "regex_token_iterator", "regex_split",
        "regex_error", "regex_constants", "error_type",
        "ECMAScript", "basic", "extended", "awk", "grep", "egrep",
        "match", "search", "find_first_of", "find_all", "find_once",
        "sub_match", "match_results", "cmatch", "smatch", "ccmatch",
        "wcmatch", "regex_word_boundary", "regex_not_word_boundary",
        "regex_line_boundary", "regex_any", "regex_empty",
        "regex_swap", "regex_alnum", "regex_alpha", "regex_blank",
        "regex_cntrl", "regex_digit", "regex_graph", "regex_lower",
        "regex_print", "regex_punct", "regex_space", "regex_upper",
        "regex_xdigit", "regex_unicode", "regex_charclass",
    ],

    # ── STL numeric ───────────────────────────────────────────────────

    "numeric": [
        "accumulate", "adjacent_difference", "inner_product",
        "partial_sum", "multiplies", "divides", "modulus", "negate",
        "plus", "minus", "logical_and", "logical_or", "logical_not",
        "equal_to", "not_equal_to", "greater", "less", "greater_equal",
        "less_equal", "plus", "minus", "multiplies", "divides",
        "modulus", "negate", "accumulate", "adjacent_difference",
        "inner_product", "partial_sum", "iota", "reduce",
        "transform_reduce", "transform_exclusive_scan",
        "transform_inclusive_scan", "transform_scan",
    ],

    # ── STL utility ───────────────────────────────────────────────────

    "utility": [
        "pair", "make_pair", "swap", "move", "forward", "forward_like",
        "swap", "declval", "addressof", "addressof_const", "addressof_v",
        "exchange", "rel_ops", "operator!=", "operator<", "operator<=",
        "operator>", "operator>=", "operator==", "move_if_noexcept",
        "piecewise_construct", "piecewise_construct_t", "launder",
        "restrict", "assume_aligned", "assume_aligned_v",
    ],

    # ── STL memory ────────────────────────────────────────────────────

    "memory": [
        "allocator", "allocator_traits", "scoped_allocator_adaptor",
        "make_shared", "allocate_shared", "make_unique", "allocate_unique",
        "enable_shared_from_this", "shared_from_this", "owner_before",
        "bad_weak_ptr", "weak_ptr", "enable_shared_from_this",
        "default_delete", "owner_before", "bad_weak_ptr",
        "allocator_traits", "scoped_allocator_adaptor",
    ],

    # ── STL type traits ───────────────────────────────────────────────

    "type_traits": [
        "is_same", "is_base_of", "is_convertible", "is_abstract",
        "is_arithmetic", "is_array", "is_class", "is_const",
        "is_constructible", "is_copy_constructible", "is_move_constructible",
        "is_copy_assignable", "is_move_assignable", "is_destructible",
        "is_default_constructible", "is_empty", "is_enum",
        "is_final", "is_function", "is_fundamental", "is_literal_type",
        "is_member_function_pointer", "is_member_object_pointer",
        "is_nothrow_copy_constructible", "is_nothrow_move_constructible",
        "is_nothrow_copy_assignable", "is_nothrow_move_assignable",
        "is_nothrow_destructible", "is_nothrow_default_constructible",
        "is_polymorphic", "is_reference", "is_rvalue_reference",
        "is_scalar", "is_signed", "is_standard_layout", "is_trivial",
        "is_trivially_copy_assignable", "is_trivially_copy_constructible",
        "is_trivially_default_constructible", "is_trivially_destructible",
        "is_trivially_move_assignable", "is_trivially_move_constructible",
        "is_union", "is_unsigned", "is_volatile", "remove_cv",
        "remove_const", "remove_volatile", "remove_reference",
        "remove_pointer", "add_pointer", "add_reference",
        "add_lvalue_reference", "add_rvalue_reference",
        "remove_cvref", "remove_all_extents", "remove_extent",
        "decay", "enable_if", "disable_if", "conditional",
        "common_type", "underlying_type", "result_of",
        "invoke_result", "void_t", "void_pointer", "void_type",
        "bool_constant", "integral_constant", "true_type", "false_type",
        "negation", "conjunction", "disjunction", "identity",
        "void", "void_t", "void_pointer", "void_type",
    ],

    # ── STL optional (C++17) ──────────────────────────────────────────

    "optional": [
        "has_value", "value", "value_or", "reset", "emplace",
        "operator bool", "operator*", "operator->", "operator==",
        "operator!=", "operator<", "operator<=", "operator>", "operator>=",
        "make_optional", "nullopt", "nullopt_t", "monadic_and_then",
        "monadic_or_else", "value_or_throw", "expect", "transform",
        "and_then", "or_else", "value_or_throw",
    ],

    # ── STL variant (C++17) ───────────────────────────────────────────

    "variant": [
        "index", "holds_alternative", "get", "get_if", "visit",
        "valueless_by_exception", "monadic_and_then", "monadic_or_else",
        "variant_size", "variant_alternative", "variant_size_v",
        "variant_alternative_t", "holds_alternative_v",
        "monadic_and_then", "monadic_or_else", "variant_size",
        "variant_alternative", "variant_size_v", "variant_alternative_t",
        "holds_alternative_v",
    ],

    # ── STL any (C++17) ───────────────────────────────────────────────

    "any": [
        "has_value", "type", "cast", "emplace", "reset", "clear",
        "has_value", "type", "cast", "emplace", "reset", "clear",
        "any_cast", "any_cast", "any_cast", "any_cast", "any_cast",
        "any_size", "any_size_v", "any_type", "any_type_v",
    ],

    # ── STL string_view (C++17) ───────────────────────────────────────

    "string_view": [
        "size", "length", "max_size", "empty", "data", "c_str",
        "operator[]", "at", "front", "back", "begin", "cbegin",
        "end", "cend", "rbegin", "crbegin", "rend", "crend",
        "substr", "remove_prefix", "remove_suffix", "compare",
        "starts_with", "ends_with", "contains", "find", "rfind",
        "find_first_of", "find_last_of", "find_first_not_of",
        "find_last_not_of", "operator==", "operator!=",
        "operator<", "operator<=", "operator>", "operator>=",
        "operator<=>", "hash_value", "operator<<",
    ],

    # ── Boost libraries ───────────────────────────────────────────────

    "boost_optional": [
        "get", "get_ptr", "operator*", "operator->", "operator bool",
        "operator==", "operator!=", "has_value", "value_or",
        "value_or_throw", "reset", "operator=", "swap",
        "make_optional", "nullopt", "nullopt_t", "indirectly_unary_invocable",
        "optional", "optional_lite", "optional_t", "nullopt_t",
        "make_optional", "make_optional_in_place", "in_place",
        "in_place_t", "in_place_index_t", "in_place_type_t",
    ],
    "boost_variant": [
        "get", "get_ptr", "holds_alternative", "index", "apply_visitor",
        "static_visitor", "visitor", "visit", "apply_visitor",
        "static_visitor", "visitor", "visit", "apply_visitor",
        "variant", "variant_lite", "variant_size", "variant_alternative",
        "variant_size_v", "variant_alternative_t", "holds_alternative_v",
        "get", "get_ptr", "holds_alternative", "index",
    ],
    "boost_any": [
        "has_value", "type", "cast", "reset", "clear", "swap",
        "any", "any_cast", "any_size", "any_size_v", "any_type",
        "has_value", "type", "cast", "reset", "clear", "swap",
    ],
    "boost_string_view": [
        "size", "length", "max_size", "empty", "data", "c_str",
        "operator[]", "at", "front", "back", "begin", "cbegin",
        "end", "cend", "rbegin", "crbegin", "rend", "crend",
        "substr", "remove_prefix", "remove_suffix", "compare",
        "starts_with", "ends_with", "contains", "find", "rfind",
        "find_first_of", "find_last_of", "find_first_not_of",
        "find_last_not_of", "operator==", "operator!=",
        "operator<", "operator<=", "operator>", "operator>=",
        "operator<=>", "hash_value", "operator<<",
    ],
    "boost_thread": [
        "thread", "mutex", "recursive_mutex", "shared_mutex", "scoped_lock",
        "lock_guard", "unique_lock", "shared_lock", "try_lock",
        "try_lock_for", "try_lock_until", "lock", "unlock",
        "condition_variable", "condition_variable_any", "notify_all",
        "notify_one", "wait", "wait_for", "wait_until",
        "jthread", "stop_token", "stop_source", "stop_callback",
        "thread", "mutex", "recursive_mutex", "shared_mutex", "scoped_lock",
        "lock_guard", "unique_lock", "shared_lock", "try_lock",
        "try_lock_for", "try_lock_until", "lock", "unlock",
        "condition_variable", "condition_variable_any", "notify_all",
        "notify_one", "wait", "wait_for", "wait_until",
    ],
    "boost_chrono": [
        "duration", "time_point", "system_clock", "steady_clock",
        "high_resolution_clock", "nanoseconds", "microseconds",
        "milliseconds", "seconds", "minutes", "hours", "now",
        "duration_cast", "time_point_cast", "clock_t",
        "duration", "time_point", "system_clock", "steady_clock",
        "high_resolution_clock", "nanoseconds", "microseconds",
        "milliseconds", "seconds", "minutes", "hours", "now",
        "duration_cast", "time_point_cast", "clock_t",
    ],
    "boost_filesystem": [
        "path", "directory_iterator", "recursive_directory_iterator",
        "file_status", "file_type", "space_info", "file_size",
        "exists", "is_regular_file", "is_directory", "is_symlink",
        "is_block_file", "is_character_file", "is_fifo", "is_socket",
        "is_other", "is_empty", "is_complete", "is_relative",
        "is_absolute", "has_extension", "has_filename", "has_parent_path",
        "has_stem", "has_suffix", "has_root_name", "has_root_directory",
        "has_root_path", "lexically_normal", "lexically_relative",
        "create_directory", "create_directories", "remove",
        "remove_all", "rename", "copy", "copy_file",
        "canonical", "weakly_canonical", "relative", "proximate",
        "space", "last_write_time", "file_size", "permissions",
        "status", "symlink_status", "file_status", "status_known",
    ],
}
