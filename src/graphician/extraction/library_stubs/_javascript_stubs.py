"""JavaScript / TypeScript stubs for call resolution.

Contains stub definitions for JavaScript/TypeScript built-in types,
DOM APIs, and Node.js globals.
"""

from __future__ import annotations

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
        "exitCode", "on", "emit", "hrtime",
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
