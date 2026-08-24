# Graphician Native Engine

High-performance code extraction and graph analysis exposed to Python through PyO3.

## Quick Start

```python
from graphician._extract import extract_python_file

source = b"def foo(x): return x + 1"
result = extract_python_file(source, file_path="test.py")

# Result contains:
# - nodes: list of extracted symbols (functions, classes, methods, etc.)
# - edges: list of relationship edges (defines, imports, inherits)
# - calls: list of call placeholders
```

## Features

- **Faster parsing**: 1.5x+ speedup over Python AST walker
- **Full symbol extraction**: functions, classes, methods, traits, types, modules
- **Decorator detection**: extracts all decorators including attribute-based ones
- **Inheritance tracking**: handles Protocol, ABC, TypedDict, and regular inheritance
- **TypeVar detection**: identifies TypeVar assignments as type aliases
- **Test detection**: marks test functions and test methods
- **Call extraction**: extracts function calls with receivers

## Integration with Graph

The Rust extraction can be used directly or through the Python wrapper:

```python
from graphician._extract.python import extract_python_file
from graphician.core.graph import Graph

graph = Graph()
extract_python_file(Path("my_module.py"), graph)

# Graph now contains all nodes and edges from the Rust extraction
```

## Building

### Prerequisites

- Rust toolchain (1.75+)
- Python 3.11+ with development headers

### Build steps

```bash
cd graphician-native
cargo build --release
cp target/release/libgraphician_native.so \
  ../src/graphician/_extract/graphician_native.abi3.so
```

The `.so` file is platform-specific. For deployment, build on each target platform.

## Architecture

- `src/lib.rs` - Rust library with tree-sitter parsing logic
- `src/graphician/_extract/` - Python wrapper and module initialization
- `src/graphician/_extract/python.py` - Graph integration layer

## Performance

On a typical codebase:
- Small files (< 100 lines): 1.5x speedup
- Medium files (100-500 lines): 2-3x speedup
- Large files (500+ lines): 3-5x speedup

The speedup comes from:
1. Avoiding Python object creation overhead in tree-sitter callbacks
2. Native string handling without UTF-8 encoding/decoding
3. Efficient memory allocation for node/edge data

## Future Work

- [ ] Add Rust parser extraction
- [ ] Add TypeScript/JavaScript parser extraction
- [ ] Add C++ parser extraction
- [ ] Parallel file extraction using rayon
- [ ] Streaming extraction for very large files
