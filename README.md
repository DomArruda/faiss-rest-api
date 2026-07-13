# faiss-rest-api

A **thin REST API** over FAISS (C++). Exposes core FAISS operations (create via `index_factory`, add, train, search, save/load indices) as a lightweight HTTP/JSON API. Designed for simplicity, minimal dependencies beyond FAISS itself, and compiles to a single binary (ideally fully static).

- **Single binary goal**: Header-only HTTP (cpp-httplib) + JSON (nlohmann/json) via CMake FetchContent + static-linked FAISS (when you build FAISS with `BUILD_SHARED_LIBS=OFF` + static BLAS).
- **Thin**: No heavy frameworks, direct 1:1 mapping to FAISS C++ calls, coarse-grained locking for thread-safety (sufficient for dev/small-scale use).
- **Use cases**: Quick vector search microservice, prototyping, embedding in larger C++ systems, or as a minimal vector DB backend.

## Quick Start (on your machine with FAISS + build tools)

### 1. Prerequisites (Ubuntu / WSL / Debian)

```bash
sudo apt update
sudo apt install -y build-essential cmake git \
    libopenblas-dev liblapack-dev pkg-config
```

### 2. Build FAISS (CPU, static recommended)

**Important**: Use a *clean* build directory. The previous command you ran had a shell parsing error because of the inline comment and line continuations.

```bash
git clone https://github.com/facebookresearch/faiss.git
cd faiss
rm -rf build
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_BLAS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build --parallel $(nproc)
sudo cmake --install build
```

- `-DBUILD_SHARED_LIBS=OFF` is the key for easier single-binary linking later.
- FAISS will auto-detect OpenBLAS when the dev package is installed.
- If you want a fully static binary later, you can experiment with extra linker flags (see below).
- See `faiss/INSTALL.md` (in the cloned repo) for GPU or MKL options if needed.

### 3. Build this REST API (after FAISS is installed)

```bash
# Extract the tarball you downloaded (or git clone / copy the folder)
cd faiss-rest-api
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# If find_package still fails:
#   cmake .. -Dfaiss_DIR=/usr/local/lib/cmake/faiss -DCMAKE_BUILD_TYPE=Release

cmake --build . --parallel $(nproc)
```

The binary will be at `build/faiss_rest_api`.

**Common fix for "cmake_install.cmake" error**: You ran `cmake --install` (or equivalent) on a build directory whose `cmake` configure step had previously failed. Always `rm -rf build` and re-run the two `cmake` commands above after fixing any dependency issues.

### 4. Run

```bash
./faiss_rest_api [port]   # default 8080
```

Server listens on `0.0.0.0:port`. Test with `curl http://localhost:8080/`

### Single-binary / static linking tips

After you have a working dynamic build, you can try:

```bash
cd build
rm -rf *
cmake .. -DBUILD_STATIC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel
```

This attempts `-static -static-libgcc -static-libstdc++`. Success depends on having static versions of OpenBLAS + transitive libs available on your system. On many Linux setups it works well; on WSL it can be trickier — start with the normal (dynamic) build.

### 2. Build this REST API

```bash
cd faiss-rest-api
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
# Optional: -DBUILD_STATIC=ON (experimental, requires static FAISS + deps)
cmake --build . --parallel
```

The binary `faiss_rest_api` will be in `build/`.

If `find_package(faiss)` fails, pass the install location:
```bash
cmake .. -Dfaiss_DIR=/usr/local/lib/cmake/faiss
# or add to CMAKE_PREFIX_PATH
```

### 3. Run

```bash
./faiss_rest_api [port]   # default 8080
# e.g. ./faiss_rest_api 9000
```

Server listens on `0.0.0.0:port`.

## API Endpoints (JSON)

All responses are JSON. Errors return `{"error": "..."}` with appropriate HTTP status (400/404/500).

### Health / Info
```
GET /
```
Returns basic status and index count.

### List Indices
```
GET /indices
```
Returns `{ "indices": [ {name, dimension, ntotal, is_trained, metric_type}, ... ] }`

### Create Index
```
POST /indices
Content-Type: application/json

{
  "name": "myvecs",
  "dimension": 128,
  "metric_type": "L2",           # or "IP"
  "description": "Flat"          # or "IVF256,Flat", "HNSW32", "PQ16x8" etc.
                                 # See FAISS index_factory docs
}
```
- Uses `faiss::index_factory`.
- Returns 201-like info or error (bad description, dup name, etc.).

### Load Existing Index from Disk
```
POST /load
{
  "name": "myvecs",
  "path": "/data/myvecs.faiss"
}
```
Loads via `faiss::read_index`. Overwrites name if exists.

### Save Index to Disk
```
POST /save
{
  "name": "myvecs",
  "path": "/data/myvecs.faiss"
}
```
Saves via `faiss::write_index`. Creates parent dirs? No (user ensure path writable).

### Delete Index (from memory)
```
DELETE /indices/{name}
```

### Add Vectors (optionally with custom IDs)
```
POST /indices/{name}/add
{
  "vectors": [
    [0.1, 0.2, ..., 0.128],
    ...
  ],
  "ids": [1001, 1002, ...]     # optional, same length; uses add_with_ids if present
}
```
- `ids` are int64.
- Returns new `ntotal`.

**Note on training**: For IVF/HNSW/PQ etc. you **must** call `/train` first (or use indices that don't require it like Flat). Otherwise `add` will fail or produce bad results.

### Train Index (for IVF etc.)
```
POST /indices/{name}/train
{
  "vectors": [ [..], ... ]   # representative training set
}
```
Calls `index->train()`. Idempotent-ish; some indices become trained after.

### Search
```
POST /indices/{name}/search
{
  "query_vectors": [ [0.1,...], ... ],
  "k": 10,
  # optional: "params": { "nprobe": 8 } for IVF etc. (advanced, extend as needed)
}
```
Response:
```json
{
  "results": [
    {
      "query_index": 0,
      "neighbors": [
        {"id": 42, "distance": 0.123},
        ...
      ]
    },
    ...
  ]
}
```
- Returns up to `k` per query (may include `id=-1` if fewer results available).
- Distances are raw (L2 or negative IP depending on metric).

### Get Details for One Index
```
GET /indices/{name}
```
Similar to list item + more if you extend.

## Thread Safety & Concurrency Notes
- Coarse global mutex around all operations (map + index mutations/searches).
- Fine for development, local use, or moderate load.
- For high QPS: move to per-index `std::shared_mutex` (reader/writer), shard indices, or use async workers. FAISS search is CPU-bound; consider batching.

## Extending (Ideas for Thin but More Powerful)
- Add `search_params` support (nprobe, efSearch, etc.) via `faiss::SearchParameters*` subclasses.
- Binary protocol or MessagePack for large vector batches (JSON has size limits).
- Persistent index registry (auto-load from a directory on startup).
- GPU support (faiss::gpu::index_factory + move to GPU).
- Authentication, TLS (httplib supports OpenSSL), rate limiting.
- Prometheus `/metrics` endpoint.
- But remember the goal: **thin**.

## Single Binary Tips
- Build FAISS static + this with `-DBUILD_STATIC=ON` + static OpenBLAS.
- On Alpine/musl you get smaller fully static binaries more easily.
- `strip --strip-all faiss_rest_api`
- Result: one executable you can `scp` and run anywhere compatible (no .so hell for FAISS/BLAS).

## Limitations / Gotchas
- JSON vector transport: fine for <~10k vectors/batch; for huge use chunking or extend to binary upload.
- No built-in sharding/replication (that's what Milvus/Pinecone etc. add on top of FAISS).
- Error messages from FAISS are propagated (e.g. "not trained", dim mismatch).
- IDs: custom IDs must be unique per index for some structures (HNSW etc.).

## Example curl Session

```bash
# create
curl -X POST http://localhost:8080/indices -H 'Content-Type: application/json' \
  -d '{"name":"demo","dimension":2,"description":"Flat"}'

# add
curl -X POST http://localhost:8080/indices/demo/add -H 'Content-Type: application/json' \
  -d '{"vectors":[[0.1,0.2],[0.3,0.4]],"ids":[10,20]}'

# search
curl -X POST http://localhost:8080/indices/demo/search -H 'Content-Type: application/json' \
  -d '{"query_vectors":[[0.15,0.25]],"k":5}'

# save
curl -X POST http://localhost:8080/save -H 'Content-Type: application/json' \
  -d '{"name":"demo","path":"./demo.faiss"}'
```

## Why This Instead of ... ?
- Python + FastAPI + faiss-cpu: heavier runtime, GIL, not "C++".
- Full vector DBs (Milvus, Weaviate, Qdrant): much heavier, more features you may not need.
- This: pure C++ speed, minimal attack surface, single binary deploy, full control over FAISS features via the factory string.

Built with ❤️ for low-level control and simplicity. Extend as needed.

If you hit build issues (esp. FAISS find_package or static linking), paste the error and I'll help refine the CMake.

Happy vector searching!
