# FAISS REST API

**A thin, high-performance REST API over FAISS (C++)**

Exposes core FAISS operations (`index_factory`, add, train, search, save/load) as a lightweight HTTP/JSON API. Designed for speed, simplicity, and easy deployment as a single binary.

- **Single binary** (cpp-httplib + nlohmann/json)
- **Full FAISS power** via `index_factory` strings
- **Minimal overhead** — ideal for embedding + search microservices

## Quick Start

### Using Docker

```bash
docker build -t faiss-rest-api .
docker run -d -p 8080:8080 --name faiss-api faiss-rest-api
# With benchmark mode:
docker run -d -p 8080:8080 --name faiss-api faiss-rest-api --benchmark
````



## CLI Options



```Bash
faiss_rest_api [options] [port]

Options:
  -h, --help        Show this help message
  --benchmark       Include "took_ms" timing in every JSON response

Examples:
  faiss_rest_api                    # default port 8080
  faiss_rest_api 9000
  faiss_rest_api --benchmark
  faiss_rest_api --benchmark 9000
```

## API Endpoints

All responses are JSON. Errors return {"error": "..."}.

|Method|Endpoint|Description|
|---|---|---|
|GET|/|Health check + stats|
|GET|/indices|List all indices|
|GET|/indices/{name}|Get index details|
|POST|/indices|Create new index|
|POST|/load|Load .faiss file|
|POST|/save|Save index to disk|
|DELETE|/indices/{name}|Delete index|
|POST|/indices/{name}/add|Add vectors (optional IDs)|
|POST|/indices/{name}/train|Train index (IVF, HNSW, PQ…)|
|POST|/indices/{name}/search|Search|

### Create Index Example



```Bash
curl -X POST http://localhost:8080/indices \
  -H "Content-Type: application/json" \
  -d '{
    "name": "demo",
    "dimension": 128,
    "description": "HNSW32",           
    "metric_type": "L2"
  }'
```

See the [FAISS index_factory documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) for more string examples (Flat, IVF256,Flat, OPQ16_64,IVF256,PQ16x4fs, etc.).

### Search Example (with timing)



```Bash
curl -X POST http://localhost:8080/indices/demo/search \
  -H "Content-Type: application/json" \
  -d '{
    "query_vectors": [[0.1, 0.2, ..., 0.128]],
    "k": 10
  }'
```

When --benchmark is enabled, every response includes:



```JSON
{
  ...
  "took_ms": 1.234
}
```

## Features & Limitations

- Full control over FAISS via index_factory
- Coarse-grained mutex (good for dev / moderate load)
- Single binary friendly
- No built-in persistence / auth / scaling (use external tools or extend)
