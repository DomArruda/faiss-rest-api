#include <faiss/Index.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/impl/FaissException.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cctype>

using json = nlohmann::json;
using faiss::idx_t;

struct ServerState {
    std::unordered_map<std::string, std::unique_ptr<faiss::Index>> indices;
    std::mutex mtx;

    faiss::Index* get_index(const std::string& name) {
        auto it = indices.find(name);
        if (it == indices.end()) {
            throw std::runtime_error("index not found: " + name);
        }
        return it->second.get();
    }
};

namespace {

std::vector<float> json_to_vectors(const json& j, int64_t d, int64_t& n_out) {
    if (!j.is_array()) {
        throw std::runtime_error("vectors must be a JSON array of arrays");
    }
    n_out = static_cast<int64_t>(j.size());
    if (n_out == 0) return {};

    std::vector<float> buf;
    buf.reserve(static_cast<size_t>(n_out * d));

    for (const auto& row : j) {
        if (!row.is_array() || static_cast<int64_t>(row.size()) != d) {
            throw std::runtime_error("each vector must be an array of exactly 'dimension' floats");
        }
        for (const auto& val : row) {
            buf.push_back(val.get<float>());
        }
    }
    return buf;
}

std::vector<idx_t> json_to_ids(const json& j, int64_t expected_n) {
    if (!j.is_array()) {
        throw std::runtime_error("ids must be a JSON array");
    }
    if (static_cast<int64_t>(j.size()) != expected_n) {
        throw std::runtime_error("length of 'ids' must match length of 'vectors'");
    }
    std::vector<idx_t> ids(static_cast<size_t>(expected_n));
    for (int64_t i = 0; i < expected_n; ++i) {
        ids[static_cast<size_t>(i)] = j[static_cast<size_t>(i)].get<int64_t>();
    }
    return ids;
}

faiss::MetricType parse_metric_type(const std::string& s) {
    std::string lower;
    for (char c : s) lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower == "ip" || lower == "inner_product" || lower == "cosine") {
        return faiss::METRIC_INNER_PRODUCT;
    }
    return faiss::METRIC_L2;
}

std::string metric_type_to_string(faiss::MetricType m) {
    return (m == faiss::METRIC_INNER_PRODUCT) ? "IP" : "L2";
}

} // anonymous namespace

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--help" || arg == "-h") {
            std::cout << "faiss-rest-api - Thin REST API over FAISS\n\n"
                      << "Usage:\n"
                      << "  faiss_rest_api [port]\n\n"
                      << "Options:\n"
                      << "  --help, -h     Show this help message\n"
                      << "  [port]         Port to listen on (default: 8080)\n\n"
                      << "Examples:\n"
                      << "  faiss_rest_api\n"
                      << "  faiss_rest_api 9000\n";
            return 0;
        }
    }

    int port = 8080;
    if (argc > 1) {
        port = std::atoi(argv[1]);
        if (port <= 0 || port > 65535) {
            std::cerr << "Invalid port, using 8080\n";
            port = 8080;
        }
    }

    ServerState state;
    httplib::Server svr;

    svr.set_default_headers({
        {"Access-Control-Allow-Origin", "*"},
        {"Access-Control-Allow-Headers", "Content-Type"},
        {"Access-Control-Allow-Methods", "GET,POST,DELETE"}
    });

    // GET /
    svr.Get("/", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mtx);
        json j = {
            {"status", "ok"},
            {"service", "faiss-rest-api"},
            {"indices_count", state.indices.size()},
            {"port", port}
        };
        res.set_content(j.dump(2), "application/json");
    });

    // GET /indices
    svr.Get("/indices", [&](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mtx);
        json list = json::array();
        for (const auto& [name, idxp] : state.indices) {
            faiss::Index* idx = idxp.get();
            list.push_back({
                {"name", name},
                {"dimension", idx->d},
                {"ntotal", idx->ntotal},
                {"is_trained", idx->is_trained},
                {"metric_type", metric_type_to_string(idx->metric_type)}
            });
        }
        res.set_content(json{{"indices", list}}.dump(2), "application/json");
    });

    // GET /indices/{name}
    svr.Get(R"(/indices/([a-zA-Z0-9_-]+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            faiss::Index* idx = state.get_index(name);
            json item = {
                {"name", name},
                {"dimension", idx->d},
                {"ntotal", idx->ntotal},
                {"is_trained", idx->is_trained},
                {"metric_type", metric_type_to_string(idx->metric_type)}
            };
            res.set_content(item.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 404;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    // POST /indices (create)
    svr.Post("/indices", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            json j = json::parse(req.body);
            std::string name = j.at("name").get<std::string>();

            if (state.indices.count(name)) {
                throw std::runtime_error("index with this name already exists");
            }

            int64_t dim = j.at("dimension").get<int64_t>();
            if (dim <= 0) throw std::runtime_error("dimension must be positive");

            std::string metric_str = j.value("metric_type", "L2");
            auto metric = parse_metric_type(metric_str);
            std::string desc = j.value("description", "Flat");

            faiss::Index* raw = faiss::index_factory(static_cast<int>(dim), desc.c_str(), metric);
            state.indices[name] = std::unique_ptr<faiss::Index>(raw);

            json resp = {
                {"status", "created"},
                {"name", name},
                {"dimension", dim},
                {"description", desc},
                {"metric_type", metric_type_to_string(metric)}
            };
            res.status = 201;
            res.set_content(resp.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    // POST /load
    svr.Post("/load", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            json j = json::parse(req.body);
            std::string name = j.at("name").get<std::string>();
            std::string path = j.at("path").get<std::string>();

            faiss::Index* raw = faiss::read_index(path.c_str());
            state.indices[name] = std::unique_ptr<faiss::Index>(raw);

            json resp = {
                {"status", "loaded"},
                {"name", name},
                {"path", path},
                {"dimension", raw->d},
                {"ntotal", raw->ntotal}
            };
            res.set_content(resp.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    // POST /save
    svr.Post("/save", [&](const httplib::Request& req, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            json j = json::parse(req.body);
            std::string name = j.at("name").get<std::string>();
            std::string path = j.at("path").get<std::string>();

            faiss::Index* idx = state.get_index(name);
            faiss::write_index(idx, path.c_str());

            res.set_content(json{{"status", "saved"}, {"name", name}, {"path", path}}.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    // DELETE /indices/{name}
    svr.Delete(R"(/indices/([a-zA-Z0-9_-]+))", [&](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        std::lock_guard<std::mutex> lock(state.mtx);

        if (state.indices.erase(name) == 0) {
            res.status = 404;
            res.set_content(json{{"error", "index not found"}, {"name", name}}.dump(2), "application/json");
            return;
        }
        res.set_content(json{{"status", "deleted"}, {"name", name}}.dump(2), "application/json");
    });

    // POST /indices/{name}/add
    svr.Post(R"(/indices/([a-zA-Z0-9_-]+)/add)", [&](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            faiss::Index* idx = state.get_index(name);

            if (!idx->is_trained) {
                throw std::runtime_error(
                    "Index requires training before adding vectors. "
                    "Please call POST /indices/" + name + "/train first."
                );
            }

            json j = json::parse(req.body);
            json jvecs = j.at("vectors");
            int64_t n = 0;
            std::vector<float> x = json_to_vectors(jvecs, idx->d, n);

            json jids = j.value("ids", json::array());

            if (!jids.empty()) {
                std::vector<idx_t> ids = json_to_ids(jids, n);
                try {
                    idx->add_with_ids(static_cast<idx_t>(n), x.data(), ids.data());
                } catch (const faiss::FaissException& e) {
                    throw std::runtime_error(
                        "This index type does not support custom IDs via add_with_ids. "
                        "Remove the 'ids' field or use IndexIDMap."
                    );
                }
            } else {
                idx->add(static_cast<idx_t>(n), x.data());
            }

            res.set_content(json{
                {"status", "added"},
                {"name", name},
                {"added", n},
                {"ntotal", idx->ntotal}
            }.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    // POST /indices/{name}/train
    svr.Post(R"(/indices/([a-zA-Z0-9_-]+)/train)", [&](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            faiss::Index* idx = state.get_index(name);

            json j = json::parse(req.body);
            json jvecs = j.at("vectors");
            int64_t n = 0;
            std::vector<float> x = json_to_vectors(jvecs, idx->d, n);

            idx->train(static_cast<idx_t>(n), x.data());

            res.set_content(json{
                {"status", "trained"},
                {"name", name},
                {"trained_on", n},
                {"is_trained", idx->is_trained}
            }.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    // POST /indices/{name}/search
    svr.Post(R"(/indices/([a-zA-Z0-9_-]+)/search)", [&](const httplib::Request& req, httplib::Response& res) {
        std::string name = req.matches[1];
        std::lock_guard<std::mutex> lock(state.mtx);
        try {
            faiss::Index* idx = state.get_index(name);

            json j = json::parse(req.body);
            json jq = j.at("query_vectors");
            int64_t requested_k = j.value("k", 10);
            if (requested_k <= 0) requested_k = 10;

            // Never ask for more results than exist in the index
            int64_t effective_k = std::min(requested_k, idx->ntotal > 0 ? idx->ntotal : 1);

            int64_t nq = 0;
            std::vector<float> xq = json_to_vectors(jq, idx->d, nq);

            std::vector<float> distances(static_cast<size_t>(nq * effective_k));
            std::vector<idx_t> labels(static_cast<size_t>(nq * effective_k));

            idx->search(static_cast<idx_t>(nq), xq.data(), static_cast<idx_t>(effective_k),
                        distances.data(), labels.data());

            json results = json::array();
            for (int64_t qi = 0; qi < nq; ++qi) {
                json neighbors = json::array();
                for (int64_t r = 0; r < effective_k; ++r) {
                    size_t off = static_cast<size_t>(qi * effective_k + r);
                    if (labels[off] == -1) continue;   // Filter out invalid results
                    neighbors.push_back({{"id", labels[off]}, {"distance", distances[off]}});
                }
                results.push_back({{"query_index", qi}, {"neighbors", neighbors}});
            }

            res.set_content(json{
                {"results", results},
                {"k", requested_k},
                {"effective_k", effective_k},
                {"n_queries", nq}
            }.dump(2), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(2), "application/json");
        }
    });

    std::cout << "Starting faiss-rest-api on 0.0.0.0:" << port << "\n";
    std::cout << "  POST /indices to create, /load to load from .faiss file\n";
    std::cout << "  Thin C++ REST over FAISS\n";

    if (!svr.listen("0.0.0.0", port)) {
        std::cerr << "Failed to listen on port " << port << "\n";
        return 1;
    }
    return 0;
}
