// examples/usage.js
// Basic usage example for faiss-rest-api
// Ensure your FAISS Server is already running (assumes local host 8080)
// To run with node =>    node/examples/usage.js
// To run with bun =>     bun/examples/usage.js


const http = require('http');
const util = require('util');

const BASE_URL = 'http://localhost:8080';

function makeRequest(method, path, body = null) {
  return new Promise((resolve, reject) => {
    const options = {
      method: method,
      hostname: 'localhost',
      port: 8080,
      path: path,
      headers: body ? { 'Content-Type': 'application/json' } : {}
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        try {
          resolve({ status: res.statusCode, body: JSON.parse(data) });
        } catch (e) {
          resolve({ status: res.statusCode, body: data });
        }
      });
    });

    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function logResponse(label, response) {
  console.log(`${label}:`);
  console.log(util.inspect(response.body, { depth: null, colors: true }));
  console.log('');
}

async function runExample() {
  console.log('Running basic usage example for faiss-rest-api...\n');

  try {
    // Create a new index
    let response = await makeRequest('POST', '/indices', {
      name: 'example_index',
      dimension: 4,
      description: 'Flat'
    });
    logResponse('Create index', response);

    // Add vectors to the index
    response = await makeRequest('POST', '/indices/example_index/add', {
      vectors: [
        [0.1, 0.2, 0.3, 0.4],
        [0.9, 0.8, 0.7, 0.6]
      ]
    });
    logResponse('Add vectors', response);

    // Perform a search
    response = await makeRequest('POST', '/indices/example_index/search', {
      query_vectors: [[0.15, 0.25, 0.35, 0.45]],
      k: 5
    });
    logResponse('Search', response);

    console.log('Basic usage example completed successfully.');
  } catch (error) {
    console.error('Example failed:', error.message || error);
  }
}

runExample();
