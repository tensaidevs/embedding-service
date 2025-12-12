// embedding-service/server.js
// Local embedding microservice using Transformers.js
// Zero-cost alternative to OpenAI embeddings

import express from "express";
import { pipeline, env } from "@xenova/transformers";
import cors from "cors";

// Disable local model caching in production (use /tmp for serverless)
env.cacheDir = process.env.CACHE_DIR || "/tmp/transformers-cache";

const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(express.json({ limit: "10mb" }));

// Global embedding pipeline (loaded once at startup)
let embedder = null;
const MODEL_NAME = "Xenova/all-MiniLM-L6-v2"; // 384 dimensions, fast & accurate

/**
 * Initialize the embedding model
 */
async function initializeModel() {
  console.log("[Embedding Service] Loading model:", MODEL_NAME);
  const startTime = Date.now();

  try {
    embedder = await pipeline("feature-extraction", MODEL_NAME);
    const loadTime = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`[Embedding Service] ✅ Model loaded in ${loadTime}s`);
    return true;
  } catch (error) {
    console.error("[Embedding Service] ❌ Failed to load model:", error);
    return false;
  }
}

/**
 * Generate embedding for text
 */
async function generateEmbedding(text) {
  if (!embedder) {
    throw new Error("Model not initialized");
  }

  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data);
}

/**
 * Health check endpoint
 */
app.get("/health", (req, res) => {
  const status = embedder ? "ready" : "loading";
  res.json({
    status,
    model: MODEL_NAME,
    timestamp: new Date().toISOString(),
  });
});

/**
 * Embed single text
 */
app.post("/embed", async (req, res) => {
  try {
    const { text } = req.body;

    if (!text || typeof text !== "string") {
      return res
        .status(400)
        .json({ error: "Text is required and must be a string" });
    }

    if (!embedder) {
      return res
        .status(503)
        .json({ error: "Model is still loading, please try again" });
    }

    const startTime = Date.now();
    const embedding = await generateEmbedding(text);
    const duration = Date.now() - startTime;

    res.json({
      embedding,
      dimensions: embedding.length,
      duration_ms: duration,
      model: MODEL_NAME,
    });
  } catch (error) {
    console.error("[Embed] Error:", error);
    res
      .status(500)
      .json({ error: "Failed to generate embedding", details: error.message });
  }
});

/**
 * Embed batch of texts
 */
app.post("/embed/batch", async (req, res) => {
  try {
    const { texts } = req.body;

    if (!Array.isArray(texts) || texts.length === 0) {
      return res.status(400).json({ error: "texts must be a non-empty array" });
    }

    if (!embedder) {
      return res
        .status(503)
        .json({ error: "Model is still loading, please try again" });
    }

    const startTime = Date.now();
    const embeddings = await Promise.all(
      texts.map((text) => generateEmbedding(text))
    );
    const duration = Date.now() - startTime;

    res.json({
      embeddings,
      count: embeddings.length,
      dimensions: embeddings[0].length,
      duration_ms: duration,
      model: MODEL_NAME,
    });
  } catch (error) {
    console.error("[Embed Batch] Error:", error);
    res
      .status(500)
      .json({ error: "Failed to generate embeddings", details: error.message });
  }
});

/**
 * Model info endpoint
 */
app.get("/model", (req, res) => {
  res.json({
    name: MODEL_NAME,
    dimensions: 384,
    description: "Sentence transformer optimized for semantic similarity",
    status: embedder ? "loaded" : "loading",
  });
});

// Start server
async function start() {
  console.log("[Embedding Service] Starting...");

  // Load model first
  const loaded = await initializeModel();

  if (!loaded) {
    console.error("[Embedding Service] ❌ Failed to load model, exiting...");
    process.exit(1);
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Embedding Service] 🚀 Running on http://0.0.0.0:${PORT}`);
    console.log(`[Embedding Service] Model: ${MODEL_NAME} (384 dimensions)`);
    console.log(
      `[Embedding Service] Health check: http://0.0.0.0:${PORT}/health`
    );
  });
}

start().catch((error) => {
  console.error("[Embedding Service] Fatal error:", error);
  process.exit(1);
});
