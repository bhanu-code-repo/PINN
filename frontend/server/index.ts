import express from "express";
import { createProxyMiddleware } from "http-proxy-middleware";

const app = express();
const FASTAPI_URL = process.env.FASTAPI_URL ?? "http://localhost:8000";
const PORT = Number(process.env.BFF_PORT ?? 3001);

// Proxy all /api/v1/* requests to FastAPI
// Mount at /api/v1 — Express strips the mount prefix, so set target to include it
app.use(
  "/api/v1",
  createProxyMiddleware({
    target: `${FASTAPI_URL}/api/v1`,
    changeOrigin: true,
    cookieDomainRewrite: "",
    on: {
      proxyReq: (_proxyReq, req) => {
        console.log(`[BFF] ${req.method} ${req.url} -> ${FASTAPI_URL}/api/v1`);
      },
    },
  }),
);

app.listen(PORT, () => {
  console.log(`[BFF] Proxy server running on http://localhost:${PORT}`);
  console.log(`[BFF] Forwarding /api/v1/* -> ${FASTAPI_URL}`);
});
