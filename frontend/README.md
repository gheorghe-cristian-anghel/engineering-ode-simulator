# Engineering Simulation Toolkit frontend

The React frontend is a Vite + TypeScript application that complements the existing Streamlit app; it does not replace or modify it.

## Run locally

From the repository root, start the FastAPI backend in one terminal:

```powershell
python -m uvicorn backend.main:app --reload
```

Then start the frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000`, so no FastAPI CORS changes are needed for local development. To use a separately hosted API, set `VITE_API_BASE_URL` to its origin before running Vite.

## Checks

```powershell
npm run test
npm run test:coverage
npm run build
```

The visualization layer uses browser-native Canvas for scalar fields and SVG for line and path plots, avoiding a large charting dependency. The 2D Heat page calls `POST /api/pde/heat-2d`; its Canvas heatmap uses `ImageData` and is intended for bounded regular grids. Keep field payloads at or below roughly 250,000 cells (about a 500 × 500 grid), and time/path payloads at or below roughly 10,000 samples per series. The plotting components decimate rendered line vertices, but payload limits should still be enforced by the API as the application grows.
