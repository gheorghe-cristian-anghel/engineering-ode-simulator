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

The 2D Heat page calls `POST /api/pde/heat-2d`. Its Canvas heatmap uses `ImageData`, which is lightweight for the API's bounded regular grids and avoids adding a large charting dependency.
