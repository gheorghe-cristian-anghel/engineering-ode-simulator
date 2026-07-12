import { Route, Routes } from 'react-router-dom'
import { AppShell } from './components/AppShell'
import { Heat2DPage } from './features/heat2d/Heat2DPage'
import { HomePage } from './pages/HomePage'
import { NotFoundPage } from './pages/NotFoundPage'

export function App() { return <Routes><Route element={<AppShell />}><Route path="/" element={<HomePage />} /><Route path="/simulations/heat-2d" element={<Heat2DPage />} /><Route path="*" element={<NotFoundPage />} /></Route></Routes> }
