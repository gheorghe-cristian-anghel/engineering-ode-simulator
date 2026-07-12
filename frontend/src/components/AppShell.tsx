import { NavLink, Outlet } from 'react-router-dom'
import { useState } from 'react'

const navigation = [{ to: '/', label: 'Overview', end: true }, { to: '/simulations/heat-2d', label: '2D Heat Equation', end: false }, { to: '/simulations/estimation/kalman', label: 'Kalman State Estimation', end: false }, { to: '/simulations/uav-path', label: 'UAV Path Planning', end: false }]

export function AppShell() {
  const [isOpen, setIsOpen] = useState(false)
  return <div className="app-shell">
    <header className="topbar"><button className="menu-button" aria-label="Toggle navigation" aria-controls="primary-navigation" aria-expanded={isOpen} onClick={() => setIsOpen(!isOpen)}>☰</button><span className="topbar__eyebrow">Simulation workspace</span><span className="topbar__status">Local prototype</span></header>
    <aside id="primary-navigation" className={`sidebar ${isOpen ? 'sidebar--open' : ''}`} aria-label="Primary navigation">
      <div className="brand"><span className="brand__mark">∿</span><div><strong>EST</strong><small>Engineering Simulation Toolkit</small></div></div>
      <nav>{navigation.map(({ to, label, end }) => <NavLink key={to} to={to} end={end} onClick={() => setIsOpen(false)}>{label}</NavLink>)}</nav>
      <p className="sidebar__footer">Educational numerical models. Validate independently before engineering use.</p>
    </aside>
    <main><Outlet /></main>
  </div>
}
