import { Link } from 'react-router-dom'
export function NotFoundPage() { return <div className="page"><span className="eyebrow">404</span><h1>Page not found</h1><p>This simulation workspace does not have that route yet.</p><Link className="primary-button" to="/">Return to overview</Link></div> }
