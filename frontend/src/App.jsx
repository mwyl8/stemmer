import { Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import JobPage from './pages/JobPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      {/* Not /jobs/:jobId -- see lib/routes.js for why that collides with the backend's own /jobs/{id} API. */}
      <Route path="/results/:jobId" element={<JobPage />} />
    </Routes>
  )
}
