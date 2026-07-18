import { Route, Routes } from 'react-router-dom'
import HomePage from './pages/HomePage'
import JobPage from './pages/JobPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/jobs/:jobId" element={<JobPage />} />
    </Routes>
  )
}
