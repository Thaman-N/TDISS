import React from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { Toaster } from 'sonner'
// import useTheme from './hooks/useTheme'
// import useDarkMode from './hooks/useDarkMode'
import useTheme from './hooks/useTheme'
import LandingPage from './components/LandingPage'
import UploadInterface from './components/UploadInterface'
import ProcessingDashboard from './components/ProcessingDashboard'
import ResultsViewer from './components/ResultsViewer'
import Navigation from './components/Navigation'
import { WebSocketProvider } from './contexts/WebSocketContext'
import LiveStreamDashboard from './components/LiveStreamDashboard'
import StreamFullScreen from './components/StreamFullScreen'
import MultiVideoGridViewer from './components/MultiVideoGridViewer'
import MasterDashboard from './components/MasterDashboard'

function AppContent() {
  const location = useLocation()
  const { 
    darkMode, 
    toggleDarkMode, 
    currentTheme, 
    setTheme, 
    availableThemes, 
    getThemeName 
  } = useTheme()

  // Hide navigation for fullscreen routes
  const isFullscreenRoute = location.pathname.startsWith('/stream-fullscreen')

  return (
    <div className={`min-h-screen ${darkMode ? 'dark' : ''}`} data-theme={currentTheme}>
      <WebSocketProvider>
        {!isFullscreenRoute && (
          <Navigation 
            darkMode={darkMode}
            toggleDarkMode={toggleDarkMode}
            currentTheme={currentTheme}
            setTheme={setTheme}
            availableThemes={availableThemes}
            getThemeName={getThemeName}
          />
        )}
        
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/upload" element={<UploadInterface />} />
          <Route path="/dashboard" element={<ProcessingDashboard />} />
          <Route path="/master-dashboard" element={<MasterDashboard />} />
          <Route path="/results/:jobId" element={<ResultsViewer />} />
          {/* NEW: Add this line for stream events */}
          <Route path="/results/stream-event-:eventId" element={<ResultsViewer />} />
          <Route path="/multi-analysis" element={<MultiVideoGridViewer />} />
          <Route path="/live-streams" element={<LiveStreamDashboard />} />
          <Route path="/stream-fullscreen/:streamId" element={<StreamFullScreen />} />
          <Route path="/results/incident-:incidentId" element={<ResultsViewer />} />
        </Routes>
        
        <Toaster 
          position="top-right"
          richColors
          expand={true}
          closeButton
        />
      </WebSocketProvider>
    </div>
  )
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  )
}

export default App