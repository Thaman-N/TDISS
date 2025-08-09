import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
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

function App() {
  const { 
    darkMode, 
    toggleDarkMode, 
    currentTheme, 
    setTheme, 
    availableThemes, 
    getThemeName 
  } = useTheme()

  return (
    <div className={`min-h-screen ${darkMode ? 'dark' : ''}`} data-theme={currentTheme}>
      <Router>
        <WebSocketProvider>
          <Navigation 
            darkMode={darkMode}
            toggleDarkMode={toggleDarkMode}
            currentTheme={currentTheme}
            setTheme={setTheme}
            availableThemes={availableThemes}
            getThemeName={getThemeName}
          />
          
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/upload" element={<UploadInterface />} />
            <Route path="/dashboard" element={<ProcessingDashboard />} />
            <Route path="/results/:jobId" element={<ResultsViewer />} />
          </Routes>
          
          <Toaster 
            position="top-right"
            richColors
            expand={true}
            closeButton
          />
        </WebSocketProvider>
      </Router>
    </div>
  )
}

export default App