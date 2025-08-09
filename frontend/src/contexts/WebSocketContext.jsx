import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react'

const WebSocketContext = createContext()

export const useWebSocket = () => {
  const context = useContext(WebSocketContext)
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

export const WebSocketProvider = ({ children }) => {
  const wsRef = useRef(null)
  const [isConnected, setIsConnected] = useState(false)
  const jobUpdateCallbacksRef = useRef(new Set())

  const setupWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return // Already connected
    }

    const wsUrl = process.env.NODE_ENV === 'development' 
      ? 'ws://localhost:8000/ws'
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
    
    console.log('Attempting to connect to WebSocket:', wsUrl)
    
    wsRef.current = new WebSocket(wsUrl)
    
    wsRef.current.onopen = () => {
      setIsConnected(true)
      console.log('WebSocket connected successfully')
    }
    
    wsRef.current.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        console.log('WebSocket message received:', message)
        
        if (message.type === 'job_update') {
          // Notify all registered callbacks
          jobUpdateCallbacksRef.current.forEach(callback => {
            callback(message.job_id, message.data)
          })
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }
    
    wsRef.current.onclose = (event) => {
      setIsConnected(false)
      console.log('WebSocket disconnected:', event.code, event.reason)
      
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          console.log('Attempting to reconnect...')
          setupWebSocket()
        }
      }, 3000)
    }
    
    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsConnected(false)
    }
  }, [])

  const registerJobUpdateCallback = useCallback((callback) => {
    jobUpdateCallbacksRef.current.add(callback)
    return () => {
      jobUpdateCallbacksRef.current.delete(callback)
    }
  }, [])

  useEffect(() => {
    setupWebSocket()
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [setupWebSocket])

  const value = {
    isConnected,
    registerJobUpdateCallback,
    reconnect: setupWebSocket
  }

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}