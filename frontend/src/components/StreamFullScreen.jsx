import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { useWebSocket } from '@/contexts/WebSocketContext'
import { 
  Play,
  Video,
  WifiOff
} from 'lucide-react'
import { format } from 'date-fns'

const StreamFullScreen = () => {
  const { streamId } = useParams()
  const navigate = useNavigate()
  const { isConnected: wsConnected, registerJobUpdateCallback } = useWebSocket()
  const [stream, setStream] = useState(null)
  const [loading, setLoading] = useState(true)
  const [frameData, setFrameData] = useState(null)
  const frameUpdateRef = useRef(null)

  const fetchStream = useCallback(async () => {
    try {
      const response = await fetch('/api/streams')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      const foundStream = data.streams?.find(s => s.id === parseInt(streamId))
      
      if (foundStream) {
        setStream(foundStream)
      } else {
        console.error('Stream not found')
        navigate('/')
      }
    } catch (error) {
      console.error('Failed to fetch stream:', error)
      navigate('/')
    } finally {
      setLoading(false)
    }
  }, [streamId, navigate])

  useEffect(() => {
    fetchStream()
  }, [fetchStream])

  useEffect(() => {
    // Register for stream updates via WebSocket
    const unregister = registerJobUpdateCallback(handleStreamUpdate)
    return unregister
  }, [])

  const handleStreamUpdate = useCallback((jobId, data) => {
    if (data.type === 'stream_frame' && data.stream_id === parseInt(streamId)) {
      setFrameData({
        frame: data.frame,
        timestamp: Date.now()
      })
      
      // Clear previous timeout and set new one
      if (frameUpdateRef.current) {
        clearTimeout(frameUpdateRef.current)
      }
      
      frameUpdateRef.current = setTimeout(() => {
        setFrameData(null)
      }, 5000) // Clear frame if no update for 5 seconds
    }
  }, [streamId])

  const startStream = async () => {
    try {
      const response = await fetch(`/api/streams/${streamId}/start`, {
        method: 'POST'
      })
      const data = await response.json()

      if (data.success) {
        await fetchStream()
      }
    } catch (error) {
      console.error('Failed to start stream:', error)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center text-white">
          <Video className="h-12 w-12 animate-pulse mx-auto mb-4" />
          <p>Loading stream...</p>
        </div>
      </div>
    )
  }

  if (!stream) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="text-center text-white">
          <WifiOff className="h-12 w-12 mx-auto mb-4 text-red-500" />
          <p>Stream not found</p>
        </div>
      </div>
    )
  }

  const isActive = stream.status === 'active'
  const isFrameRecent = frameData && (Date.now() - frameData.timestamp < 10000) // 10 seconds

  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <style>{`
        .fullscreen-container {
          background: #000;
          position: relative;
          overflow: hidden;
        }
        
        .control-overlay {
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          background: linear-gradient(to bottom, rgba(0,0,0,0.8), transparent);
          padding: 1rem;
          z-index: 10;
          transition: opacity 0.3s;
          display: flex;
          justify-content: flex-end;
        }
        
        .stream-image {
          width: 100%;
          height: 100%;
          object-fit: contain;
          background: #000;
        }
        
        .no-signal-fullscreen {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: #9ca3af;
          background: linear-gradient(135deg, #1e293b, #0f172a);
        }
        
        .last-update-info {
          position: absolute;
          top: 20px;
          right: 20px;
          background: rgba(0, 0, 0, 0.8);
          padding: 10px 14px;
          border-radius: 6px;
          font-size: 12px;
          z-index: 10;
          color: #9ca3af;
          text-align: right;
        }
        
        .fullscreen-controls {
          transition: all 0.3s ease;
        }
        
        .fullscreen-controls:hover {
          transform: translateY(-2px);
        }
      `}</style>

      {/* Start Button - only when inactive */}
      {!isActive && (
        <div className="control-overlay">
          <Button 
            onClick={startStream}
            size="sm"
            className="fullscreen-controls"
          >
            <Play className="h-4 w-4 mr-2" />
            Start
          </Button>
        </div>
      )}

      {/* Main Stream Display */}
      <div className="flex-1 relative fullscreen-container">
        {isActive && frameData && isFrameRecent ? (
          <>
            <img
              src={`data:image/jpeg;base64,${frameData.frame}`}
              alt="Live stream"
              className="stream-image"
            />
            
            {/* Camera Info & Last Update - top right */}
            <div className="last-update-info">
              <div className="font-medium text-white mb-1">{stream.name}</div>
              <div>Last update: {format(new Date(frameData.timestamp), 'HH:mm:ss')}</div>
            </div>
          </>
        ) : (
          <div className="no-signal-fullscreen">
            {isActive ? (
              <>
                <Video className="h-16 w-16 mb-4 animate-pulse" />
                <h2 className="text-2xl font-bold mb-2">Connecting...</h2>
                <p className="text-gray-400">Waiting for video signal</p>
              </>
            ) : (
              <>
                <WifiOff className="h-16 w-16 mb-4" />
                <h2 className="text-2xl font-bold mb-2">Stream Inactive</h2>
                <p className="text-gray-400">No video signal</p>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default StreamFullScreen