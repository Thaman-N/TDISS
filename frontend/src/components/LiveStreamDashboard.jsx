import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useWebSocket } from '@/contexts/WebSocketContext'
import { 
  Video,
  Plus, 
  Play, 
  Square, 
  MoreVertical,
  RefreshCw,
  Trash2,
  ExternalLink,
  Grid3X3,
  Grid2X2,
  Maximize2,
  Settings,
  Wifi,
  WifiOff,
  AlertTriangle,
  CheckCircle,
  Clock,
  Eye,
  Camera,
  Monitor,
  PlayCircle,
  PauseCircle,
  History,
  Calendar,
  Download,
  Filter,
  Search,
  Shield,
  Activity,
  TrendingUp,
  BarChart3,
  FileVideo,
  ChevronDown,
  ChevronRight,
  Copy,
  Brain
} from 'lucide-react'

// Helper function for relative time
const getRelativeTime = (date) => {
  const now = new Date()
  const diffMs = now - new Date(date)
  const diffSecs = Math.floor(diffMs / 1000)
  const diffMins = Math.floor(diffSecs / 60)
  const diffHours = Math.floor(diffMins / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffSecs < 60) return 'just now'
  if (diffMins < 60) return `${diffMins}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  return `${diffDays}d ago`
}

// Helper function for formatting dates
const formatDate = (date, options = {}) => {
  return new Date(date).toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    ...options
  })
}

const LiveStreamDashboard = () => {
  const navigate = useNavigate()
  const { isConnected: wsConnected, registerJobUpdateCallback } = useWebSocket()
  const [activeTab, setActiveTab] = useState('streams')
  const [streams, setStreams] = useState([])
  const [events, setEvents] = useState([])
  const [loading, setLoading] = useState(true)
  const [eventsLoading, setEventsLoading] = useState(false)
  const [addDialogOpen, setAddDialogOpen] = useState(false)
  const [newStreamName, setNewStreamName] = useState('')
  const [newStreamUrl, setNewStreamUrl] = useState('')
  const [gridLayout, setGridLayout] = useState('2x2')
  const [activeStreams, setActiveStreams] = useState(new Set())
  const [streamFrames, setStreamFrames] = useState({})
  const [eventFilter, setEventFilter] = useState({
    search: '',
    dateRange: '24h',
    streamId: 'all',
    minConfidence: 0.7
  })
  const [expandedEvents, setExpandedEvents] = useState(new Set())
  const frameUpdateRefs = useRef({})

  const fetchStreams = useCallback(async () => {
    try {
      const response = await fetch('/api/streams')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setStreams(data.streams || [])
    } catch (error) {
      console.error('Failed to fetch streams:', error)
      toast.error('Failed to load streams', {
        description: error.message
      })
    }
  }, [])

  const fetchStreamEvents = useCallback(async () => {
    setEventsLoading(true)
    try {
      const params = new URLSearchParams()
      
      // Apply filters
      if (eventFilter.dateRange !== 'all') {
        const now = new Date()
        let startDate
        switch (eventFilter.dateRange) {
          case '1h':
            startDate = new Date(now.getTime() - 60 * 60 * 1000)
            break
          case '24h':
            startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000)
            break
          case '7d':
            startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
            break
          case '30d':
            startDate = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
            break
          default:
            startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000)
        }
        params.append('start_date', startDate.toISOString())
      }
      
      params.append('limit', '100')
      
      const response = await fetch(`/api/stream-events?${params}`)
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      
      let filteredEvents = data.events || []
      
      // Apply client-side filters
      if (eventFilter.streamId !== 'all') {
        filteredEvents = filteredEvents.filter(event => 
          event.source_id === eventFilter.streamId
        )
      }
      
      if (eventFilter.search) {
        const searchLower = eventFilter.search.toLowerCase()
        filteredEvents = filteredEvents.filter(event =>
          event.filename?.toLowerCase().includes(searchLower) ||
          event.source_id?.toLowerCase().includes(searchLower)
        )
      }
      
      if (eventFilter.minConfidence > 0) {
        filteredEvents = filteredEvents.filter(event =>
          event.confidence >= eventFilter.minConfidence
        )
      }
      
      setEvents(filteredEvents)
    } catch (error) {
      console.error('Failed to fetch stream events:', error)
      toast.error('Failed to load events', {
        description: error.message
      })
    } finally {
      setEventsLoading(false)
    }
  }, [eventFilter])

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchStreams(), fetchStreamEvents()])
      setLoading(false)
    }
    loadData()
  }, [fetchStreams, fetchStreamEvents])

  useEffect(() => {
    // Register for stream updates via WebSocket
    const unregister = registerJobUpdateCallback(handleStreamUpdate)
    return unregister
  }, [])

  // Refresh events when filters change
  useEffect(() => {
    if (!loading) {
      fetchStreamEvents()
    }
  }, [eventFilter, fetchStreamEvents, loading])

  const handleStreamUpdate = useCallback((jobId, data) => {
    if (data.type === 'stream_frame' && data.stream_id) {
      const streamId = data.stream_id
      
      setStreamFrames(prev => ({
        ...prev,
        [streamId]: {
          frame: data.frame,
          timestamp: Date.now()
        }
      }))
      
      if (frameUpdateRefs.current[streamId]) {
        clearTimeout(frameUpdateRefs.current[streamId])
      }
      
      frameUpdateRefs.current[streamId] = setTimeout(() => {
        setStreamFrames(prev => {
          const updated = { ...prev }
          delete updated[streamId]
          return updated
        })
      }, 5000)
    }
    
    // Fix: Only refresh events if user is NOT actively viewing events tab
    // or if it's been more than 10 seconds since last refresh
    if (data.type === 'violence_detected') {
      const now = Date.now()
      const lastRefresh = localStorage.getItem('lastEventRefresh') || 0
      const shouldRefresh = (activeTab !== 'events') || (now - parseInt(lastRefresh) > 10000)
      
      if (shouldRefresh) {
        setTimeout(() => {
          fetchStreamEvents()
          localStorage.setItem('lastEventRefresh', now.toString())
        }, 1000) // Small delay to avoid spam
      }
      
      // Determine button action based on incident status
      const isOngoingIncident = data.is_ongoing_incident || data.incident_status === 'active'
      const buttonLabel = isOngoingIncident ? 'View Stream' : 'View Event'
      const buttonAction = () => {
        if (isOngoingIncident) {
          // Navigate to live stream fullscreen for ongoing incidents
          navigate(`/stream-fullscreen/${data.stream_id}`)
        } else {
          // Switch to events tab for completed incidents
          setActiveTab('events')
          setTimeout(() => {
            fetchStreamEvents()
          }, 200)
        }
      }
      
      toast.warning('Violence detected!', {
        description: `Stream: ${data.stream_name}${isOngoingIncident ? ' (Ongoing)' : ''}`,
        action: {
          label: buttonLabel,
          onClick: buttonAction
        }
      })
    }
    
    // Handle incident finalization notifications
    if (data.type === 'incident_finalized') {
      toast.success('Incident Analysis Complete', {
        description: `${data.stream_name}: ${data.detection_count} detections, ${data.total_duration.toFixed(1)}s duration`,
        action: {
          label: 'View Analysis',
          onClick: () => {
            // Navigate to the first event in the incident for analysis
            if (data.event_ids && data.event_ids.length > 0) {
              navigate(`/results/stream-event-${data.event_ids[0]}`)
            }
          }
        }
      })
      
      // Refresh events list
      if (activeTab === 'events') {
        fetchStreamEvents()
      }
    }
  }, [fetchStreamEvents, activeTab])

  const addStream = async () => {
    if (!newStreamName.trim() || !newStreamUrl.trim()) {
      toast.error('Please fill in all fields')
      return
    }

    try {
      const response = await fetch('/api/streams', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newStreamName.trim(),
          rtsp_url: newStreamUrl.trim()
        })
      })

      const data = await response.json()

      if (data.success) {
        toast.success('Stream added successfully!')
        setNewStreamName('')
        setNewStreamUrl('')
        setAddDialogOpen(false)
        await fetchStreams()
      } else {
        toast.error('Failed to add stream', {
          description: data.message
        })
      }
    } catch (error) {
      toast.error('Failed to add stream', {
        description: error.message
      })
    }
  }

  const startStream = async (streamId) => {
    try {
      const response = await fetch(`/api/streams/${streamId}/start`, {
        method: 'POST'
      })
      const data = await response.json()

      if (data.success) {
        setActiveStreams(prev => new Set([...prev, streamId]))
        toast.success('Stream started')
        await fetchStreams()
      } else {
        toast.error('Failed to start stream', {
          description: data.message
        })
      }
    } catch (error) {
      toast.error('Failed to start stream', {
        description: error.message
      })
    }
  }

  const stopStream = async (streamId) => {
    try {
      const response = await fetch(`/api/streams/${streamId}/stop`, {
        method: 'POST'
      })
      const data = await response.json()

      if (data.success) {
        setActiveStreams(prev => {
          const updated = new Set(prev)
          updated.delete(streamId)
          return updated
        })
        
        // Clear frame data
        setStreamFrames(prev => {
          const updated = { ...prev }
          delete updated[streamId]
          return updated
        })
        
        toast.success('Stream stopped')
        await fetchStreams()
      } else {
        toast.error('Failed to stop stream')
      }
    } catch (error) {
      toast.error('Failed to stop stream')
    }
  }

  const deleteStream = async (streamId) => {
    try {
      const response = await fetch(`/api/streams/${streamId}`, {
        method: 'DELETE'
      })
      const data = await response.json()

      if (data.success) {
        setActiveStreams(prev => {
          const updated = new Set(prev)
          updated.delete(streamId)
          return updated
        })
        toast.success('Stream deleted')
        await fetchStreams()
      } else {
        toast.error('Failed to delete stream')
      }
    } catch (error) {
      toast.error('Failed to delete stream')
    }
  }

  const openFullScreen = (stream) => {
    const fullScreenUrl = `/stream-fullscreen/${stream.id}`
    window.open(fullScreenUrl, '_blank', 'width=1280,height=720,scrollbars=no,resizable=yes')
  }

  // Direct navigation to results - no more dialog
  const viewEventAnalysis = (event) => {
    navigate(`/results/stream-event-${event.id}`)
  }

  const downloadEventClip = async (event) => {
    if (!event.clip_path) {
      toast.error('No clip available for this event')
      return
    }
    
    try {
      const response = await fetch(event.clip_path)
      if (!response.ok) throw new Error('Failed to download clip')
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `violence_event_${event.id}.mp4`
      a.click()
      window.URL.revokeObjectURL(url)
      
      toast.success('Clip downloaded successfully')
    } catch (error) {
      toast.error('Failed to download clip', {
        description: error.message
      })
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <CheckCircle className="h-4 w-4 text-green-500 status-icon" />
      case 'connecting':
        return <Clock className="h-4 w-4 text-yellow-500 status-icon animate-pulse" />
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500 status-icon" />
      default:
        return <WifiOff className="h-4 w-4 text-gray-500 status-icon" />
    }
  }

  const getStatusBadge = (status) => {
    const variants = {
      active: 'default',
      connecting: 'outline',
      error: 'destructive',
      inactive: 'secondary'
    }
    return (
      <Badge variant={variants[status] || 'secondary'} className="capitalize status-badge">
        {status}
      </Badge>
    )
  }

  const getGridClasses = () => {
    switch (gridLayout) {
      case '1x1':
        return 'grid-cols-1'
      case '2x2':
        return 'grid-cols-1 md:grid-cols-2'
      case '3x3':
        return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'
      default:
        return 'grid-cols-1 md:grid-cols-2'
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.9) return 'text-red-600'
    if (confidence >= 0.8) return 'text-orange-600'
    if (confidence >= 0.7) return 'text-yellow-600'
    return 'text-gray-600'
  }

  const toggleEventExpansion = (eventId) => {
    setExpandedEvents(prev => {
      const newSet = new Set(prev)
      if (newSet.has(eventId)) {
        newSet.delete(eventId)
      } else {
        newSet.add(eventId)
      }
      return newSet
    })
  }

  // Calculate stats
  const totalEvents = events.length
  const last24hEvents = events.filter(event => {
    const eventTime = new Date(event.timestamp)
    const now = new Date()
    return (now - eventTime) < 24 * 60 * 60 * 1000
  }).length
  const avgConfidence = events.length > 0 
    ? events.reduce((sum, event) => sum + event.confidence, 0) / events.length 
    : 0
  const activeStreamCount = streams.filter(s => s.status === 'active').length

  const StreamCard = ({ stream }) => {
    const isActive = stream.status === 'active'
    const frameData = streamFrames[stream.id]
    const streamEvents = events.filter(event => event.source_id === stream.id.toString())

    return (
      <Card className="stream-card group relative overflow-hidden">
        <style>{`
          .stream-card {
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            cursor: pointer;
            position: relative;
            overflow: hidden;
          }
          .stream-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 2px;
            background: linear-gradient(90deg, transparent, hsl(var(--primary)), transparent);
            transition: left 0.6s cubic-bezier(0.4, 0, 0.2, 1);
          }
          .stream-card:hover::before {
            left: 100%;
          }
          .stream-card:hover {
            transform: translateY(-4px) scale(1.02);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
            border-color: hsl(var(--primary) / 0.2);
          }
          .stream-card:hover .stream-title {
            color: hsl(var(--primary));
            transform: translateX(2px);
          }
          .stream-card:hover .stream-video {
            transform: scale(1.05);
          }
          .stream-card:hover .shimmer-button {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          }
          .stream-card:hover .status-icon {
            transform: scale(1.2) rotate(5deg);
          }
          
          .stream-title {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }
          .stream-video {
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          }
          .shimmer-button {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }
          .status-icon {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }
          .status-badge {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          }
          
          /* Shimmer effect for stop and fullscreen buttons */
          .shimmer-button {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
          }
          .shimmer-button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
          }
          .shimmer-button:hover::before {
            left: 100%;
          }
          .shimmer-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
          }
          
          .stream-video-container {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-radius: 0.5rem;
            overflow: hidden;
            position: relative;
          }
          
          .no-signal {
            background: linear-gradient(135deg, #374151, #1f2937);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            color: #9ca3af;
            min-height: 200px;
          }
          
          .live-indicator {
            position: absolute;
            top: 8px;
            left: 8px;
            background: rgba(239, 68, 68, 0.9);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
            z-index: 10;
            animation: pulse 2s infinite;
          }
          
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
          }
        `}</style>

        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {getStatusIcon(stream.status)}
              <CardTitle className="text-lg stream-title">{stream.name}</CardTitle>
            </div>
            <div className="flex items-center space-x-2">
              {getStatusBadge(stream.status)}
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="ghost" size="sm" className="dropdown-trigger">
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/95">
                  <DropdownMenuItem onClick={() => openFullScreen(stream)} className="dropdown-item">
                    <ExternalLink className="h-4 w-4 mr-2" />
                    Full Screen
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => fetchStreams()} className="dropdown-item">
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </DropdownMenuItem>
                  <DropdownMenuItem 
                    onClick={() => deleteStream(stream.id)} 
                    className="text-red-600 dropdown-item"
                  >
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>
          <CardDescription className="flex items-center space-x-4">
            <span className="flex items-center">
              <Video className="h-3 w-3 mr-1" />
              {stream.rtsp_url.length > 30 ? `${stream.rtsp_url.substring(0, 30)}...` : stream.rtsp_url}
            </span>
            {streamEvents.length > 0 && (
              <span className="flex items-center">
                <AlertTriangle className="h-3 w-3 mr-1" />
                {streamEvents.length} events
              </span>
            )}
          </CardDescription>
        </CardHeader>

        <CardContent>
          <div className="stream-video-container mb-4">
            {isActive ? (
              <div className="relative">
                <div className="live-indicator">LIVE</div>
                {frameData && frameData.frame ? (
                  <img
                    src={`data:image/jpeg;base64,${frameData.frame}`}
                    alt="Live stream"
                    className="w-full h-48 object-cover stream-video"
                  />
                ) : (
                  <div className="no-signal">
                    <Video className="h-8 w-8 mb-2 animate-pulse" />
                    <span>Connecting...</span>
                  </div>
                )}
              </div>
            ) : (
              <div className="no-signal">
                <WifiOff className="h-8 w-8 mb-2" />
                <span>No Signal</span>
              </div>
            )}
          </div>

          <div className="flex items-center justify-between space-x-2">
            <div className="flex space-x-2">
              {!isActive ? (
                <Button 
                  onClick={() => startStream(stream.id)} 
                  size="sm" 
                  className="shimmer-button"
                >
                  <Play className="h-4 w-4 mr-1" />
                  Start
                </Button>
              ) : (
                <Button 
                  onClick={() => stopStream(stream.id)} 
                  variant="destructive" 
                  size="sm" 
                  className="shimmer-button"
                >
                  <Square className="h-4 w-4 mr-1" />
                  Stop
                </Button>
              )}
              
              <Button 
                onClick={() => openFullScreen(stream)} 
                variant="outline" 
                size="sm" 
                className="shimmer-button"
              >
                <Maximize2 className="h-4 w-4" />
              </Button>
            </div>

            {stream.last_detection && (
              <div className="text-xs text-muted-foreground">
                Last alert: {new Date(stream.last_detection).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    )
  }

  const EventCard = ({ event }) => {
    const isExpanded = expandedEvents.has(event.id)
    const stream = streams.find(s => s.id.toString() === event.source_id)
    
    return (
      <Card className="event-card cursor-pointer">
        <CardContent className="p-6">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center space-x-3 mb-3">
                <Badge variant="destructive" className="text-xs">
                  Violence Detected
                </Badge>
                <Badge variant="outline" className={`text-xs ${getConfidenceColor(event.confidence)}`}>
                  {(event.confidence * 100).toFixed(1)}%
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {getRelativeTime(event.timestamp)}
                </span>
              </div>
              
              <div className="flex items-center space-x-2 mb-3">
                <Camera className="h-4 w-4 text-muted-foreground" />
                <span className="font-medium">{stream?.name || event.filename}</span>
              </div>
              
              <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                <span className="flex items-center">
                  <Clock className="h-3 w-3 mr-1" />
                  {formatDate(event.timestamp, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
                <span className="flex items-center">
                  <Activity className="h-3 w-3 mr-1" />
                  {event.duration?.toFixed(1)}s
                </span>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 ml-4">
              {/* Direct navigation to results - no dialog */}
              <Button
                variant="default"
                size="sm"
                onClick={() => viewEventAnalysis(event)}
              >
                <Eye className="h-4 w-4 mr-1" />
                View Analysis
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={() => toggleEventExpansion(event.id)}
              >
                {isExpanded ? (
                  <ChevronDown className="h-4 w-4" />
                ) : (
                  <ChevronRight className="h-4 w-4" />
                )}
              </Button>
            </div>
          </div>
          
          {isExpanded && (
            <div className="mt-6 pt-6 border-t space-y-4">
              {event.thumbnail_path && (
                <div className="flex justify-center">
                  <img 
                    src={event.thumbnail_path} 
                    alt="Event thumbnail"
                    className="w-32 h-24 object-cover rounded border cursor-pointer hover:opacity-80 transition-opacity"
                    onClick={() => viewEventAnalysis(event)}
                  />
                </div>
              )}
              
              <div className="grid grid-cols-2 gap-6 text-sm">
                <div>
                  <span className="text-muted-foreground">Source:</span>
                  <div className="font-medium">{event.source_type}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Stream ID:</span>
                  <div className="font-medium">{event.source_id}</div>
                </div>
                <div>
                  <span className="text-muted-foreground">Start Time:</span>
                  <div className="font-medium">{event.start_time?.toFixed(1)}s</div>
                </div>
                <div>
                  <span className="text-muted-foreground">End Time:</span>
                  <div className="font-medium">{event.end_time?.toFixed(1)}s</div>
                </div>
              </div>
              
              <div className="flex justify-center space-x-2 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => viewEventAnalysis(event)}
                >
                  <Eye className="h-4 w-4 mr-2" />
                  Full Analysis
                </Button>
                
                {event.clip_path && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => downloadEventClip(event)}
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Download Clip
                  </Button>
                )}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    )
  }

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <Video className="h-8 w-8 animate-pulse mx-auto mb-4" />
        <p>Loading streams...</p>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <style>{`
        /* Enhanced Dashboard Animations */
        .layout-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .layout-button::before {
          content: '';
          position: absolute;
          top: 50%;
          left: 50%;
          width: 0;
          height: 0;
          background: hsl(var(--primary) / 0.1);
          border-radius: 50%;
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          transform: translate(-50%, -50%);
        }
        .layout-button:hover::before {
          width: 120%;
          height: 120%;
        }
        .layout-button:hover {
          transform: translateY(-2px) scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        .layout-button.active {
          background: hsl(var(--primary));
          color: hsl(var(--primary-foreground));
        }
        
        .add-stream-button {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .add-stream-button:hover {
          transform: translateY(-3px) scale(1.05);
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }
        
        .connection-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .connection-badge:hover {
          transform: scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .tab-trigger {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .tab-trigger:hover {
          transform: translateY(-1px);
        }
        
        .dropdown-trigger {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .dropdown-trigger:hover {
          transform: scale(1.1) rotate(90deg);
          background: hsl(var(--primary) / 0.1);
        }
        
        .dropdown-item {
          transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .dropdown-item:hover {
          transform: translateX(4px);
          background: hsl(var(--primary) / 0.05);
        }
        
        .empty-state-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .empty-state-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        }

        /* Stats Cards Effect from ProcessingDashboard */
        .stat-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
          position: relative;
          overflow: hidden;
        }
        .stat-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 0;
          background: linear-gradient(135deg, hsl(var(--primary) / 0.05), transparent);
          transition: height 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stat-card:hover::before {
          height: 100%;
        }
        .stat-card:hover {
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        
        .stat-value {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Event Cards */
        .event-card {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .event-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
          border-color: hsl(var(--primary) / 0.2);
        }
      `}</style>
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">Live Stream Security</h1>
          <p className="text-muted-foreground">
            Monitor multiple RTSP camera feeds with real-time violence detection and event playback
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <Dialog open={addDialogOpen} onOpenChange={setAddDialogOpen}>
            <DialogTrigger asChild>
              <Button className="add-stream-button">
                <Plus className="h-4 w-4 mr-2" />
                Add Stream
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/95">
              <DialogHeader>
                <DialogTitle>Add RTSP Stream</DialogTitle>
                <DialogDescription>
                  Add a new camera stream to monitor for violence detection
                </DialogDescription>
              </DialogHeader>
              <div className="grid gap-4 py-4">
                <div className="grid gap-2">
                  <Label htmlFor="name">Stream Name</Label>
                  <Input
                    id="name"
                    value={newStreamName}
                    onChange={(e) => setNewStreamName(e.target.value)}
                    placeholder="e.g., Front Door Camera"
                  />
                </div>
                <div className="grid gap-2">
                  <Label htmlFor="url">RTSP URL</Label>
                  <Input
                    id="url"
                    value={newStreamUrl}
                    onChange={(e) => setNewStreamUrl(e.target.value)}
                    placeholder="rtsp://username:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"
                  />
                </div>
              </div>
              <DialogFooter>
                <Button variant="outline" onClick={() => setAddDialogOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={addStream}>Add Stream</Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Stats Dashboard */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center">
              <Camera className="h-4 w-4 mr-2" />
              Active Streams
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">{activeStreamCount}</div>
            <div className="text-xs text-muted-foreground">of {streams.length} total</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center">
              <Shield className="h-4 w-4 mr-2" />
              Events (24h)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">{last24hEvents}</div>
            <div className="text-xs text-muted-foreground">of {totalEvents} total</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center">
              <TrendingUp className="h-4 w-4 mr-2" />
              Avg Confidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">
              {avgConfidence > 0 ? `${(avgConfidence * 100).toFixed(1)}%` : '0%'}
            </div>
            <div className="text-xs text-muted-foreground">detection accuracy</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium flex items-center">
              {wsConnected ? (
                <Wifi className="h-4 w-4 mr-2" />
              ) : (
                <WifiOff className="h-4 w-4 mr-2" />
              )}
              Connection Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold stat-value ${
              wsConnected ? 'text-green-600' : 'text-red-600'
            }`}>
              {wsConnected ? 'Connected' : 'Disconnected'}
            </div>
            <div className="text-xs text-muted-foreground">
              {wsConnected ? 'live updates active' : 'no live updates'}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">        
        <div className="flex items-center justify-between">
          <TabsList>
            <TabsTrigger value="streams">Live Streams</TabsTrigger>
            <TabsTrigger value="events" data-value="events">Security Events</TabsTrigger>
          </TabsList>
          
          {/* Layout Controls for Grid View */}
          <div className="flex items-center space-x-2">
            <span className="text-sm text-muted-foreground mr-2">Layout:</span>
            {[
              { id: '1x1', icon: Monitor, label: '1x1' },
              { id: '2x2', icon: Grid2X2, label: '2x2' },
              { id: '3x3', icon: Grid3X3, label: '3x3' }
            ].map(layout => (
              <Button
                key={layout.id}
                variant={gridLayout === layout.id ? 'default' : 'outline'}
                size="sm"
                onClick={() => setGridLayout(layout.id)}
                className="layout-button"
              >
                <layout.icon className="h-4 w-4" />
              </Button>
            ))}
          </div>
        </div>

        <TabsContent value="streams" className="space-y-6">
          {streams.length === 0 ? (
            <Card className="empty-state-card">
              <CardContent className="pt-6 text-center">
                <Camera className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg font-medium mb-2">No streams configured</p>
                <p className="text-muted-foreground mb-4">
                  Add your first RTSP camera stream to get started
                </p>
                <Button onClick={() => setAddDialogOpen(true)} className="add-stream-button">
                  <Plus className="h-4 w-4 mr-2" />
                  Add Stream
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className={`grid gap-6 ${getGridClasses()}`}>
              {streams.map(stream => (
                <StreamCard key={stream.id} stream={stream} />
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="events" className="space-y-6">
          {/* Event Filters */}
          <Card>
            <CardHeader className="pb-4">
              <CardTitle className="flex items-center">
                <Filter className="h-5 w-5 mr-2" />
                Event Filters
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="search">Search</Label>
                  <Input
                    id="search"
                    placeholder="Search events..."
                    value={eventFilter.search}
                    onChange={(e) => setEventFilter(prev => ({ ...prev, search: e.target.value }))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="dateRange">Time Range</Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button 
                        variant="outline" 
                        className="w-full justify-between"
                      >
                        {eventFilter.dateRange === '1h' ? 'Last Hour' : 
                         eventFilter.dateRange === '24h' ? 'Last 24 Hours' : 
                         eventFilter.dateRange === '7d' ? 'Last 7 Days' : 
                         eventFilter.dateRange === '30d' ? 'Last 30 Days' : 'All Time'}
                        <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="center" className="bg-background/99 backdrop-blur supports-[backdrop-filter]:bg-background/99 w-full">
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, dateRange: '1h' }))}>
                        Last Hour
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, dateRange: '24h' }))}>
                        Last 24 Hours
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, dateRange: '7d' }))}>
                        Last 7 Days
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, dateRange: '30d' }))}>
                        Last 30 Days
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, dateRange: 'all' }))}>
                        All Time
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="streamFilter">Stream</Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button 
                        variant="outline" 
                        className="w-full justify-between"
                      >
                        {eventFilter.streamId === 'all' 
                          ? 'All Streams' 
                          : streams.find(s => s.id.toString() === eventFilter.streamId)?.name || 'Select Stream'}
                        <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="center" className="bg-background/99 backdrop-blur supports-[backdrop-filter]:bg-background/99 w-full">
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, streamId: 'all' }))}>
                        All Streams
                      </DropdownMenuItem>
                      {streams.map(stream => (
                        <DropdownMenuItem 
                          key={stream.id} 
                          onClick={() => setEventFilter(prev => ({ ...prev, streamId: stream.id.toString() }))}
                        >
                          {stream.name}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="confidence">Min Confidence</Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button 
                        variant="outline" 
                        className="w-full justify-between"
                      >
                        {eventFilter.minConfidence === 0 ? 'Any' : 
                         eventFilter.minConfidence === 0.7 ? '70%+' : 
                         eventFilter.minConfidence === 0.8 ? '80%+' : 
                         eventFilter.minConfidence === 0.9 ? '90%+' : `${eventFilter.minConfidence * 100}%+`}
                        <ChevronDown className="h-4 w-4 ml-2 opacity-50" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="center" className="bg-background/99 backdrop-blur supports-[backdrop-filter]:bg-background/99 w-full">
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, minConfidence: 0 }))}>
                        Any
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, minConfidence: 0.7 }))}>
                        70%+
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, minConfidence: 0.8 }))}>
                        80%+
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setEventFilter(prev => ({ ...prev, minConfidence: 0.9 }))}>
                        90%+
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Events List */}
          {eventsLoading ? (
            <Card className="empty-state-card">
              <CardContent className="pt-6 text-center">
                <History className="h-8 w-8 animate-pulse mx-auto mb-4" />
                <p>Loading events...</p>
              </CardContent>
            </Card>
          ) : events.length === 0 ? (
            <Card className="empty-state-card">
              <CardContent className="pt-6 text-center">
                <Shield className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg font-medium mb-2">No security events found</p>
                <p className="text-muted-foreground">
                  Violence detection events will appear here when detected
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-6">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium">
                  {events.length} event{events.length !== 1 ? 's' : ''} found
                </h3>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={fetchStreamEvents}
                  disabled={eventsLoading}
                >
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </Button>
              </div>
              
              <div className="space-y-4">
                {events.map(event => (
                  <EventCard key={event.id} event={event} />
                ))}
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default LiveStreamDashboard