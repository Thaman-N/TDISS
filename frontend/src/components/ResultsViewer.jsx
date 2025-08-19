import React, { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts'
import {
  ArrowLeft,
  Download,
  Share2,
  AlertTriangle,
  CheckCircle,
  Clock,
  Eye,
  Brain,
  Zap,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Volume2,
  Maximize,
  Copy,
  Calendar,
  FileVideo,
  Target,
  TrendingUp,
  Info,
  Upload
} from 'lucide-react'
import { format } from 'date-fns'

const ResultsViewer = () => {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [videoLoading, setVideoLoading] = useState(true)
  const [videoError, setVideoError] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  
  // Use useRef for the video element
  const videoRef = useRef(null)

  useEffect(() => {
    fetchResult()
  }, [jobId])

  const fetchResult = async () => {
    try {
      let response;
      
      // Check if this is a stream event or regular job
      if (jobId.startsWith('stream-event-')) {
        const eventId = jobId.replace('stream-event-', '')
        response = await fetch(`/api/stream-event/${eventId}`)
      } else {
        response = await fetch(`/api/result/${jobId}`)
      }
      
      if (!response.ok) {
        throw new Error('Result not found')
      }
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Failed to fetch result:', error)
      toast.error('Failed to load results', {
        description: error.message
      })
      // Navigate back to appropriate dashboard
      if (jobId.startsWith('stream-event-')) {
        navigate('/live-streams')
      } else {
        navigate('/dashboard')
      }
    } finally {
      setLoading(false)
    }
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  // Enhanced jump to time function for HTML5 video
  const jumpToTime = (time) => {
    if (videoRef.current && !videoError) {
      try {
        videoRef.current.currentTime = time
        setCurrentTime(time)
        if (videoRef.current.paused) {
          videoRef.current.play()
          setPlaying(true)
        }
        toast.success(`Jumped to ${formatTime(time)}`)
      } catch (error) {
        console.error('Error seeking to time:', error)
        toast.error('Failed to jump to segment')
      }
    } else {
      toast.warning('Video not available for seeking')
    }
  }

  // Enhanced clipboard function
  const copyToClipboard = async (text) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text)
        toast.success('Copied to clipboard')
      } else {
        // Fallback for older browsers or non-HTTPS
        const textArea = document.createElement('textarea')
        textArea.value = text
        document.body.appendChild(textArea)
        textArea.focus()
        textArea.select()
        try {
          document.execCommand('copy')
          toast.success('Copied to clipboard')
        } catch (err) {
          toast.error('Failed to copy to clipboard')
        }
        document.body.removeChild(textArea)
      }
    } catch (err) {
      console.error('Failed to copy: ', err)
      toast.error('Failed to copy to clipboard')
    }
  }

  // Enhanced download function
  const downloadReport = () => {
    try {
      const reportData = {
        jobId: result.job_id,
        filename: result.filename,
        timestamp: result.timestamp,
        analysis: {
          hasViolence: result.has_violence,
          confidence: result.overall_result.confidence,
          segments: result.segments,
          violenceDuration: result.violence_duration,
          violencePercentage: result.violence_percentage
        },
        metadata: result.metadata,
        modelInfo: result.model_info
      }
      
      const blob = new Blob([JSON.stringify(reportData, null, 2)], { 
        type: 'application/json' 
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `violence-analysis-${result.job_id}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      toast.success('Report downloaded successfully')
    } catch (error) {
      console.error('Download error:', error)
      toast.error('Failed to download report')
    }
  }

  // Enhanced share function
  const shareResults = async () => {
    try {
      const shareData = {
        title: `Violence Analysis: ${result.filename}`,
        text: `Analysis Results: ${result.has_violence ? 'Violence Detected' : 'No Violence Detected'} (${(result.overall_result.confidence * 100).toFixed(1)}% confidence)`,
        url: window.location.href
      }

      if (navigator.share && window.isSecureContext) {
        await navigator.share(shareData)
        toast.success('Shared successfully')
      } else {
        await copyToClipboard(window.location.href)
        toast.success('URL copied to clipboard')
      }
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Share error:', error)
        toast.error('Failed to share')
      }
    }
  }

  // Context-aware back navigation
  const getBackDestination = () => {
    if (jobId.startsWith('stream-event-')) {
      return '/live-streams'
    } else {
      return '/dashboard'
    }
  }

  const getBackButtonText = () => {
    if (jobId.startsWith('stream-event-')) {
      return 'Back to Live Streams'
    } else {
      return 'Back to Dashboard'
    }
  }

  // Create timeline data for visualization
  const createTimelineData = () => {
    if (!result?.metadata?.duration) return []
    
    const duration = result.metadata.duration
    const interval = 10 // 10-second intervals
    const data = []
    
    for (let time = 0; time < duration; time += interval) {
      const segmentData = {
        time: time,
        timeLabel: formatTime(time),
        violence: 0
      }
      
      // Check if this time intersects with any violence segments
      result.segments?.forEach(segment => {
        if (time >= segment.start && time < segment.end) {
          segmentData.violence = segment.confidence * 100
        }
      })
      
      data.push(segmentData)
    }
    
    return data
  }

  // Get proper video URL with better debugging
  const getVideoUrl = () => {
    if (!result) return null
    
    console.log('Result video_path:', result.video_path)
    
    // For uploaded videos, construct the API endpoint
    if (result.video_path) {
      // Check if it's already a URL
      if (result.video_path.startsWith('http') || result.video_path.startsWith('/api')) {
        console.log('Using direct URL:', result.video_path)
        return result.video_path
      } else {
        // Extract filename from path - handle both Windows and Unix paths
        const filename = result.video_path.split(/[/\\]/).pop()
        console.log('Extracted filename:', filename)
        
        const videoUrl = `/api/uploads/${filename}`
        console.log('Constructed video URL:', videoUrl)
        return videoUrl
      }
    }
    
    // For stream events, check clip_path
    if (result.clip_path) {
      console.log('Using clip_path:', result.clip_path)
      return result.clip_path
    }
    
    console.log('No video URL found')
    return null
  }

  // Video event handlers
  const handleVideoLoadStart = () => {
    setVideoLoading(true)
    setVideoError(false)
    console.log('Video loading started')
  }

  const handleVideoCanPlay = () => {
    setVideoLoading(false)
    setVideoError(false)
    console.log('Video can play')
    toast.success('Video loaded successfully')
  }

  const handleVideoError = (e) => {
    setVideoLoading(false)
    setVideoError(true)
    console.error('Video error:', e.target.error)
    toast.error('Failed to load video')
  }

  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const handlePlayPause = () => {
    if (videoRef.current) {
      if (videoRef.current.paused) {
        videoRef.current.play()
        setPlaying(true)
      } else {
        videoRef.current.pause()
        setPlaying(false)
      }
    }
  }

  // Test video URL function
  const testVideoUrl = async (url) => {
    try {
      const response = await fetch(url, { method: 'HEAD' })
      if (response.ok) {
        toast.success(`Video URL is accessible (${response.status})`)
        console.log('Video response headers:', response.headers)
      } else {
        toast.error(`Video URL failed: ${response.status} ${response.statusText}`)
      }
    } catch (error) {
      toast.error(`Video URL test failed: ${error.message}`)
      console.error('Video URL test error:', error)
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
        <p>Loading results...</p>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
        <h2 className="text-2xl font-bold mb-2">Results Not Found</h2>
        <p className="text-muted-foreground mb-4">
          The analysis results could not be loaded.
        </p>
        <Button onClick={() => navigate(getBackDestination())}>
          <ArrowLeft className="h-4 w-4 mr-2" />
          {getBackButtonText()}
        </Button>
      </div>
    )
  }

  const timelineData = createTimelineData()
  const videoUrl = getVideoUrl()

  return (
    <div className="container mx-auto px-4 py-8">
      <style>{`
        /* Enhanced Results Viewer Animations */
        .back-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .back-button::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
          transition: left 0.5s;
        }
        .back-button:hover::before {
          left: 100%;
        }
        .back-button:hover {
          transform: translateX(-4px) translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        .back-button:hover .back-icon {
          transform: translateX(-2px);
        }
        
        .action-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .action-button::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
          transition: left 0.5s;
        }
        .action-button:hover::before {
          left: 100%;
        }
        .action-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        
        .result-banner {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
        }
        .result-banner:hover {
          transform: translateY(-2px);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.08);
        }
        .result-banner:hover .banner-icon {
          transform: scale(1.1) rotate(5deg);
        }
        .result-banner:hover .banner-badge {
          transform: scale(1.05);
        }
        
        .video-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .video-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 0;
          background: linear-gradient(135deg, hsl(var(--primary) / 0.03), transparent);
          transition: height 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .video-card:hover::before {
          height: 100%;
        }
        .video-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        
        .segment-card {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
          position: relative;
        }
        .segment-card::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          width: 0;
          height: 100%;
          background: linear-gradient(90deg, hsl(var(--destructive) / 0.1), transparent);
          transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .segment-card:hover::before {
          width: 100%;
        }
        .segment-card:hover {
          transform: translateX(6px) translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
          border-color: hsl(var(--destructive) / 0.3);
        }
        .segment-card:hover .segment-badge {
          transform: scale(1.1);
        }
        .segment-card:hover .segment-number {
          transform: scale(1.1);
          color: hsl(var(--destructive));
        }
        
        .metadata-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
          position: relative;
          overflow: hidden;
        }
        .metadata-card::before {
          content: '';
          position: absolute;
          top: 0;
          right: 0;
          width: 0;
          height: 100%;
          background: linear-gradient(270deg, hsl(var(--primary) / 0.05), transparent);
          transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .metadata-card:hover::before {
          width: 100%;
        }
        .metadata-card:hover {
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        .metadata-card:hover .card-icon {
          transform: scale(1.1) rotate(5deg);
          color: hsl(var(--primary));
        }
        
        .quick-action {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .quick-action:hover {
          transform: translateX(6px) translateY(-2px);
          background: hsl(var(--primary) / 0.05);
          border-color: hsl(var(--primary) / 0.2);
        }
        .quick-action:hover .action-icon {
          transform: scale(1.1);
          color: hsl(var(--primary));
        }
        
        /* Icon animations */
        .back-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .banner-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .banner-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .segment-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .segment-number {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .card-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .action-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Video container styling */
        .video-container {
          position: relative;
          background: #000;
          border-radius: 8px;
          overflow: hidden;
        }
        .video-element {
          width: 100%;
          height: 100%;
          border-radius: 8px;
        }
        .video-fallback {
          background: linear-gradient(135deg, #1f2937, #111827);
          border: 2px dashed #374151;
          border-radius: 8px;
        }
      `}</style>
      
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-4">
          <Button variant="outline" onClick={() => navigate(getBackDestination())} className="back-button">
            <ArrowLeft className="h-4 w-4 mr-2 back-icon" />
            {getBackButtonText()}
          </Button>
          <div>
            <div className="flex items-center space-x-2 mb-1">
              <h1 className="text-3xl font-bold">{result.filename}</h1>
              {jobId.startsWith('stream-event-') && (
                <Badge variant="outline" className="text-sm">
                  Live Stream Event
                </Badge>
              )}
            </div>
            <p className="text-muted-foreground">
              Analyzed on {format(new Date(result.timestamp), 'PPpp')}
              {jobId.startsWith('stream-event-') && result.stream_metadata && (
                <span> • Stream: {result.stream_metadata.stream_name}</span>
              )}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={downloadReport} className="action-button">
            <Download className="h-4 w-4 mr-2" />
            Download Report
          </Button>
          <Button variant="outline" onClick={shareResults} className="action-button">
            <Share2 className="h-4 w-4 mr-2" />
            Share Results
          </Button>
        </div>
      </div>

      {/* Overall Result Banner */}
      <Card className={`mb-8 result-banner ${result.has_violence ? 'border-red-200 bg-red-50 dark:bg-red-950/20' : 'border-green-200 bg-green-50 dark:bg-green-950/20'}`}>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              {result.has_violence ? (
                <AlertTriangle className="h-8 w-8 text-red-500 banner-icon" />
              ) : (
                <CheckCircle className="h-8 w-8 text-green-500 banner-icon" />
              )}
              <div>
                <h2 className="text-2xl font-bold">
                  {result.has_violence ? 'Violence Detected' : 'No Violence Detected'}
                </h2>
                <p className="text-muted-foreground">
                  Overall confidence: {(result.overall_result.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            
            <div className="text-right">
              <Badge 
                variant={result.has_violence ? 'destructive' : 'default'}
                className="text-lg px-4 py-2 banner-badge"
              >
                {result.has_violence ? 'VIOLENT' : 'SAFE'}
              </Badge>
            </div>
          </div>
          
          {result.has_violence && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {result.segments?.length || 0}
                </div>
                <div className="text-sm text-muted-foreground">Violent Segments</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {formatTime(result.violence_duration || 0)}
                </div>
                <div className="text-sm text-muted-foreground">Total Duration</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-red-600">
                  {(result.violence_percentage || 0).toFixed(1)}%
                </div>
                <div className="text-sm text-muted-foreground">Of Video</div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Video Player and Timeline */}
        <div className="lg:col-span-2 space-y-6">
          {/* Video Player */}
          <Card className="video-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Play className="h-5 w-5 mr-2" />
                Video Playback
              </CardTitle>
              <CardDescription>
                {videoUrl ? (
                  <div>
                    <div>Click on segments below to jump to specific times</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Video URL: <code>{videoUrl}</code>
                      <Button 
                        variant="ghost" 
                        size="sm" 
                        className="ml-2 h-6 px-2 text-xs"
                        onClick={() => testVideoUrl(videoUrl)}
                      >
                        Test URL
                      </Button>
                    </div>
                  </div>
                ) : (
                  'Video playback not available'
                )}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-video">
                {videoUrl ? (
                  <div className="video-container w-full h-full">
                    {videoLoading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
                        <span className="ml-2 text-white">Loading video...</span>
                      </div>
                    )}
                    <video
                      ref={videoRef}
                      className="video-element"
                      controls
                      onLoadStart={handleVideoLoadStart}
                      onCanPlay={handleVideoCanPlay}
                      onError={handleVideoError}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onPlay={() => setPlaying(true)}
                      onPause={() => setPlaying(false)}
                      crossOrigin="anonymous"
                      preload="metadata"
                    >
                      <source src={videoUrl} type="video/mp4" />
                      <source src={videoUrl} type="video/avi" />
                      <source src={videoUrl} type="video/quicktime" />
                      <source src={videoUrl} type="video/x-msvideo" />
                      Your browser does not support the video tag.
                    </video>
                  </div>
                ) : (
                  <div className="w-full h-full flex items-center justify-center video-fallback">
                    <div className="text-center">
                      <FileVideo className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground mb-2">
                        Video playback not available
                      </p>
                      <p className="text-sm text-muted-foreground">
                        The video file cannot be accessed for playback
                      </p>
                      {result.thumbnail && (
                        <div className="mt-4">
                          <img 
                            src={result.thumbnail} 
                            alt="Video thumbnail"
                            className="mx-auto rounded border max-w-48"
                          />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Timeline Visualization */}
          {timelineData.length > 0 && (
            <Card className="video-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <TrendingUp className="h-5 w-5 mr-2" />
                  Violence Timeline
                </CardTitle>
                <CardDescription>
                  Violence confidence levels throughout the video
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="timeLabel" 
                      interval="preserveStartEnd"
                    />
                    <YAxis 
                      domain={[0, 100]}
                      label={{ value: 'Confidence %', angle: -90, position: 'insideLeft' }}
                    />
                    <Tooltip 
                      formatter={(value) => [`${value.toFixed(1)}%`, 'Violence Confidence']}
                      labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Bar 
                      dataKey="violence" 
                      fill="hsl(var(--destructive))"
                      opacity={0.8}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          )}

          {/* Violent Segments */}
          {result.segments && result.segments.length > 0 && (
            <Card className="video-card">
              <CardHeader>
                <CardTitle className="flex items-center">
                  <Target className="h-5 w-5 mr-2" />
                  Detected Segments
                </CardTitle>
                <CardDescription>
                  Click on any segment to jump to that time in the video
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {result.segments.map((segment, index) => (
                    <div 
                      key={index}
                      className="flex items-center justify-between p-3 border rounded-lg segment-card"
                      onClick={() => jumpToTime(segment.start)}
                    >
                      <div className="flex items-center space-x-3">
                        <Badge variant="outline" className="segment-number">#{index + 1}</Badge>
                        <div>
                          <div className="font-medium">
                            {segment.start_formatted} - {segment.end_formatted}
                          </div>
                          <div className="text-sm text-muted-foreground">
                            Duration: {formatTime(segment.end - segment.start)}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <Badge variant="destructive" className="segment-badge">
                          {(segment.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* Metadata and Analysis Details */}
        <div className="space-y-6">
          {/* Video Metadata */}
          <Card className="metadata-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Info className="h-5 w-5 mr-2 card-icon" />
                Video Details
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Duration:</span>
                <span className="font-medium">{result.metadata?.duration_formatted || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Resolution:</span>
                <span className="font-medium">
                  {result.metadata?.width && result.metadata?.height 
                    ? `${result.metadata.width}x${result.metadata.height}` 
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Frame Rate:</span>
                <span className="font-medium">
                  {result.metadata?.fps ? `${result.metadata.fps.toFixed(1)} fps` : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Frames:</span>
                <span className="font-medium">
                  {result.metadata?.frame_count ? result.metadata.frame_count.toLocaleString() : 'N/A'}
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Model Information */}
          <Card className="metadata-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Brain className="h-5 w-5 mr-2 card-icon" />
                Model Details
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Architecture:</span>
                <span className="font-medium">{result.model_info?.architecture || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Input Frames:</span>
                <span className="font-medium">{result.model_info?.input_frames || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Resolution:</span>
                <span className="font-medium">{result.model_info?.input_resolution || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Motion Enhancement:</span>
                <Badge variant={result.model_info?.motion_enhancement ? 'default' : 'outline'}>
                  {result.model_info?.motion_enhancement ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <Card className="metadata-card">
            <CardHeader>
              <CardTitle className="flex items-center">
                <Zap className="h-5 w-5 mr-2 card-icon" />
                Performance
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Inference Time:</span>
                <span className="font-medium">
                  {result.overall_result?.inference_time 
                    ? `${result.overall_result.inference_time.toFixed(3)}s`
                    : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Processing Speed:</span>
                <span className="font-medium">
                  {result.metadata?.duration && result.overall_result?.inference_time 
                    ? `${(result.metadata.duration / result.overall_result.inference_time).toFixed(1)}x`
                    : 'N/A'
                  }
                </span>
              </div>
            </CardContent>
          </Card>

          {/* Quick Actions */}
          <Card className="metadata-card">
            <CardHeader>
              <CardTitle>Quick Actions</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button 
                variant="outline" 
                className="w-full justify-start quick-action"
                onClick={() => copyToClipboard(result.job_id)}
              >
                <Copy className="h-4 w-4 mr-2 action-icon" />
                Copy Job ID
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start quick-action"
                onClick={downloadReport}
              >
                <Download className="h-4 w-4 mr-2 action-icon" />
                Export Results
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start quick-action"
                onClick={() => navigate('/upload')}
              >
                <Upload className="h-4 w-4 mr-2 action-icon" />
                Analyze Another Video
              </Button>
              <Button 
                variant="outline" 
                className="w-full justify-start quick-action"
                onClick={() => copyToClipboard(window.location.href)}
              >
                <Share2 className="h-4 w-4 mr-2 action-icon" />
                Copy Results Link
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default ResultsViewer