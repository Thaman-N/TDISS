import React, { useState, useEffect, useRef, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area
} from 'recharts'
import {
  ArrowLeft,
  Download,
  Share2,
  AlertTriangle,
  CheckCircle,
  Brain,
  Zap,
  Play,
  Copy,
  FileVideo,
  Target,
  TrendingUp,
  Info,
  Upload,
  BarChart3,
  Activity,
  LineChart as LineChartIcon
} from 'lucide-react'
import { format } from 'date-fns'

// Chart configuration
const chartConfig = {
  violence: {
    label: "Violence Confidence",
    color: "hsl(var(--destructive))",
  },
}

// Helper: ChartTypeSelector Component (Moved Outside)
const ChartTypeSelector = ({ chartType, setChartType }) => (
  <div className="flex items-center space-x-2 mb-4">
    <span className="text-sm font-medium">Chart Type:</span>
    <div className="flex space-x-1">
      <Button
        variant={chartType === 'area' ? 'default' : 'outline'}
        size="sm"
        onClick={() => setChartType('area')}
        className="h-8 px-3"
      >
        <Activity className="h-3 w-3 mr-1" />
        Area
      </Button>
      <Button
        variant={chartType === 'bar' ? 'default' : 'outline'}
        size="sm"
        onClick={() => setChartType('bar')}
        className="h-8 px-3"
      >
        <BarChart3 className="h-3 w-3 mr-1" />
        Bar
      </Button>
      <Button
        variant={chartType === 'line' ? 'default' : 'outline'}
        size="sm"
        onClick={() => setChartType('line')}
        className="h-8 px-3"
      >
        <LineChartIcon className="h-3 w-3 mr-1" />
        Line
      </Button>
    </div>
  </div>
)

// Helper: CustomChartTooltip Component
const CustomChartTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload
    return (
      <ChartTooltipContent>
        <div className="bg-background p-3 border rounded-lg shadow-lg">
          <p className="font-medium">Time: {label}</p>
          <p className="text-destructive">
            Violence: {payload[0].value?.toFixed(1)}% ({data.violenceLevel})
          </p>
        </div>
      </ChartTooltipContent>
    )
  }
  return null
}

// Helper: Enhanced ViolenceChart Component (Moved Outside)
const ViolenceChart = ({ data, type, onChartClick }) => {
  const commonProps = {
    data,
    margin: { top: 10, right: 30, left: 20, bottom: 20 },
    onClick: onChartClick,
  }

  const renderChart = () => {
    switch (type) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <defs>
              <linearGradient id="violenceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="hsl(var(--destructive))" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="hsl(var(--destructive))" stopOpacity={0.1}/>
              </linearGradient>
            </defs>
            <CartesianGrid opacity={0.1} vertical={false}/>
            <XAxis 
              dataKey="timeLabel" 
              interval="preserveStartEnd"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={[0, 100]}
              label={{ value: 'Confidence %', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomChartTooltip />} cursor={{ stroke: 'hsl(var(--destructive))', strokeWidth: 1, strokeDasharray: '3 3' }} />
            <Area
              type="monotone"
              dataKey="violence"
              stroke="hsl(var(--destructive))"
              strokeWidth={2}
              fillOpacity={1}
              fill="url(#violenceGradient)"
            />
          </AreaChart>
        )
      
      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid opacity={0.1} vertical={false}/>
            <XAxis 
              dataKey="timeLabel" 
              interval="preserveStartEnd"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={[0, 100]}
              label={{ value: 'Confidence %', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomChartTooltip />} cursor={{ fill: 'hsl(var(--muted))' }} />
            <Bar 
              dataKey="violence" 
              fill="hsl(var(--destructive))"
              opacity={0.8}
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        )
      
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid opacity={0.1} vertical={false}/>
            <XAxis 
              dataKey="timeLabel" 
              interval="preserveStartEnd"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              domain={[0, 100]}
              label={{ value: 'Confidence %', angle: -90, position: 'insideLeft' }}
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomChartTooltip />} cursor={{ stroke: 'hsl(var(--destructive))', strokeWidth: 1 }} />
            <Line
              type="monotone"
              dataKey="violence"
              stroke="hsl(var(--destructive))"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6, stroke: 'hsl(var(--destructive))', strokeWidth: 2 }}
            />
          </LineChart>
        )
      
      default:
        return null
    }
  }

  return (
    <ChartContainer config={chartConfig} className="h-[300px] w-full cursor-pointer">
      <ResponsiveContainer width="100%" height="100%">
        {renderChart()}
      </ResponsiveContainer>
    </ChartContainer>
  )
}

const ResultsViewer = () => {
  const { jobId, incidentId } = useParams()
  const actualJobId = jobId || `incident-${incidentId}`
  const navigate = useNavigate()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [videoLoading, setVideoLoading] = useState(true)
  const [videoError, setVideoError] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [chartType, setChartType] = useState('area')
  
  const videoRef = useRef(null)

  useEffect(() => {
    const fetchResult = async () => {
      setLoading(true)
      try {
        let response;
        
        if (actualJobId.startsWith('incident-')) {
          const incidentIdFromUrl = actualJobId.replace('incident-', '') // <-- FIX: Corrected typo 'actualJob-Id'
          response = await fetch(`/api/incident-result/${incidentIdFromUrl}`)
        } else if (actualJobId.startsWith('stream-event-')) {
          const eventId = actualJobId.replace('stream-event-', '')
          response = await fetch(`/api/stream-event/${eventId}`)
        } else {
          response = await fetch(`/api/result/${actualJobId}`)
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
        if (actualJobId.startsWith('stream-event-') || actualJobId.startsWith('incident-')) {
          navigate('/live-streams')
        } else {
          navigate('/dashboard')
        }
      } finally {
        setLoading(false)
      }
    }
    
    fetchResult()
  }, [actualJobId, navigate])

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const jumpToTime = (time) => {
    if (videoRef.current && !videoError) {
      try {
        videoRef.current.currentTime = time
        setCurrentTime(time)
        if (videoRef.current.paused) {
          videoRef.current.play()
        }
      } catch (error) {
        console.error('Error seeking to time:', error)
        toast.error('Failed to jump to segment')
      }
    } else {
      toast.warning('Video not available for seeking')
    }
  }

  const copyToClipboard = async (text) => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text)
        toast.success('Copied to clipboard')
      } else {
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
      
      const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' })
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

  const getBackDestination = () => {
    if (actualJobId.startsWith('stream-event-') || actualJobId.startsWith('incident-')) {
      return '/live-streams'
    } else {
      return '/dashboard'
    }
  }

  const getBackButtonText = () => {
    if (actualJobId.startsWith('stream-event-') || actualJobId.startsWith('incident-')) {
      return 'Back to Live Streams'
    } else {
      return 'Back to Dashboard'
    }
  }
  
  const handleChartClick = (e) => {
    if (e && e.activePayload && e.activePayload.length > 0) {
      const time = e.activePayload[0].payload.time;
      jumpToTime(time);
    }
  };

  // Video event handlers
  const handleVideoLoadStart = () => setVideoLoading(true)
  const handleVideoCanPlay = () => setVideoLoading(false)
  const handleVideoError = (e) => {
    setVideoLoading(false)
    setVideoError(true)
    const error = e.target?.error
    console.error('Video error:', error)
    if (error) {
      toast.error(`Failed to load video - ${error.message || 'Check file format'}`)
    } else {
      toast.error('Failed to load video - Unknown error')
    }
  }
  const handleVideoTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const testVideoUrl = async (url) => {
    try {
      const response = await fetch(url, { method: 'HEAD' })
      if (response.ok) {
        toast.success(`Video URL is accessible (${response.status})`)
      } else {
        toast.error(`Video URL failed: ${response.status} ${response.statusText}`)
      }
    } catch (error) {
      toast.error(`Video URL test failed: ${error.message}`)
      console.error('Video URL test error:', error)
    }
  }

  // FIX: Memoize expensive calculations to prevent re-running on every render
  const timelineData = useMemo(() => {
    if (!result?.metadata?.duration) return []
    
    const duration = result.metadata.duration
    const interval = 0.5
    const data = []
    
    for (let time = 0; time <= duration; time += interval) {
      const segmentData = {
        time: time,
        timeLabel: formatTime(time),
        violence: 0,
        violenceLevel: 'None'
      }
      
      let maxConfidence = 0
      result.segments?.forEach(segment => {
        if (time >= segment.start && time <= segment.end) {
          maxConfidence = Math.max(maxConfidence, segment.confidence * 100)
        }
      })
      
      segmentData.violence = maxConfidence
      
      if (maxConfidence === 0) {
        segmentData.violenceLevel = 'None'
      } else if (maxConfidence < 30) {
        segmentData.violenceLevel = 'Low'
      } else if (maxConfidence < 70) {
        segmentData.violenceLevel = 'Medium' 
      } else {
        segmentData.violenceLevel = 'High'
      }
      
      data.push(segmentData)
    }
    
    return data
  }, [result])

  const videoUrl = useMemo(() => {
    if (!result) return null
    
    if (result.clip_path) return result.clip_path
    
    if (result.video_path) {
      if (result.video_path.startsWith('http') || result.video_path.startsWith('/api')) {
        return result.video_path
      } else {
        const filename = result.video_path.split(/[/\\]/).pop()
        return `/api/uploads/${filename}`
      }
    }
    
    return null
  }, [result])


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

  return (
    <div className="container mx-auto px-4 py-8">
      <style>{`
        /* Enhanced Results Viewer Animations */
        .back-button, .action-button { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden; }
        .back-button:hover, .action-button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); }
        .compact-banner, .video-card { transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
        .compact-banner:hover, .video-card:hover { transform: translateY(-2px); box-shadow: 0 12px 25px rgba(0, 0, 0, 0.08); }
        .segments-sidebar { height: fit-content; max-height: calc(100vh - 200px); position: sticky; top: 20px; min-width: 280px; }
        .segments-list { max-height: 400px; overflow-y: auto; overflow-x: hidden; }
        .segments-list::-webkit-scrollbar { width: 6px; }
        .segments-list::-webkit-scrollbar-track { background: hsl(var(--muted)); border-radius: 3px; }
        .segments-list::-webkit-scrollbar-thumb { background: hsl(var(--muted-foreground) / 0.3); border-radius: 3px; }
        .segments-list::-webkit-scrollbar-thumb:hover { background: hsl(var(--muted-foreground) / 0.5); }
        .segment-card { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); cursor: pointer; }
        .segment-card:hover { transform: translateX(6px) translateY(-2px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08); border-color: hsl(var(--destructive) / 0.3); }
        .metadata-card { transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
        .metadata-card:hover { transform: translateY(-3px) scale(1.02); box-shadow: 0 12px 25px rgba(0, 0, 0, 0.1); border-color: hsl(var(--primary) / 0.2); }
        .video-container { position: relative; background: #000; border-radius: 8px; overflow: hidden; }
        .video-element { width: 100%; height: 100%; border-radius: 8px; }
        .video-fallback { background: linear-gradient(135deg, #1f2937, #111827); border: 2px dashed #374151; border-radius: 8px; }
        .quick-action { transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); width: 100%; text-align: left; }
        .quick-action:hover { transform: translateX(6px) translateY(-2px); background: hsl(var(--primary) / 0.05); border-color: hsl(var(--primary) / 0.2); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); }
      `}</style>

      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <Button variant="outline" onClick={() => navigate(getBackDestination())} className="back-button">
            <ArrowLeft className="h-4 w-4 mr-2" />
            {getBackButtonText()}
          </Button>
          <div>
            <h1 className="text-3xl font-bold truncate max-w-lg">{result.filename}</h1>
            <p className="text-muted-foreground text-sm">
              Analyzed on {format(new Date(result.timestamp), 'PPpp')}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" onClick={downloadReport} className="action-button">
            <Download className="h-4 w-4 mr-2" />
            Download
          </Button>
          <Button variant="outline" onClick={shareResults} className="action-button">
            <Share2 className="h-4 w-4 mr-2" />
            Share
          </Button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column - Video Player and Timeline */}
        <div className="lg:col-span-3 space-y-6">
          <Card className="video-card">
            <CardHeader>
              <CardTitle className="flex items-center"><Play className="h-5 w-5 mr-2" /> Video Playback</CardTitle>
              <CardDescription>Click on segments or the timeline graph to jump to specific times</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="aspect-video">
                {videoUrl ? (
                  <div className="video-container w-full h-full">
                    {videoLoading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white"></div>
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
                      src={videoUrl}
                    >
                      Your browser does not support the video tag.
                    </video>
                  </div>
                ) : (
                  <div className="w-full h-full flex items-center justify-center video-fallback">
                    <div className="text-center"><FileVideo className="h-16 w-16 text-muted-foreground mx-auto mb-4" />
                      <p className="text-muted-foreground">Video playback not available</p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {timelineData.length > 0 && (
            <Card className="video-card">
              <CardHeader>
                <CardTitle className="flex items-center"><TrendingUp className="h-5 w-5 mr-2" /> Violence Timeline</CardTitle>
                <CardDescription>Violence confidence levels throughout the video.</CardDescription>
              </CardHeader>
              <CardContent>
                <ChartTypeSelector chartType={chartType} setChartType={setChartType} />
                <ViolenceChart data={timelineData} type={chartType} onChartClick={handleChartClick} />
              </CardContent>
            </Card>
          )}
        </div>

        {/* Right Sidebar */}
        <div className="lg:col-span-1">
          <Card className="segments-sidebar">
            <CardHeader>
              <CardTitle className="flex items-center"><Target className="h-5 w-5 mr-2" /> Detected Segments</CardTitle>
              <CardDescription>
                {result.segments?.length > 0 ? `${result.segments.length} segment(s) found` : 'No violent segments detected'}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {result.segments?.length > 0 ? (
                <div className="segments-list space-y-3">
                  {result.segments.map((segment, index) => (
                    <div key={index} className="flex items-center justify-between p-2.5 border rounded-lg segment-card" onClick={() => jumpToTime(segment.start)}>
                      <div className="flex items-center space-x-2 min-w-0 flex-1">
                        <Badge variant="outline" className="flex-shrink-0 text-xs">#{index + 1}</Badge>
                        <div className="min-w-0 flex-1">
                          <div className="font-medium text-xs truncate">{segment.start_formatted} - {segment.end_formatted}</div>
                          <div className="text-xs text-muted-foreground truncate">{formatTime(segment.end - segment.start)}</div>
                        </div>
                      </div>
                      <Badge variant="destructive" className="text-xs flex-shrink-0 ml-2">{(segment.confidence * 100).toFixed(1)}%</Badge>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8"><CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground">No violent segments detected.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Bottom Metadata Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-8">
        <Card className="metadata-card">
          <CardHeader><CardTitle className="flex items-center text-lg"><Info className="h-5 w-5 mr-2" /> Video Details</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-muted-foreground">Duration:</span><span className="font-medium">{result.metadata?.duration_formatted || 'N/A'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Resolution:</span><span className="font-medium">{result.metadata?.width ? `${result.metadata.width}x${result.metadata.height}` : 'N/A'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Frame Rate:</span><span className="font-medium">{result.metadata?.fps ? `${result.metadata.fps.toFixed(1)} fps` : 'N/A'}</span></div>
          </CardContent>
        </Card>
        <Card className="metadata-card">
          <CardHeader><CardTitle className="flex items-center text-lg"><Brain className="h-5 w-5 mr-2" /> Model Details</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-muted-foreground">Architecture:</span><span className="font-medium">{result.model_info?.architecture || 'N/A'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Input Frames:</span><span className="font-medium">{result.model_info?.input_frames || 'N/A'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Motion:</span><Badge variant={result.model_info?.motion_enhancement ? 'default' : 'outline'} className="text-xs">{result.model_info?.motion_enhancement ? 'Enabled' : 'Disabled'}</Badge></div>
          </CardContent>
        </Card>
        <Card className="metadata-card">
          <CardHeader><CardTitle className="flex items-center text-lg"><Zap className="h-5 w-5 mr-2" /> Performance</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm">
            <div className="flex justify-between"><span className="text-muted-foreground">Inference Time:</span><span className="font-medium">{result.overall_result?.inference_time ? `${result.overall_result.inference_time.toFixed(3)}s` : 'N/A'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Speed:</span><span className="font-medium">{result.metadata?.duration && result.overall_result?.inference_time ? `${(result.metadata.duration / result.overall_result.inference_time).toFixed(1)}x` : 'N/A'}</span></div>
            <div className="flex justify-between"><span className="text-muted-foreground">Confidence:</span><span className="font-medium">{result.overall_result?.confidence ? `${(result.overall_result.confidence * 100).toFixed(1)}%` : 'N/A'}</span></div>
          </CardContent>
        </Card>
        <Card className="metadata-card">
          <CardHeader><CardTitle className="text-lg">Quick Actions</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            <Button variant="outline" size="sm" className="w-full justify-start quick-action" onClick={() => copyToClipboard(result.job_id)}><Copy className="h-4 w-4 mr-2" /> Copy Job ID</Button>
            <Button variant="outline" size="sm" className="w-full justify-start quick-action" onClick={downloadReport}><Download className="h-4 w-4 mr-2" /> Export Results</Button>
            <Button variant="outline" size="sm" className="w-full justify-start quick-action" onClick={() => navigate('/upload')}><Upload className="h-4 w-4 mr-2" /> Analyze Another</Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default ResultsViewer