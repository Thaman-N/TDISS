import React, { useState, useEffect, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useWebSocket } from '@/contexts/WebSocketContext'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Play, 
  Pause, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  ArrowLeft,
  Volume2,
  VolumeX,
  Maximize,
  Download
} from 'lucide-react'

const MultiVideoGridViewer = () => {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const { isConnected: wsConnected, registerJobUpdateCallback } = useWebSocket()
  
  const jobIds = searchParams.get('jobs')?.split(',') || []
  const [videos, setVideos] = useState({})
  const [playingVideos, setPlayingVideos] = useState(new Set())
  const [mutedVideos, setMutedVideos] = useState(new Set(jobIds)) // Start all muted
  const [loadingStates, setLoadingStates] = useState({})

  // Calculate grid dimensions based on number of videos
  const getGridClass = (count) => {
    if (count === 1) return 'grid-cols-1'
    if (count === 2) return 'grid-cols-2'
    if (count <= 4) return 'grid-cols-2'
    if (count <= 6) return 'grid-cols-3'
    if (count <= 9) return 'grid-cols-3'
    return 'grid-cols-4'
  }

  const getVideoSize = (count) => {
    if (count === 1) return 'aspect-video'
    if (count === 2) return 'aspect-video'
    if (count <= 4) return 'aspect-video'
    if (count <= 9) return 'aspect-video'
    return 'aspect-[4/3]'
  }

  const getVideoUrl = (video) => {
    if (video.processed_video_path) {
      const filename = video.processed_video_path.split(/[/\\]/).pop()
      return `/api/results/${filename}`
    } else if (video.video_path) {
      const filename = video.video_path.split(/[/\\]/).pop()
      return `/api/uploads/${filename}`
    }
    return null
  }

  // Fetch video results for each job (only once on mount)
  useEffect(() => {
    const fetchVideoResults = async () => {
      for (const jobId of jobIds) {
        setLoadingStates(prev => ({ ...prev, [jobId]: true }))
        
        try {
          // First get job status
          const jobResponse = await fetch(`/api/job/${jobId}`)
          if (jobResponse.ok) {
            const jobData = await jobResponse.json()
            
            // If job is completed, get detailed results
            let detailedResult = null
            if (jobData.status === 'completed') {
              try {
                const resultResponse = await fetch(`/api/result/${jobId}`)
                if (resultResponse.ok) {
                  detailedResult = await resultResponse.json()
                }
              } catch (resultError) {
                console.warn(`Could not fetch detailed results for ${jobId}:`, resultError)
              }
            }

            setVideos(prev => ({
              ...prev,
              [jobId]: {
                ...jobData,
                ...detailedResult,
                jobId,
                fileName: jobData.metadata?.original_filename || detailedResult?.video_path?.split('/').pop() || `Video ${jobId}`
              }
            }))
          }
        } catch (error) {
          console.error(`Failed to fetch results for job ${jobId}:`, error)
          setVideos(prev => ({
            ...prev,
            [jobId]: {
              jobId,
              status: 'error',
              fileName: `Video ${jobId}`,
              error: 'Failed to load video results'
            }
          }))
        } finally {
          setLoadingStates(prev => ({ ...prev, [jobId]: false }))
        }
      }
    }

    if (jobIds.length > 0) {
      fetchVideoResults()
    }
  }, []) // Empty dependency array - only run once on mount

  // Handle WebSocket job updates
  const handleJobUpdate = useCallback((jobId, jobData) => {
    if (jobIds.includes(jobId)) {
      setVideos(prev => {
        const existing = prev[jobId]
        const hasChanges = 
          !existing ||
          existing.status !== jobData.status ||
          existing.progress !== jobData.progress ||
          existing.message !== jobData.message

        if (hasChanges) {
          // If job just completed, fetch detailed results
          if (jobData.status === 'completed' && existing?.status !== 'completed') {
            fetchJobResults(jobId)
          }

          return {
            ...prev,
            [jobId]: {
              ...existing,
              ...jobData,
              jobId,
              fileName: existing?.fileName || jobData.metadata?.original_filename || `Video ${jobId}`
            }
          }
        }
        return prev
      })
    }
  }, [jobIds])

  const fetchJobResults = async (jobId) => {
    try {
      const resultResponse = await fetch(`/api/result/${jobId}`)
      if (resultResponse.ok) {
        const detailedResult = await resultResponse.json()
        setVideos(prev => ({
          ...prev,
          [jobId]: {
            ...prev[jobId],
            ...detailedResult
          }
        }))
      }
    } catch (error) {
      console.warn(`Could not fetch detailed results for ${jobId}:`, error)
    }
  }

  useEffect(() => {
    // Register for job updates via WebSocket
    const unregister = registerJobUpdateCallback(handleJobUpdate)
    return unregister
  }, [registerJobUpdateCallback, handleJobUpdate])

  // Periodic refresh for active jobs (every 10 seconds)
  useEffect(() => {
    const refreshActiveJobs = async () => {
      const activeJobIds = Object.entries(videos)
        .filter(([_, video]) => ['queued', 'processing'].includes(video.status))
        .map(([jobId]) => jobId)

      if (activeJobIds.length > 0 && wsConnected) {
        for (const jobId of activeJobIds) {
          try {
            const jobResponse = await fetch(`/api/job/${jobId}`)
            if (jobResponse.ok) {
              const jobData = await jobResponse.json()
              handleJobUpdate(jobId, jobData)
            }
          } catch (error) {
            console.warn(`Failed to refresh job ${jobId}:`, error)
          }
        }
      }
    }

    const interval = setInterval(refreshActiveJobs, 10000)
    return () => clearInterval(interval)
  }, [videos, wsConnected, handleJobUpdate])

  const togglePlay = (jobId) => {
    const video = document.getElementById(`video-${jobId}`)
    if (video) {
      if (playingVideos.has(jobId)) {
        video.pause()
        setPlayingVideos(prev => {
          const newSet = new Set(prev)
          newSet.delete(jobId)
          return newSet
        })
      } else {
        video.play()
        setPlayingVideos(prev => new Set([...prev, jobId]))
      }
    }
  }

  const toggleMute = (jobId) => {
    const video = document.getElementById(`video-${jobId}`)
    if (video) {
      if (mutedVideos.has(jobId)) {
        video.muted = false
        setMutedVideos(prev => {
          const newSet = new Set(prev)
          newSet.delete(jobId)
          return newSet
        })
      } else {
        video.muted = true
        setMutedVideos(prev => new Set([...prev, jobId]))
      }
    }
  }

  const getStatusIcon = (video) => {
    if (loadingStates[video.jobId]) {
      return <Clock className="h-4 w-4 animate-spin" />
    }
    
    switch (video.status) {
      case 'completed':
        return video.violence_detected ? (
          <AlertTriangle className="h-4 w-4 text-red-500" />
        ) : (
          <CheckCircle className="h-4 w-4 text-green-500" />
        )
      case 'processing':
        return <Clock className="h-4 w-4 animate-spin text-blue-500" />
      case 'error':
        return <AlertTriangle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (video) => {
    if (loadingStates[video.jobId]) {
      return <Badge variant="secondary">Loading...</Badge>
    }

    switch (video.status) {
      case 'completed':
        return video.has_violence ? (
          <Badge variant="destructive">Violence Detected</Badge>
        ) : (
          <Badge variant="secondary" className="bg-green-100 text-green-800">Safe</Badge>
        )
      case 'processing':
        return <Badge variant="secondary">Processing...</Badge>
      case 'error':
        return <Badge variant="destructive">Error</Badge>
      default:
        return <Badge variant="outline">Pending</Badge>
    }
  }

  const downloadResults = (jobId) => {
    const video = videos[jobId]
    if (video && video.status === 'completed') {
      // Create downloadable report
      const report = {
        jobId: video.jobId,
        fileName: video.fileName,
        status: video.status,
        hasViolence: video.has_violence,
        confidence: video.overall_confidence,
        segments: video.segments || [],
        violenceDuration: video.violence_duration,
        violencePercentage: video.violence_percentage,
        processingStats: video.processing_stats,
        timestamp: new Date().toISOString()
      }
      
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `analysis-report-${jobId}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
    }
  }

  if (jobIds.length === 0) {
    return (
      <div className="container mx-auto px-4 py-12">
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertDescription>
            No video jobs specified. Please upload videos first.
          </AlertDescription>
        </Alert>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <Button 
            variant="outline" 
            onClick={() => navigate('/')}
            className="mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Upload
          </Button>
          <h1 className="text-3xl font-bold">Multi-Video Analysis</h1>
          <p className="text-muted-foreground">
            Analyzing {jobIds.length} video{jobIds.length !== 1 ? 's' : ''}
          </p>
        </div>
      </div>

      {/* Video Grid */}
      <div className={`grid ${getGridClass(jobIds.length)} gap-6`}>
        {jobIds.map((jobId) => {
          const video = videos[jobId] || { jobId, status: 'loading', fileName: `Video ${jobId}` }
          const isHighlighted = video.status === 'completed' && video.has_violence

          return (
            <Card 
              key={jobId} 
              className={`transition-all duration-300 ${
                isHighlighted 
                  ? 'ring-2 ring-red-500 shadow-lg shadow-red-500/20' 
                  : 'hover:shadow-md'
              }`}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div className="flex items-center space-x-2">
                    {getStatusIcon(video)}
                    <CardTitle className="text-sm truncate max-w-32">
                      {video.fileName}
                    </CardTitle>
                  </div>
                  {getStatusBadge(video)}
                </div>
              </CardHeader>
              
              <CardContent className="space-y-4">
                {/* Video Player */}
                <div className={`relative ${getVideoSize(jobIds.length)} bg-black rounded-lg overflow-hidden`}>
                  {video.status === 'completed' && (video.processed_video_path || video.video_path) ? (
                    <>
                      <video
                        id={`video-${jobId}`}
                        src={getVideoUrl(video)}
                        className="w-full h-full object-contain"
                        muted={mutedVideos.has(jobId)}
                        onPlay={() => setPlayingVideos(prev => new Set([...prev, jobId]))}
                        onPause={() => setPlayingVideos(prev => {
                          const newSet = new Set(prev)
                          newSet.delete(jobId)
                          return newSet
                        })}
                      />
                      
                      {/* Video Controls Overlay */}
                      <div className="absolute inset-0 bg-black/20 opacity-0 hover:opacity-100 transition-opacity duration-200 flex items-center justify-center">
                        <div className="flex space-x-2">
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => togglePlay(jobId)}
                            className="bg-white/20 hover:bg-white/30"
                          >
                            {playingVideos.has(jobId) ? (
                              <Pause className="h-4 w-4" />
                            ) : (
                              <Play className="h-4 w-4" />
                            )}
                          </Button>
                          
                          <Button
                            size="sm"
                            variant="secondary"
                            onClick={() => toggleMute(jobId)}
                            className="bg-white/20 hover:bg-white/30"
                          >
                            {mutedVideos.has(jobId) ? (
                              <VolumeX className="h-4 w-4" />
                            ) : (
                              <Volume2 className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-full text-muted-foreground">
                      {loadingStates[jobId] || video.status === 'processing' ? (
                        <div className="text-center space-y-2">
                          <Clock className="h-8 w-8 animate-spin mx-auto" />
                          <p className="text-sm">Processing...</p>
                        </div>
                      ) : video.status === 'error' ? (
                        <div className="text-center space-y-2">
                          <AlertTriangle className="h-8 w-8 text-red-500 mx-auto" />
                          <p className="text-sm text-red-500">Error processing video</p>
                        </div>
                      ) : (
                        <div className="text-center space-y-2">
                          <Clock className="h-8 w-8 mx-auto" />
                          <p className="text-sm">Waiting to start...</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Processing Progress */}
                {video.status === 'processing' && video.progress !== undefined && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span>Processing</span>
                      <span>{video.progress}%</span>
                    </div>
                    <Progress value={video.progress} className="w-full" />
                  </div>
                )}

                {/* Violence Detection Results */}
                {video.status === 'completed' && (
                  <div className="space-y-2">
                    {video.has_violence && video.segments?.length > 0 && (
                      <div className="text-xs space-y-1">
                        <p className="font-medium text-red-600">
                          Violence detected at:
                        </p>
                        {video.segments.slice(0, 3).map((segment, idx) => (
                          <p key={idx} className="text-muted-foreground">
                            {Math.round(segment.start)}s - {Math.round(segment.end)}s 
                            ({(segment.confidence * 100).toFixed(1)}%)
                          </p>
                        ))}
                        {video.segments.length > 3 && (
                          <p className="text-muted-foreground">
                            +{video.segments.length - 3} more segments
                          </p>
                        )}
                      </div>
                    )}
                    
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-muted-foreground">
                        Job ID: {jobId}
                      </span>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => downloadResults(jobId)}
                        className="h-6 px-2 text-xs"
                      >
                        <Download className="h-3 w-3" />
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* Summary Stats */}
      {Object.keys(videos).length > 0 && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold">{Object.keys(videos).length}</div>
              <div className="text-sm text-muted-foreground">Total Videos</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-green-600">
                {Object.values(videos).filter(v => v.status === 'completed' && !v.has_violence).length}
              </div>
              <div className="text-sm text-muted-foreground">Safe Videos</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-red-600">
                {Object.values(videos).filter(v => v.status === 'completed' && v.has_violence).length}
              </div>
              <div className="text-sm text-muted-foreground">Violence Detected</div>
            </CardContent>
          </Card>
          
          <Card>
            <CardContent className="p-4 text-center">
              <div className="text-2xl font-bold text-blue-600">
                {Object.values(videos).filter(v => v.status === 'processing').length}
              </div>
              <div className="text-sm text-muted-foreground">Processing</div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

export default MultiVideoGridViewer