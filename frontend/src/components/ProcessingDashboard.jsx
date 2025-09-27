import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useWebSocket } from '@/contexts/WebSocketContext'
import { 
  Eye, 
  Clock, 
  CheckCircle, 
  AlertCircle, 
  Loader2, 
  MoreVertical,
  RefreshCw,
  Trash2,
  Download,
  Play,
  Calendar,
  TrendingUp,
  BarChart3,
  Wifi,
  WifiOff
} from 'lucide-react'
import { format } from 'date-fns'

const ProcessingDashboard = () => {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { isConnected: wsConnected, registerJobUpdateCallback } = useWebSocket()
  const [jobs, setJobs] = useState([])
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [jobsLoading, setJobsLoading] = useState(false)
  
  // Track which jobs have been updated to avoid unnecessary re-renders
  const jobUpdateTimestamps = useRef({})
  const refreshTimeoutRef = useRef(null)

  // Highlighted job from URL params
  const highlightedJobId = searchParams.get('job')

  const fetchJobs = useCallback(async (showLoading = true) => {
    if (showLoading) setJobsLoading(true)
    try {
      const response = await fetch('/api/jobs')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      
      // Update jobs with proper ordering (backend now handles this)
      setJobs(data || [])
    } catch (error) {
      console.error('Failed to fetch jobs:', error)
      if (showLoading) {
        toast.error('Failed to load jobs', {
          description: error.message
        })
      }
    } finally {
      if (showLoading) setJobsLoading(false)
    }
  }, [])

  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch('/api/history')
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setHistory(data.history || [])
    } catch (error) {
      console.error('Failed to fetch history:', error)
      toast.error('Failed to load history', {
        description: error.message
      })
    }
  }, [])

  // Optimized job update handler
  const handleJobUpdate = useCallback((jobId, jobData) => {
    const now = Date.now()
    const lastUpdate = jobUpdateTimestamps.current[jobId] || 0
    
    // Throttle updates to avoid excessive re-renders (min 500ms between updates)
    if (now - lastUpdate < 500) {
      return
    }
    
    jobUpdateTimestamps.current[jobId] = now

    setJobs(prevJobs => {
      const existingJobIndex = prevJobs.findIndex(job => job.id === jobId)
      
      if (existingJobIndex >= 0) {
        // Update existing job without re-rendering the entire list
        const updatedJobs = [...prevJobs]
        const existingJob = updatedJobs[existingJobIndex]
        
        // Only update if there are actual changes
        const hasChanges = 
          existingJob.status !== jobData.status ||
          existingJob.progress !== jobData.progress ||
          existingJob.message !== jobData.message
        
        if (hasChanges) {
          updatedJobs[existingJobIndex] = { 
            ...existingJob, 
            ...jobData,
            // Ensure we keep the original timestamp for ordering
            timestamp: existingJob.timestamp || jobData.timestamp
          }
          return updatedJobs
        }
        return prevJobs // No changes, don't trigger re-render
      } else {
        // Add new job at the beginning (most recent first)
        return [{ id: jobId, ...jobData }, ...prevJobs]
      }
    })

    // Handle completion/error notifications
    if (jobData.status === 'completed') {
      // Refresh history after a brief delay
      setTimeout(() => {
        fetchHistory()
      }, 1000)
      
      toast.success('Analysis complete!', {
        description: `${jobData.filename} has been processed`,
        action: {
          label: 'View Results',
          onClick: () => navigate(`/results/${jobId}`)
        }
      })
    } else if (jobData.status === 'error') {
      toast.error('Analysis failed', {
        description: `${jobData.filename}: ${jobData.message}`
      })
    }
  }, [navigate, fetchHistory])

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([fetchJobs(false), fetchHistory()])
      setLoading(false)
    }
    
    loadData()
  }, [fetchJobs, fetchHistory])

  useEffect(() => {
    // Register for job updates via WebSocket
    const unregister = registerJobUpdateCallback(handleJobUpdate)
    
    return unregister
  }, [registerJobUpdateCallback, handleJobUpdate])

  // Periodic refresh for active jobs (every 10 seconds)
  useEffect(() => {
    const refreshActiveJobs = () => {
      const hasActiveJobs = jobs.some(job => ['queued', 'processing'].includes(job.status))
      if (hasActiveJobs && wsConnected) {
        // Only refresh jobs, not history
        fetchJobs(false)
      }
    }

    // Set up periodic refresh
    const interval = setInterval(refreshActiveJobs, 10000)
    
    return () => clearInterval(interval)
  }, [jobs, wsConnected, fetchJobs])

  const getStatusIcon = (status) => {
    switch (status) {
      case 'queued':
        return <Clock className="h-4 w-4 text-yellow-500" />
      case 'processing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status) => {
    const variants = {
      queued: 'outline',
      processing: 'default',
      completed: 'default',
      error: 'destructive'
    }
    return (
      <Badge variant={variants[status] || 'outline'} className="capitalize">
        {status}
      </Badge>
    )
  }

  const formatDuration = (seconds) => {
    if (seconds < 60) return `${seconds}s`
    const minutes = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${minutes}m ${secs}s`
  }

  const handleRefresh = useCallback(async () => {
    // Clear any existing timeout
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current)
    }
    
    await Promise.all([fetchJobs(), fetchHistory()])
  }, [fetchJobs, fetchHistory])

  // Memoized JobCard to prevent unnecessary re-renders
  const JobCard = React.memo(({ job, isHighlighted = false }) => (
    <Card className={`${isHighlighted ? 'ring-2 ring-primary highlighted-card' : ''} job-card transition-all hover:shadow-md`}>
      <style>{`
        .job-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
          position: relative;
          overflow: hidden;
        }
        .job-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 2px;
          background: linear-gradient(90deg, transparent, hsl(var(--primary)), transparent);
          transition: left 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .job-card:hover::before {
          left: 100%;
        }
        .job-card:hover {
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        .job-card:hover .job-title {
          color: hsl(var(--primary));
          transform: translateX(2px);
        }
        .job-card:hover .job-thumbnail {
          transform: scale(1.05);
        }
        .job-card:hover .status-icon {
          transform: scale(1.2) rotate(5deg);
        }
        .job-card:hover .action-button {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .highlighted-card {
          background: linear-gradient(135deg, hsl(var(--primary) / 0.05), hsl(var(--primary) / 0.02));
          border-color: hsl(var(--primary) / 0.3);
          box-shadow: 0 0 20px hsl(var(--primary) / 0.1);
        }
        
        .job-title {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .job-thumbnail {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .status-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .action-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
      `}</style>
      
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="status-icon">{getStatusIcon(job.status)}</span>
            <CardTitle className="text-lg job-title">{job.filename}</CardTitle>
          </div>
          <div className="flex items-center space-x-2">
            {getStatusBadge(job.status)}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="dropdown-trigger">
                  <MoreVertical className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {job.status === 'completed' && (
                  <DropdownMenuItem onClick={() => navigate(`/results/${job.id}`)} className="dropdown-item">
                    <Eye className="h-4 w-4 mr-2" />
                    View Results
                  </DropdownMenuItem>
                )}
                <DropdownMenuItem onClick={handleRefresh} className="dropdown-item">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Refresh
                </DropdownMenuItem>
                <DropdownMenuItem className="text-red-600 dropdown-item">
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <CardDescription className="flex items-center space-x-4">
          <span className="flex items-center">
            <Calendar className="h-3 w-3 mr-1" />
            {job.timestamp ? format(new Date(job.timestamp), 'MMM dd, HH:mm') : 'No date'}
          </span>
          {job.metadata && (
            <span className="flex items-center">
              <Play className="h-3 w-3 mr-1" />
              {job.metadata.duration_formatted}
            </span>
          )}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="pt-0">
        {job.status === 'processing' && (
          <div className="space-y-2 mb-4">
            <div className="flex justify-between text-sm">
              <span>{job.message}</span>
              <span>{job.progress}%</span>
            </div>
            <Progress value={job.progress} className="w-full" />
          </div>
        )}

        {job.thumbnail && (
          <div className="mb-4 overflow-hidden rounded-md">
            <img 
              src={job.thumbnail} 
              alt="Video thumbnail"
              className="w-full h-32 object-cover job-thumbnail"
            />
          </div>
        )}

        {job.result && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">Aggression Detected:</span>
              <Badge variant={job.result.has_violence ? 'destructive' : 'default'}>
                {job.result.has_violence ? 'Yes' : 'No'}
              </Badge>
            </div>
            {job.result.has_violence && (
              <div className="text-sm text-muted-foreground">
                {job.result.segments?.length || 0} segment(s) • 
                {job.result.violence_percentage?.toFixed(1)}% of video
              </div>
            )}
          </div>
        )}

        {job.status === 'completed' && (
          <Button 
            className="w-full mt-4 action-button" 
            onClick={() => navigate(`/results/${job.id}`)}
          >
            <Eye className="h-4 w-4 mr-2" />
            View Detailed Results
          </Button>
        )}
      </CardContent>
    </Card>
  ))

  // Memoized HistoryCard to prevent unnecessary re-renders
  const HistoryCard = React.memo(({ item }) => (
    <Card className="history-card cursor-pointer" 
          onClick={() => navigate(`/results/${item.job_id}`)}>
      <style>{`
        .history-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .history-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 0;
          height: 100%;
          background: linear-gradient(90deg, hsl(var(--primary) / 0.05), transparent);
          transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .history-card:hover::before {
          width: 100%;
        }
        .history-card:hover {
          transform: translateY(-6px) scale(1.03);
          box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        .history-card:hover .history-title {
          color: hsl(var(--primary));
          transform: translateX(4px);
        }
        .history-card:hover .history-thumbnail {
          transform: scale(1.1);
        }
        .history-card:hover .history-badge {
          transform: scale(1.05);
        }
        
        .history-title {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .history-thumbnail {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .history-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
      `}</style>
      
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base history-title">{item.filename}</CardTitle>
          <Badge variant={item.has_violence ? 'destructive' : 'default'} className="history-badge">
            {item.has_violence ? 'Violence' : 'Safe'}
          </Badge>
        </div>
        <CardDescription>
          {item.timestamp ? format(new Date(item.timestamp), 'MMM dd, yyyy HH:mm') : 'No date'}
        </CardDescription>
      </CardHeader>
      
      <CardContent className="pt-0">
        {item.thumbnail && (
          <div className="overflow-hidden rounded-md mb-3">
            <img 
              src={item.thumbnail} 
              alt="Video thumbnail"
              className="w-full h-24 object-cover history-thumbnail"
            />
          </div>
        )}
        
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Confidence:</span>
            <div className="font-medium">{(item.overall_confidence * 100).toFixed(1)}%</div>
          </div>
          {item.has_violence && (
            <div>
              <span className="text-muted-foreground">Duration:</span>
              <div className="font-medium">{formatDuration(Math.round(item.violence_duration))}</div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  ))

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12 text-center">
        <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
        <p>Loading dashboard...</p>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <style>{`
        /* Enhanced Dashboard Animations */
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
        
        .connection-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .connection-badge:hover {
          transform: scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .refresh-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .refresh-button::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
          transition: left 0.5s;
        }
        .refresh-button:hover::before {
          left: 100%;
        }
        .refresh-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        .refresh-button:hover .refresh-icon {
          transform: rotate(180deg);
        }
        
        .tab-trigger {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .tab-trigger:hover {
          transform: translateY(-1px);
        }
        
        .upload-button {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .upload-button:hover {
          transform: translateY(-3px) scale(1.05);
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
        }
        
        .dropdown-trigger {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .dropdown-trigger:hover {
          transform: scale(1.1);
          background: hsl(var(--primary) / 0.1);
        }
        
        .dropdown-item {
          transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .dropdown-item:hover {
          transform: translateX(4px);
          background: hsl(var(--primary) / 0.05);
        }
        
        .stat-value {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .refresh-icon {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .empty-state-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .empty-state-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
        }
      `}</style>
      
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-4xl font-bold mb-2">Processing Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your video analysis jobs in real-time
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm connection-badge ${
            wsConnected ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          }`}>
            {wsConnected ? (
              <Wifi className="w-3 h-3" />
            ) : (
              <WifiOff className="w-3 h-3" />
            )}
            {wsConnected ? 'Live Updates' : 'Disconnected'}
          </div>
          
          <Button 
            variant="outline" 
            onClick={handleRefresh} 
            className="refresh-button"
            disabled={jobsLoading}
          >
            <RefreshCw className={`h-4 w-4 mr-2 refresh-icon ${jobsLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">
              {jobs.filter(job => ['queued', 'processing'].includes(job.status)).length}
            </div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Completed Today</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">
              {jobs.filter(job => job.status === 'completed').length}
            </div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Total Processed</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">{history.length}</div>
          </CardContent>
        </Card>
        
        <Card className="stat-card">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Aggression Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold stat-value">
              {history.length > 0 
                ? ((history.filter(h => h.has_violence).length / history.length) * 100).toFixed(1)
                : 0}%
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="active" className="space-y-6">
        <TabsList>
          <TabsTrigger value="active" className="tab-trigger">Active Jobs</TabsTrigger>
          <TabsTrigger value="history" className="tab-trigger">History</TabsTrigger>
        </TabsList>

        <TabsContent value="active" className="space-y-6">
          {jobs.length === 0 ? (
            <Card className="empty-state-card">
              <CardContent className="pt-6 text-center">
                <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg font-medium mb-2">No active jobs</p>
                <p className="text-muted-foreground mb-4">
                  Upload a video to start processing
                </p>
                <Button onClick={() => navigate('/upload')} className="upload-button">
                  Upload Video
                </Button>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {jobs.map(job => (
                <JobCard 
                  key={job.id} 
                  job={job} 
                  isHighlighted={job.id === highlightedJobId}
                />
              ))}
            </div>
          )}
        </TabsContent>

        <TabsContent value="history" className="space-y-6">
          {history.length === 0 ? (
            <Card className="empty-state-card">
              <CardContent className="pt-6 text-center">
                <TrendingUp className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg font-medium mb-2">No history yet</p>
                <p className="text-muted-foreground">
                  Your completed analyses will appear here
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              {history.slice(0, 12).map(item => (
                <HistoryCard key={item.job_id} item={item} />
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default ProcessingDashboard