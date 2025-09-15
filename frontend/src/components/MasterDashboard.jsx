import React, { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { 
  ChartContainer, 
  ChartTooltip, 
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent 
} from '@/components/ui/chart'
import { 
  Breadcrumb,
  BreadcrumbList,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbSeparator,
  BreadcrumbPage
} from '@/components/ui/breadcrumb'
import { useWebSocket } from '@/contexts/WebSocketContext'
import { 
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer
} from 'recharts'
import { 
  Activity,
  AlertCircle,
  BarChart3,
  Calendar,
  CheckCircle,
  Clock,
  Eye,
  FileVideo,
  Home,
  Loader2,
  Monitor,
  Play,
  RefreshCw,
  Shield,
  TrendingUp,
  Users,
  Video,
  Wifi,
  WifiOff,
  Zap
} from 'lucide-react'
import { format, subDays, parseISO } from 'date-fns'

const MasterDashboard = () => {
  const navigate = useNavigate()
  const { isConnected: wsConnected, registerJobUpdateCallback } = useWebSocket()
  
  // State management
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState({})
  const [jobs, setJobs] = useState([])
  const [streams, setStreams] = useState([])
  const [history, setHistory] = useState([])
  const [events, setEvents] = useState([])
  const [streamStats, setStreamStats] = useState([])
  const [chartData, setChartData] = useState([])
  const [refreshing, setRefreshing] = useState(false)
  
  // Refs for preventing excessive re-renders
  const jobUpdateTimestamps = useRef({})
  const refreshTimeoutRef = useRef(null)

  // Fetch all dashboard data
  const fetchDashboardData = useCallback(async (showLoading = true) => {
    if (showLoading) setLoading(true)
    setRefreshing(true)
    
    try {
      const [
        statsRes,
        jobsRes,
        streamsRes,
        historyRes,
        eventsRes,
        streamStatsRes
      ] = await Promise.allSettled([
        fetch('/api/stats'),
        fetch('/api/jobs'),
        fetch('/api/streams'),
        fetch('/api/history'),
        fetch('/api/events'),
        fetch('/api/stream-stats')
      ])

      // Process stats
      if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        const statsData = await statsRes.value.json()
        setStats(statsData)
      }

      // Process jobs
      if (jobsRes.status === 'fulfilled' && jobsRes.value.ok) {
        const jobsData = await jobsRes.value.json()
        setJobs(jobsData || [])
      }

      // Process streams
      if (streamsRes.status === 'fulfilled' && streamsRes.value.ok) {
        const streamsData = await streamsRes.value.json()
        setStreams(streamsData.streams || [])
      }

      // Process history
      if (historyRes.status === 'fulfilled' && historyRes.value.ok) {
        const historyData = await historyRes.value.json()
        setHistory(historyData.history || [])
      }

      // Process events
      if (eventsRes.status === 'fulfilled' && eventsRes.value.ok) {
        const eventsData = await eventsRes.value.json()
        setEvents(eventsData.events || [])
      }

      // Process stream stats
      if (streamStatsRes.status === 'fulfilled' && streamStatsRes.value.ok) {
        const streamStatsData = await streamStatsRes.value.json()
        setStreamStats(streamStatsData.stats || [])
      }

    } catch (error) {
      console.error('Failed to fetch dashboard data:', error)
      toast.error('Failed to load dashboard data', {
        description: error.message
      })
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  // Generate chart data from history and events
  const generateChartData = useCallback(() => {
    if (!history.length && !events.length) return

    // Create detection timeline data (last 7 days)
    const days = Array.from({ length: 7 }, (_, i) => {
      const date = subDays(new Date(), 6 - i)
      return {
        date: format(date, 'MMM dd'),
        detections: 0,
        uploads: 0,
        streams: 0
      }
    })

    // Process history data
    history.forEach(item => {
      if (item.timestamp) {
        const itemDate = format(parseISO(item.timestamp), 'MMM dd')
        const dayData = days.find(d => d.date === itemDate)
        if (dayData) {
          dayData.uploads += 1
          if (item.has_violence) {
            dayData.detections += 1
          }
        }
      }
    })

    // Process events data (for streams)
    events.forEach(event => {
      if (event.timestamp) {
        const eventDate = format(parseISO(event.timestamp), 'MMM dd')
        const dayData = days.find(d => d.date === eventDate)
        if (dayData) {
          dayData.streams += 1
          dayData.detections += 1
        }
      }
    })

    setChartData(days)
  }, [history, events])

  // Handle job updates via WebSocket
  const handleJobUpdate = useCallback((jobId, jobData) => {
    const now = Date.now()
    const lastUpdate = jobUpdateTimestamps.current[jobId] || 0
    
    // Throttle updates to avoid excessive re-renders
    if (now - lastUpdate < 1000) {
      return
    }
    
    jobUpdateTimestamps.current[jobId] = now

    setJobs(prevJobs => {
      const existingJobIndex = prevJobs.findIndex(job => job.id === jobId)
      
      if (existingJobIndex >= 0) {
        const updatedJobs = [...prevJobs]
        updatedJobs[existingJobIndex] = { 
          ...updatedJobs[existingJobIndex], 
          ...jobData
        }
        return updatedJobs
      } else {
        return [{ id: jobId, ...jobData }, ...prevJobs]
      }
    })

    // Show completion notifications
    if (jobData.status === 'completed') {
      toast.success('Analysis complete!', {
        description: `${jobData.filename} has been processed`,
        action: {
          label: 'View Results',
          onClick: () => navigate(`/results/${jobId}`)
        }
      })
      
      // Refresh dashboard data after completion
      setTimeout(() => {
        fetchDashboardData(false)
      }, 1000)
    } else if (jobData.status === 'error') {
      toast.error('Analysis failed', {
        description: `${jobData.filename}: ${jobData.message}`
      })
    }
  }, [navigate, fetchDashboardData])

  // Effects
  useEffect(() => {
    fetchDashboardData()
  }, [fetchDashboardData])

  useEffect(() => {
    generateChartData()
  }, [generateChartData])

  useEffect(() => {
    // Register for job updates via WebSocket
    const unregister = registerJobUpdateCallback(handleJobUpdate)
    return unregister
  }, [registerJobUpdateCallback, handleJobUpdate])

  // Periodic refresh for active content
  useEffect(() => {
    const interval = setInterval(() => {
      const hasActiveJobs = jobs.some(job => ['queued', 'processing'].includes(job.status))
      const hasActiveStreams = streams.some(stream => stream.status === 'active')
      
      if ((hasActiveJobs || hasActiveStreams) && wsConnected) {
        fetchDashboardData(false)
      }
    }, 15000) // 15 seconds

    return () => clearInterval(interval)
  }, [jobs, streams, wsConnected, fetchDashboardData])

  // Utility functions
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
      case 'active':
        return <Activity className="h-4 w-4 text-green-500" />
      default:
        return <Clock className="h-4 w-4 text-gray-500" />
    }
  }

  const getStatusBadge = (status) => {
    const variants = {
      queued: 'outline',
      processing: 'default',
      completed: 'default',
      error: 'destructive',
      active: 'default',
      inactive: 'outline'
    }
    return (
      <Badge variant={variants[status] || 'outline'} className="capitalize">
        {status}
      </Badge>
    )
  }

  // Calculate derived stats
  const activeJobs = jobs.filter(job => ['queued', 'processing'].includes(job.status)).length
  const completedJobs = jobs.filter(job => job.status === 'completed').length
  const activeStreams = streams.filter(stream => stream.status === 'active').length
  const totalDetections = history.filter(h => h.has_violence).length + events.length
  const detectionRate = history.length > 0 ? ((totalDetections / (history.length + events.length)) * 100).toFixed(1) : '0'

  // Chart configurations
  const chartConfig = {
    detections: {
      label: "Violence Detections",
      color: "hsl(var(--chart-1))",
    },
    uploads: {
      label: "Video Uploads",
      color: "hsl(var(--chart-2))",
    },
    streams: {
      label: "Stream Events",
      color: "hsl(var(--chart-3))",
    }
  }

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-12">
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="text-center">
            <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
            <p className="text-muted-foreground">Loading master dashboard...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-6">
      {/* Breadcrumb Navigation */}
      <Breadcrumb>
        <BreadcrumbList>
          <BreadcrumbItem>
            <BreadcrumbLink onClick={() => navigate('/')} className="flex items-center cursor-pointer">
              <Home className="h-4 w-4" />
            </BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem>
            <BreadcrumbPage>Master Dashboard</BreadcrumbPage>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2">Master Dashboard</h1>
          <p className="text-muted-foreground">
            Comprehensive overview of your TDISS violence detection system
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
            wsConnected ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
          }`}>
            {wsConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
            {wsConnected ? 'Live Updates' : 'Disconnected'}
          </div>
          
          <Button 
            variant="outline" 
            onClick={() => fetchDashboardData()} 
            disabled={refreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Stats Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Jobs</CardTitle>
            <Loader2 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{activeJobs}</div>
            <p className="text-xs text-muted-foreground">
              Currently processing
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Live Streams</CardTitle>
            <Monitor className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{activeStreams}</div>
            <p className="text-xs text-muted-foreground">
              {streams.length} total configured
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Detections</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{totalDetections}</div>
            <p className="text-xs text-muted-foreground">
              Violence incidents detected
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Detection Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">{detectionRate}%</div>
            <p className="text-xs text-muted-foreground">
              Of analyzed content
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Detection Trends Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Detection Trends (Last 7 Days)</CardTitle>
            <CardDescription>
              Violence detections from uploads and live streams
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ChartContainer config={chartConfig}>
              <AreaChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <YAxis 
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Area
                  type="monotone"
                  dataKey="detections"
                  stackId="1"
                  stroke="var(--color-detections)"
                  fill="var(--color-detections)"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="uploads"
                  stackId="1"
                  stroke="var(--color-uploads)"
                  fill="var(--color-uploads)"
                  fillOpacity={0.6}
                />
                <Area
                  type="monotone"
                  dataKey="streams"
                  stackId="1"
                  stroke="var(--color-streams)"
                  fill="var(--color-streams)"
                  fillOpacity={0.6}
                />
              </AreaChart>
            </ChartContainer>
          </CardContent>
        </Card>

        {/* System Performance Chart */}
        <Card>
          <CardHeader>
            <CardTitle>System Performance</CardTitle>
            <CardDescription>
              Job completion and processing metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <CheckCircle className="h-4 w-4 text-green-500" />
                  <span className="text-sm font-medium">Completed Jobs</span>
                </div>
                <div className="text-sm font-medium">{completedJobs}</div>
              </div>
              <Progress value={(completedJobs / Math.max(jobs.length, 1)) * 100} className="w-full" />
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-blue-500" />
                  <span className="text-sm font-medium">Success Rate</span>
                </div>
                <div className="text-sm font-medium">
                  {jobs.length > 0 ? ((jobs.filter(j => j.status === 'completed').length / jobs.length) * 100).toFixed(1) : 0}%
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-yellow-500" />
                  <span className="text-sm font-medium">Avg Processing</span>
                </div>
                <div className="text-sm font-medium">&lt;20ms</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="processing" className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="processing">Live Processing</TabsTrigger>
          <TabsTrigger value="streams">Stream Monitor</TabsTrigger>
          <TabsTrigger value="results">Recent Results</TabsTrigger>
          <TabsTrigger value="incidents">Incidents</TabsTrigger>
        </TabsList>

        {/* Live Processing Tab */}
        <TabsContent value="processing" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Activity className="h-5 w-5" />
                <span>Active Processing Jobs</span>
                {activeJobs > 0 && <Badge variant="secondary">{activeJobs}</Badge>}
              </CardTitle>
              <CardDescription>
                Monitor real-time video analysis progress
              </CardDescription>
            </CardHeader>
            <CardContent>
              {jobs.length === 0 ? (
                <div className="text-center py-8">
                  <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium mb-2">No active jobs</p>
                  <p className="text-muted-foreground mb-4">Upload a video to start processing</p>
                  <Button onClick={() => navigate('/upload')}>
                    Upload Video
                  </Button>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                  {jobs.slice(0, 6).map(job => (
                    <Card key={job.id} className="cursor-pointer hover:shadow-md transition-shadow"
                          onClick={() => job.status === 'completed' && navigate(`/results/${job.id}`)}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            {getStatusIcon(job.status)}
                            <CardTitle className="text-base">{job.filename}</CardTitle>
                          </div>
                          {getStatusBadge(job.status)}
                        </div>
                        {job.timestamp && (
                          <CardDescription>
                            {format(new Date(job.timestamp), 'MMM dd, HH:mm')}
                          </CardDescription>
                        )}
                      </CardHeader>
                      <CardContent>
                        {job.status === 'processing' && (
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>{job.message || 'Processing...'}</span>
                              <span>{job.progress || 0}%</span>
                            </div>
                            <Progress value={job.progress || 0} className="w-full" />
                          </div>
                        )}
                        {job.result && job.result.has_violence && (
                          <div className="flex items-center justify-between text-sm">
                            <span>Violence Detected</span>
                            <Badge variant="destructive">Alert</Badge>
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Stream Monitor Tab */}
        <TabsContent value="streams" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Monitor className="h-5 w-5" />
                <span>RTSP Stream Monitor</span>
                {activeStreams > 0 && <Badge variant="secondary">{activeStreams} Active</Badge>}
              </CardTitle>
              <CardDescription>
                Live stream status and recent detections
              </CardDescription>
            </CardHeader>
            <CardContent>
              {streams.length === 0 ? (
                <div className="text-center py-8">
                  <Video className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium mb-2">No streams configured</p>
                  <p className="text-muted-foreground mb-4">Add RTSP streams to monitor live feeds</p>
                  <Button onClick={() => navigate('/live-streams')}>
                    Manage Streams
                  </Button>
                </div>
              ) : (
                <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-4">
                  {streams.map(stream => (
                    <Card key={stream.id} className="cursor-pointer hover:shadow-md transition-shadow"
                          onClick={() => navigate(`/stream-fullscreen/${stream.id}`)}>
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-base">{stream.name}</CardTitle>
                          {getStatusBadge(stream.status)}
                        </div>
                        <CardDescription className="text-xs">
                          {stream.total_detections || 0} detections total
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        {stream.thumbnail_path && (
                          <div className="aspect-video bg-muted rounded-md mb-3 overflow-hidden">
                            <img 
                              src={stream.thumbnail_path} 
                              alt={`${stream.name} thumbnail`}
                              className="w-full h-full object-cover"
                            />
                          </div>
                        )}
                        <div className="flex items-center justify-between text-sm">
                          <span className="flex items-center space-x-1">
                            <Activity className="h-3 w-3" />
                            <span>{stream.status === 'active' ? 'Live' : 'Offline'}</span>
                          </span>
                          {stream.last_detection && (
                            <span className="text-muted-foreground">
                              Last: {format(new Date(stream.last_detection), 'HH:mm')}
                            </span>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Recent Results Tab */}
        <TabsContent value="results" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <FileVideo className="h-5 w-5" />
                <span>Recent Analysis Results</span>
              </CardTitle>
              <CardDescription>
                Latest completed video analysis results
              </CardDescription>
            </CardHeader>
            <CardContent>
              {history.length === 0 ? (
                <div className="text-center py-8">
                  <TrendingUp className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium mb-2">No analysis history</p>
                  <p className="text-muted-foreground">Completed analyses will appear here</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>File</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Date</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {history.slice(0, 10).map((item) => (
                      <TableRow key={item.job_id} className="cursor-pointer hover:bg-muted/50"
                               onClick={() => navigate(`/results/${item.job_id}`)}>
                        <TableCell className="font-medium">
                          <div className="flex items-center space-x-2">
                            <FileVideo className="h-4 w-4" />
                            <span>{item.filename}</span>
                          </div>
                        </TableCell>
                        <TableCell>
                          <Badge variant={item.has_violence ? 'destructive' : 'default'}>
                            {item.has_violence ? 'Violence' : 'Safe'}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <span className="font-mono">
                            {(item.overall_confidence * 100).toFixed(1)}%
                          </span>
                        </TableCell>
                        <TableCell>
                          {item.timestamp ? format(new Date(item.timestamp), 'MMM dd, HH:mm') : '-'}
                        </TableCell>
                        <TableCell>
                          <Button variant="ghost" size="sm">
                            <Eye className="h-4 w-4" />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Incidents Tab */}
        <TabsContent value="incidents" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <Shield className="h-5 w-5" />
                <span>Violence Incidents</span>
                {events.length > 0 && <Badge variant="destructive">{events.length}</Badge>}
              </CardTitle>
              <CardDescription>
                Detected violence events from all sources
              </CardDescription>
            </CardHeader>
            <CardContent>
              {events.length === 0 ? (
                <div className="text-center py-8">
                  <Shield className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg font-medium mb-2">No incidents detected</p>
                  <p className="text-muted-foreground">Violence detection events will appear here</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {events.slice(0, 8).map((event, index) => (
                    <Card key={`${event.id || index}`} className="cursor-pointer hover:shadow-md transition-shadow">
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <AlertCircle className="h-4 w-4 text-red-500" />
                            <CardTitle className="text-base">{event.filename || 'Stream Event'}</CardTitle>
                          </div>
                          <Badge variant="destructive">Violence</Badge>
                        </div>
                        <CardDescription>
                          {event.timestamp ? format(new Date(event.timestamp), 'MMM dd, yyyy HH:mm:ss') : 'No timestamp'}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Confidence:</span>
                            <div className="font-medium">{((event.confidence || 0) * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Duration:</span>
                            <div className="font-medium">{(event.duration || 0).toFixed(1)}s</div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Source:</span>
                            <div className="font-medium capitalize">{event.source_type || 'unknown'}</div>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Status:</span>
                            <div className="font-medium capitalize">{event.incident_status || 'completed'}</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default MasterDashboard