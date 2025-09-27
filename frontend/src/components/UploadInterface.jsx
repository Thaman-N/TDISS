import React, { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import MetallicPaint from '@/components/react-bits/Animations/MetallicPaint/MetallicPaint.jsx'
import {
  Upload,
  FileVideo,
  AlertCircle,
  CheckCircle,
  Loader2,
  FolderOpen,
  X,
  ArrowRight,
  Eye
} from 'lucide-react'

const ALLOWED_FORMATS = ['mp4', 'avi', 'mov', 'mkv']
const MAX_FILE_SIZE = 4*500 * 1024 * 1024 // 500MB changed to 2 GB

const UploadInterface = () => {
  const navigate = useNavigate()
  const [uploadState, setUploadState] = useState({
    isUploading: false,
    uploadProgress: 0,
    uploadedFiles: [], // Changed to array for multiple files
    localPath: '',
    inputMethod: 'upload', // 'upload' or 'path'
    uploadComplete: false,
    jobIds: [] // Changed to array for multiple job IDs
  })

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0]
      if (rejection.errors.some(e => e.code === 'file-too-large')) {
        toast.error('File too large', {
          description: 'Maximum file size is 2GB'
        })
      } else if (rejection.errors.some(e => e.code === 'file-invalid-type')) {
        toast.error('Invalid file type', {
          description: `Supported formats: ${ALLOWED_FORMATS.join(', ').toUpperCase()}`
        })
      }
      return
    }

    if (acceptedFiles.length > 0) {
      setUploadState(prev => ({
        ...prev,
        uploadedFiles: [...prev.uploadedFiles, ...acceptedFiles],
        inputMethod: 'upload'
      }))
      toast.success(`${acceptedFiles.length} file(s) selected`, {
        description: acceptedFiles.length === 1 
          ? `${acceptedFiles[0].name} (${formatFileSize(acceptedFiles[0].size)})`
          : `${acceptedFiles.length} videos ready for processing`
      })
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ALLOWED_FORMATS.map(format => `.${format}`)
    },
    maxSize: MAX_FILE_SIZE,
    multiple: true
  })

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const handleSubmit = async () => {
  if (!uploadState.uploadedFiles.length && !uploadState.localPath.trim()) {
    toast.error('No input provided', {
      description: 'Please upload files or provide a local path'
    })
    return
  }

  setUploadState(prev => ({ ...prev, isUploading: true, uploadProgress: 0 }))

  try {
    const jobIds = []
    let totalFiles = uploadState.inputMethod === 'upload' ? uploadState.uploadedFiles.length : 1
    let completedFiles = 0

    const progressInterval = setInterval(() => {
      setUploadState(prev => ({
        ...prev,
        uploadProgress: Math.min(prev.uploadProgress + Math.random() * 10, 80)
      }))
    }, 300)

    if (uploadState.inputMethod === 'upload' && uploadState.uploadedFiles.length > 0) {
      if (uploadState.uploadedFiles.length === 1) {
        // Single file - use existing endpoint
        const formData = new FormData()
        formData.append('file', uploadState.uploadedFiles[0])

        const response = await fetch('/api/upload', {
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`Upload failed: ${response.status} ${errorText}`)
        }

        const result = await response.json()
        if (result.success && result.job_id) {
          jobIds.push(result.job_id)
        }
      } else {
        // Multiple files - use batch endpoint
        const formData = new FormData()
        for (const file of uploadState.uploadedFiles) {
          formData.append('files', file)  // Note: 'files' plural
        }

        const response = await fetch('/api/upload-batch', {  // Batch endpoint
          method: 'POST',
          body: formData
        })

        if (!response.ok) {
          const errorText = await response.text()
          throw new Error(`Batch upload failed: ${response.status} ${errorText}`)
        }

        const result = await response.json()
        if (result.success && result.job_ids) {
          jobIds.push(...result.job_ids)  // Spread array of job IDs
        }
      }
    } else if (uploadState.inputMethod === 'path' && uploadState.localPath.trim()) {
      // Single local path
      const formData = new FormData()
      formData.append('video_path', uploadState.localPath.trim())

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Upload failed: ${response.status} ${errorText}`)
      }

      const result = await response.json()
      if (result.success && result.job_id) {
        jobIds.push(result.job_id)
      }
    }

    clearInterval(progressInterval)

    if (jobIds.length > 0) {
      setUploadState(prev => ({
        ...prev,
        uploadProgress: 100,
        uploadComplete: true,
        jobIds: jobIds,
        isUploading: false
      }))

      toast.success('Upload successful!', {
        description: `${jobIds.length} video(s) are now being processed`
      })

      // Route based on number of videos
      setTimeout(() => {
        if (jobIds.length === 1) {
          navigate(`/dashboard?job=${jobIds[0]}`)
        } else {
          navigate(`/multi-analysis?jobs=${jobIds.join(',')}`)
        }
      }, 2000)
      
    } else {
      throw new Error('No valid uploads processed')
    }

  } catch (error) {
    console.error('Upload error:', error)
    toast.error('Upload failed', {
      description: error.message || 'Something went wrong'
    })
    setUploadState(prev => ({
      ...prev,
      isUploading: false,
      uploadProgress: 0
    }))
  }
}

  const clearSelection = () => {
    setUploadState({
      isUploading: false,
      uploadProgress: 0,
      uploadedFiles: [],
      localPath: '',
      inputMethod: 'upload',
      uploadComplete: false,
      jobIds: []
    })
  }

  const goToDashboard = () => {
    if (uploadState.jobIds.length === 1) {
      navigate(`/dashboard?job=${uploadState.jobIds[0]}`)
    } else if (uploadState.jobIds.length > 1) {
      navigate(`/multi-analysis?jobs=${uploadState.jobIds.join(',')}`)
    } else {
      navigate('/dashboard')
    }
  }

  return (
    <div className="container mx-auto px-4 py-12">
      <style>{`
        /* Enhanced Upload Interface Animations */
        .method-tab {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .method-tab::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
          transition: left 0.5s;
        }
        .method-tab:hover::before {
          left: 100%;
        }
        .method-tab:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        .method-tab:hover .tab-icon {
          transform: scale(1.1) rotate(5deg);
        }
        
        .upload-dropzone {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .upload-dropzone.drag-active {
          transform: scale(1.02);
          border-color: hsl(var(--primary));
          background: hsl(var(--primary) / 0.05);
          box-shadow: 0 0 30px hsl(var(--primary) / 0.2);
        }
        .upload-dropzone.has-file {
          border-color: hsl(var(--green-500) / 0.5);
          background: hsl(var(--green-500) / 0.05);
        }
        .upload-dropzone:not(.disabled):not(.drag-active):not(.has-file):hover {
          border-color: hsl(var(--primary) / 0.4);
          background: hsl(var(--primary) / 0.02);
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
        }
        .upload-dropzone:not(.disabled):not(.drag-active):not(.has-file):hover .upload-icon {
          transform: scale(1.1) rotate(5deg);
          color: hsl(var(--primary));
        }
        
        .file-display {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .file-display:hover .file-icon {
          transform: scale(1.2) rotate(10deg);
        }
        
        .remove-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .remove-button:hover {
          transform: scale(1.1);
          background: hsl(var(--destructive) / 0.1);
          color: hsl(var(--destructive));
        }
        
        .format-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .format-badge:hover {
          transform: translateY(-1px) scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.3);
          background: hsl(var(--primary) / 0.05);
        }
        
        .path-input-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .path-input-card::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 0;
          background: linear-gradient(135deg, hsl(var(--primary) / 0.03), transparent);
          transition: height 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          pointer-events: none;
        }
        .path-input-card:hover::before {
          height: 100%;
        }
        .path-input-card:hover {
          transform: translateY(-3px);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        .path-input-card:hover .card-icon {
          transform: scale(1.1) rotate(5deg);
          color: hsl(var(--primary));
        }
        
        .warning-card {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .warning-card:hover {
          transform: translateY(-1px);
          box-shadow: 0 6px 15px rgba(0, 0, 0, 0.05);
        }
        .warning-card:hover .warning-icon {
          transform: scale(1.1) rotate(5deg);
        }
        
        .progress-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .progress-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }
        
        .success-card {
          animation: successPulse 2s infinite;
        }
        
        @keyframes successPulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.02); }
        }
        
        .success-icon {
          animation: successSpin 0.8s ease-out;
        }
        
        @keyframes successSpin {
          0% { transform: scale(0) rotate(180deg); }
          100% { transform: scale(1) rotate(0deg); }
        }
        
        .dashboard-button {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .dashboard-button::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
          transition: left 0.6s;
        }
        .dashboard-button:hover::before {
          left: 100%;
        }
        .dashboard-button:hover {
          transform: translateY(-3px) scale(1.05);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
        }
        
        .secondary-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .secondary-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.3);
        }
        
        .start-button {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .start-button:hover {
          transform: translateY(-3px) scale(1.05);
          box-shadow: 0 12px 30px hsl(var(--primary) / 0.3);
        }
        .start-button:hover .start-icon {
          transform: translateX(4px);
        }
        
        .clear-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .clear-button:hover {
          transform: translateY(-2px);
          border-color: hsl(var(--destructive) / 0.3);
          color: hsl(var(--destructive));
        }
        
        /* Icon animations */
        .tab-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .upload-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .file-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .card-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .warning-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .start-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
      `}</style>
      
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Upload Video for Analysis
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload your video file or provide a local path to start AI-powered aggression detection
          </p>
        </div>

        {/* Success State */}
        {uploadState.uploadComplete && (
          <Card className="mb-6 border-green-200 bg-green-50 dark:bg-green-950/20 success-card">
            <CardContent className="pt-6">
              <div className="text-center space-y-4">
                <CheckCircle className="h-16 w-16 text-green-500 mx-auto success-icon" />
                <div>
                  <h3 className="text-xl font-semibold text-green-800 dark:text-green-200">
                    Upload Successful!
                  </h3>
                  <p className="text-green-700 dark:text-green-300 mt-2">
                    Your video is now being processed. You'll be redirected to the dashboard in a few seconds.
                  </p>
                </div>
                <div className="flex gap-3 justify-center">
                  <Button onClick={goToDashboard} className="bg-green-600 hover:bg-green-700 dashboard-button">
                    <Eye className="h-4 w-4 mr-2" />
                    View Progress Now
                  </Button>
                  <Button variant="outline" onClick={clearSelection} className="secondary-button">
                    Upload Another Video
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Input Method Tabs - Hide when upload is complete */}
        {!uploadState.uploadComplete && (
          <>
            <div className="flex gap-2 mb-6">
              <Button
                variant={uploadState.inputMethod === 'upload' ? 'default' : 'outline'}
                onClick={() => setUploadState(prev => ({ ...prev, inputMethod: 'upload' }))}
                className="flex-1 method-tab"
                disabled={uploadState.isUploading}
              >
                <Upload className="h-4 w-4 mr-2 tab-icon" />
                File Upload
              </Button>
              <Button
                variant={uploadState.inputMethod === 'path' ? 'default' : 'outline'}
                onClick={() => setUploadState(prev => ({ ...prev, inputMethod: 'path' }))}
                className="flex-1 method-tab"
                disabled={uploadState.isUploading}
              >
                <FolderOpen className="h-4 w-4 mr-2 tab-icon" />
                Local Path
              </Button>
            </div>

            {uploadState.inputMethod === 'upload' ? (
              /* File Upload Section */
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <FileVideo className="h-5 w-5 mr-2 card-icon" />
                    Video Upload
                  </CardTitle>
                  <CardDescription>
                    Drag and drop your video file or click to browse
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer upload-dropzone ${
                      uploadState.isUploading
                        ? 'disabled border-muted-foreground/25 cursor-not-allowed opacity-50'
                        : isDragActive
                        ? 'drag-active'
                        : uploadState.uploadedFiles.length > 0
                        ? 'has-file'
                        : 'border-muted-foreground/25'
                    }`}
                  >
                    <input {...getInputProps()} disabled={uploadState.isUploading} />
                    
                    {uploadState.uploadedFiles.length > 0 ? (
                      <div className="space-y-4 file-display">
                        <CheckCircle className="h-12 w-12 text-green-500 mx-auto file-icon" />
                        <div>
                          <p className="font-medium">
                            {uploadState.uploadedFiles.length} file(s) selected
                          </p>
                          <div className="text-sm text-muted-foreground space-y-1">
                            {uploadState.uploadedFiles.map((file, index) => (
                              <div key={index} className="flex items-center justify-between">
                                <span className="truncate max-w-48">{file.name}</span>
                                <span>{formatFileSize(file.size)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        {!uploadState.isUploading && (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation()
                              setUploadState(prev => ({ ...prev, uploadedFiles: [] }))
                            }}
                            className="remove-button"
                          >
                            <X className="h-4 w-4 mr-2" />
                            Clear All
                          </Button>
                        )}
                      </div>
                    ) : (
                      <div className="space-y-4">
                        <Upload className="h-12 w-12 text-muted-foreground mx-auto upload-icon" />
                        <div>
                          <p className="text-lg font-medium">
                            {isDragActive ? 'Drop your video here' : 'Upload a video file'}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            or click to browse your files
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* File Requirements */}
                  <div className="mt-4 space-y-2">
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline" className="format-badge">
                        Max size: 2GB
                      </Badge>
                      {ALLOWED_FORMATS.map(format => (
                        <Badge key={format} variant="outline" className="format-badge">
                          {format.toUpperCase()}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ) : (
              /* Local Path Section */
              <Card className="mb-6 path-input-card">
                <CardHeader>
                  <CardTitle className="flex items-center">
                    <FolderOpen className="h-5 w-5 mr-2 card-icon" />
                    Local File Path
                  </CardTitle>
                  <CardDescription>
                    Provide the absolute path to your video file on the server
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="localPath">File Path</Label>
                    <Input
                      id="localPath"
                      type="text"
                      placeholder="/path/to/your/video.mp4"
                      value={uploadState.localPath}
                      onChange={(e) => setUploadState(prev => ({ 
                        ...prev, 
                        localPath: e.target.value 
                      }))}
                      className="mt-1"
                      disabled={uploadState.isUploading}
                    />
                  </div>
                  
                  <div className="flex items-start space-x-2 p-3 bg-yellow-50 dark:bg-yellow-950/20 rounded-lg warning-card">
                    <AlertCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400 mt-0.5 flex-shrink-0 warning-icon" />
                    <div className="text-sm text-yellow-800 dark:text-yellow-200">
                      <strong>Note:</strong> The file must exist on the server and be accessible to the application.
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Upload Progress */}
            {uploadState.isUploading && (
              <Card className="mb-6 progress-card">
                <CardContent className="pt-6">
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>Uploading and starting analysis...</span>
                      <span>{Math.round(uploadState.uploadProgress)}%</span>
                    </div>
                    <Progress value={uploadState.uploadProgress} className="w-full" />
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Submit Button */}
            <div className="text-center">
              {uploadState.isUploading ? (
                <Button disabled size="lg" className="px-12">
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Processing...
                </Button>
              ) : (
                <Button 
                  onClick={handleSubmit}
                  disabled={!uploadState.uploadedFiles.length && !uploadState.localPath.trim()}
                  size="lg" 
                  className="px-12 start-button"
                >
                  Start Analysis
                  <ArrowRight className="h-5 w-5 ml-2 start-icon" />
                </Button>
              )}
            </div>

            {(uploadState.uploadedFiles.length > 0 || uploadState.localPath) && !uploadState.isUploading && (
              <div className="text-center mt-4">
                <Button variant="outline" onClick={clearSelection} className="clear-button">
                  Clear Selection
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default UploadInterface
