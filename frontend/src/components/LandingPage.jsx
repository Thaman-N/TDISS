import React, { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import LightRays from '@/components/react-bits/Backgrounds/LightRays/LightRays'
import MagicBento from '@/components/react-bits/Components/MagicBento/MagicBento'
import MetallicPaint from '@/components/react-bits/Animations/MetallicPaint/MetallicPaint'
import { 
  Shield, 
  Zap, 
  Eye, 
  Clock, 
  Upload, 
  Brain, 
  CheckCircle,
  ArrowRight,
  Play,
  Star
} from 'lucide-react'

const useScrollAnimation = () => {
  const [visibleElements, setVisibleElements] = useState(new Set())
  const observerRef = useRef(null)

  useEffect(() => {
    observerRef.current = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setVisibleElements(prev => new Set([...prev, entry.target.id]))
          }
        })
      },
      { threshold: 0.1, rootMargin: '0px 0px -50px 0px' }
    )

    return () => observerRef.current?.disconnect()
  }, [])

  const observeElement = (element) => {
    if (element && observerRef.current) {
      observerRef.current.observe(element)
    }
  }

  return { visibleElements, observeElement }
}

const AnimatedSection = ({ id, children, className = "", delay = 0 }) => {
  const ref = useRef(null)
  const { visibleElements, observeElement } = useScrollAnimation()
  
  useEffect(() => {
    observeElement(ref.current)
  }, [])

  const isVisible = visibleElements.has(id)

  return (
    <div
      id={id}
      ref={ref}
      className={`transition-all duration-1000 ease-out ${
        isVisible 
          ? 'opacity-100 translate-y-0' 
          : 'opacity-0 translate-y-12'
      } ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  )
}

const LandingPage = () => {
  const statsData = [
    { label: 'Stream/Playback Modes Supported', value: 'Stream & Playback', icon: Eye },
    { label: 'Processing Speed', value: '<2s', icon: Zap },
    { label: 'Videos used to train', value: '2000+', icon: Play },
    { label: 'Accuracy Rate', value: '84%', icon: CheckCircle }
  ]

  const bentoItems = [
    {
      title: 'Real-time Detection',
      description: 'Advanced X3D neural networks analyze video content in real-time',
      icon: Brain,
      className: 'col-span-2'
    },
    {
      title: 'High Accuracy',
      description: '84% validation accuracy with minimal false positives',
      icon: Eye,
      className: 'col-span-1'
    },
    {
      title: 'Lightning Fast',
      description: 'Process videos under 2 seconds',
      icon: Zap,
      className: 'col-span-1'
    },
    {
      title: 'Multiple Formats & Live Streams',
      description: 'Supports Live Streams as well as MP4, AVI, MOV, MKV and more video formats for pre-recorded videos',
      icon: Upload,
      className: 'col-span-2'
    }
  ]

  return (
    <div className="relative">
      <style>{`
        .bento-section {
          position: relative !important;
          z-index: 15 !important;
        }
        .global-spotlight {
          z-index: 16 !important;
        }
        .particle {
          z-index: 17 !important;
        }
        
        /* Enhanced hover animations */
        .interactive-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
        }
        .interactive-badge:hover {
          transform: translateY(-1px) scale(1.02);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
          background: hsl(var(--primary) / 0.1);
          border-color: hsl(var(--primary) / 0.3);
        }
        
        .stat-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
        }
        .stat-card:hover {
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
        }
        .stat-card:hover .stat-icon {
          transform: scale(1.1) rotate(5deg);
          color: hsl(var(--primary));
        }
        .stat-card:hover .stat-value {
          transform: scale(1.05);
          color: hsl(var(--primary));
        }
        
        .step-card {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: default;
        }
        .step-card:hover {
          transform: translateY(-6px) scale(1.03);
          box-shadow: 0 12px 30px rgba(0, 0, 0, 0.1);
        }
        .step-card:hover .step-number {
          background: linear-gradient(135deg, hsl(var(--primary)), hsl(var(--primary) / 0.8));
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          transform: scale(1.1);
        }
        .step-card:hover .step-title {
          color: hsl(var(--primary));
          transform: translateX(4px);
        }
        
        .demo-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .demo-button::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
          transition: left 0.5s;
        }
        .demo-button:hover::before {
          left: 100%;
        }
        .demo-button:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.5);
        }
        
        /* Icon animations */
        .stat-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .stat-value {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .step-number {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .step-title {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
      `}</style>
      {/* Hero Section with Light Rays */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0">
          <LightRays 
            count={8}
            color="hsl(var(--primary))"
            opacity={0.3}
            interactive={true}
          />
        </div>
        
        <div className="relative z-10 container mx-auto px-4 text-center">
          <AnimatedSection id="hero-content" delay={200}>
            <Badge variant="outline" className="mb-6 text-sm interactive-badge">
              AI-Powered • Real-time • Accurate
            </Badge>
            <h1 className="text-6xl md:text-8xl font-bold mb-8 bg-gradient-to-b from-foreground to-foreground/70 bg-clip-text text-transparent">
              Violence Detection
              <br />
              <span className="text-primary">Redefined</span>
            </h1>
            <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-3xl mx-auto leading-relaxed">
              Deploy cutting-edge X3D neural networks to detect aggression in videos with unprecedented accuracy and speed.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/upload">
                <MetallicPaint
                  colors={['hsl(var(--primary))', 'hsl(var(--primary)/0.8)']}
                  className="px-8 py-4 text-lg"
                >
                  Start Detection <ArrowRight className="ml-2 h-5 w-5" />
                </MetallicPaint>
              </Link>
              <Button 
                variant="outline" 
                size="lg" 
                className="px-8 py-4 text-lg demo-button"
                onClick={() => document.getElementById('demo-section')?.scrollIntoView({ behavior: 'smooth' })}
              >
                Watch Demo <Play className="ml-2 h-5 w-5" />
              </Button>
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-24 bg-muted/30">
        <div className="container mx-auto px-4">
          <AnimatedSection id="stats" delay={100}>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              {statsData.map((stat, index) => {
                const Icon = stat.icon
                return (
                  <div key={stat.label} className="text-center stat-card">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-primary/10 rounded-full mb-4">
                      <Icon className="h-8 w-8 text-primary stat-icon" />
                    </div>
                    <div className="text-3xl md:text-4xl font-bold text-primary mb-2 stat-value">
                      {stat.value}
                    </div>
                    <div className="text-sm text-muted-foreground">
                      {stat.label}
                    </div>
                  </div>
                )
              })}
            </div>
          </AnimatedSection>
        </div>
      </section>

      {/* Features Section with Magic Bento */}
      <section className="py-24">
        <div className="container mx-auto px-4">
          <AnimatedSection id="features-header" className="text-center mb-16">
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              Powerful Features
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Advanced AI capabilities designed for aggression detection
            </p>
          </AnimatedSection>

          <AnimatedSection id="features-bento" delay={200}>
          <div className="relative z-20">
            <MagicBento 
              cardData={bentoItems.map(item => ({
                color: "hsl(var(--card))",
                title: item.title,
                description: item.description,
                label: item.icon.name || "",
              }))}
              textAutoHide={true}
              enableStars={true}
              enableSpotlight={true}
              enableBorderGlow={true}
              enableTilt={true}
              enableMagnetism={true}
              clickEffect={true}
              spotlightRadius={300}
              particleCount={8}
              glowColor="255, 255, 255"
            />
          </div>
        </AnimatedSection>        
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 bg-muted/30">
        <div className="container mx-auto px-4">
          <AnimatedSection id="how-it-works-header" className="text-center mb-16">
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              How It Works
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Three simple steps to detect aggression in your video content
            </p>
          </AnimatedSection>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {[
              { step: '01', title: 'Upload Video', description: 'Drag and drop your video file or provide a file path' },
              { step: '02', title: 'AI Analysis', description: 'Our X3D model processes frames and detects aggression' },
              { step: '03', title: 'Get Results', description: 'Receive detailed analysis with timestamps and confidence scores' }
            ].map((item, index) => (
              <AnimatedSection 
                key={item.step} 
                id={`step-${index}`} 
                delay={index * 150}
                className="text-center step-card"
              >
                <div className="text-6xl font-bold text-primary/20 mb-4 step-number">
                  {item.step}
                </div>
                <h3 className="text-2xl font-semibold mb-4 step-title">{item.title}</h3>
                <p className="text-muted-foreground">{item.description}</p>
              </AnimatedSection>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-primary/5 to-primary/10"></div>
        <div className="relative container mx-auto px-4 text-center">
          <AnimatedSection id="cta-content">
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              Ready to Get Started?
            </h2>
            <p className="text-xl text-muted-foreground mb-12 max-w-2xl mx-auto">
              Join the future of video content moderation with AI-powered aggression detection
            </p>
            <div className="flex flex-col gap-4 items-center mb-8">
  <Link to="/upload">
    <Button size="lg" className="px-8 py-4 text-lg demo-button">
      Try It Now <ArrowRight className="ml-2 h-5 w-5" />
    </Button>
  </Link>
</div>
<Link to="/upload"></Link>
            {/* <Link to="/upload">
              <MetallicPaint
                colors={['hsl(var(--primary))', 'hsl(var(--primary)/0.8)']}
                className="px-12 py-6 text-xl"
              >
                Start Free Trial <ArrowRight className="ml-2 h-6 w-6" />
              </MetallicPaint>
            </Link> */}
          </AnimatedSection>
        </div>
      </section>

      {/* Demo Video Section */}
      <section id="demo-section" className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-muted/10 to-background"></div>
        <div className="relative container mx-auto px-4">
          <AnimatedSection id="demo-header" className="text-center mb-16">
            {/* <Badge variant="outline" className="mb-6 text-sm interactive-badge">
              <Play className="mr-2 h-4 w-4" />
              Live Demo
            </Badge> */}
            <h2 className="text-4xl md:text-6xl font-bold mb-6">
              See It In Action
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Watch our AI model detect aggression in real-time with sample footage
            </p>
          </AnimatedSection>

          <AnimatedSection id="demo-video" delay={200}>
            <div className="max-w-4xl mx-auto">
              <Card className="p-8 bg-gradient-to-br from-card/80 to-card/40 backdrop-blur-sm border-2 shadow-2xl">
                <div className="relative rounded-xl overflow-hidden bg-black/5 aspect-video">
                  <video 
                    className="w-full h-full object-cover rounded-lg"
                    controls
                    autoPlay
                    muted
                    loop
                    preload="metadata"
                    poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='450' viewBox='0 0 800 450'%3E%3Crect width='800' height='450' fill='%23f1f5f9'/%3E%3Cg fill='%236b7280'%3E%3Ccircle cx='400' cy='200' r='30'/%3E%3Cpolygon points='385,185 385,215 415,200'/%3E%3C/g%3E%3Ctext x='400' y='280' text-anchor='middle' fill='%236b7280' font-family='system-ui' font-size='16'%3EDemo Video - Violence Detection%3C/text%3E%3C/svg%3E"
                  >
                    <source src="/demo-video.mp4" type="video/mp4" />
                    Your browser does not support the video tag.
                  </video>
                  
                  {/* Video overlay for enhanced styling */}
                  <div className="absolute inset-0 rounded-lg ring-1 ring-black/5 pointer-events-none"></div>
                </div>
                
                <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-primary/5 rounded-lg">
                    <div className="text-2xl font-bold text-primary mb-1">Real-time</div>
                    <div className="text-sm text-muted-foreground">Processing Speed</div>
                  </div>
                  <div className="text-center p-4 bg-green-500/5 rounded-lg">
                    <div className="text-2xl font-bold text-green-600 mb-1">84%</div>
                    <div className="text-sm text-muted-foreground">Accuracy Rate</div>
                  </div>
                  <div className="text-center p-4 bg-blue-500/5 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600 mb-1">X3D</div>
                    <div className="text-sm text-muted-foreground">Neural Network</div>
                  </div>
                </div>
              </Card>
            </div>
          </AnimatedSection>
        </div>
      </section>
      
    </div>
    
  )
}

export default LandingPage
