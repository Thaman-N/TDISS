import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Moon, Sun, Palette, Shield, Menu, X } from 'lucide-react'

import {
  NavigationMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
  NavigationMenuTrigger,
  navigationMenuTriggerStyle,
} from "@/components/ui/navigation-menu"

const Navigation = ({ darkMode, toggleDarkMode, currentTheme, setTheme, availableThemes, getThemeName }) => {
  const location = useLocation()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const navItems = [
    { path: '/', label: 'Home' },
    { path: '/upload', label: 'Upload' },
    { path: '/dashboard', label: 'Dashboard' },
  ]

  const isActive = (path) => location.pathname === path

  return (
    <>
      <style>{`
        /* Enhanced Navigation Animations */
        .nav-logo {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .nav-logo:hover {
          transform: translateY(-1px) scale(1.02);
        }
        .nav-logo:hover .logo-icon {
          transform: rotate(10deg) scale(1.1);
          color: hsl(var(--primary));
          filter: drop-shadow(0 0 8px hsl(var(--primary) / 0.3));
        }
        .logo-text {
          background: linear-gradient(135deg, hsl(var(--primary)), hsl(var(--primary) / 0.8));
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .nav-logo:hover .logo-text {
          background: linear-gradient(135deg, hsl(var(--primary)), hsl(var(--primary) / 0.6));
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          filter: brightness(1.1);
        }
        
        .beta-badge {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .nav-logo:hover .beta-badge {
          transform: translateY(-2px) scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.3);
          background: hsl(var(--primary) / 0.05);
        }
        
        .nav-item {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .nav-item::before {
          content: '';
          position: absolute;
          bottom: 0;
          left: 50%;
          width: 0;
          height: 2px;
          background: hsl(var(--primary));
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          transform: translateX(-50%);
        }
        .nav-item:hover::before {
          width: 80%;
        }
        .nav-item:hover {
          transform: translateY(-1px);
          color: hsl(var(--primary));
        }
        .nav-item.active::before {
          width: 100%;
        }
        
        .control-button {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          overflow: hidden;
        }
        .control-button::before {
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
        .control-button:hover::before {
          width: 120%;
          height: 120%;
        }
        .control-button:hover {
          transform: translateY(-2px) scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
          border-color: hsl(var(--primary) / 0.2);
        }
        .control-button:hover .control-icon {
          transform: scale(1.1) rotate(5deg);
          color: hsl(var(--primary));
        }
        
        .mobile-toggle {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .mobile-toggle:hover {
          transform: translateY(-1px) scale(1.05);
          background: hsl(var(--primary) / 0.1);
        }
        .mobile-toggle:hover .mobile-icon {
          transform: scale(1.1);
          color: hsl(var(--primary));
        }
        
        .mobile-nav-item {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
          padding: 0.75rem 1rem;
          border-radius: 0.5rem;
          margin: 0.25rem 0;
        }
        .mobile-nav-item::before {
          content: '';
          position: absolute;
          left: 0;
          top: 0;
          width: 0;
          height: 100%;
          background: hsl(var(--primary) / 0.1);
          border-radius: 0.5rem;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .mobile-nav-item:hover::before {
          width: 100%;
        }
        .mobile-nav-item:hover {
          transform: translateX(4px);
          color: hsl(var(--primary));
        }
        .mobile-nav-item.active {
          color: hsl(var(--primary));
          background: hsl(var(--primary) / 0.1);
        }
        
        .dropdown-item {
          transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
          position: relative;
        }
        .dropdown-item:hover {
          transform: translateX(4px);
          background: hsl(var(--primary) / 0.05);
        }
        .dropdown-item.active {
          background: hsl(var(--primary) / 0.1);
          color: hsl(var(--primary));
        }
        
        /* Icon animations */
        .logo-icon {
          transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .logo-text {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .control-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .mobile-icon {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        /* Mobile menu animation */
        .mobile-menu {
          animation: slideDown 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        @keyframes slideDown {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
      `}</style>
      
      <nav className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto flex h-16 items-center justify-between px-4">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 nav-logo">
            <Shield className="h-8 w-8 text-primary logo-icon" />
            <span className="text-xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent logo-text">
              TDISS
            </span>
            <Badge variant="outline" className="text-xs beta-badge">
              Beta
            </Badge>
          </Link>

          {/* Desktop Navigation */}
          <NavigationMenu className="hidden md:flex">
            <NavigationMenuList>
              {navItems.map((item) => (
                <NavigationMenuItem key={item.path}>
                  <Link to={item.path}>
                    <NavigationMenuLink 
                      className={`${navigationMenuTriggerStyle()} nav-item ${isActive(item.path) ? 'active' : ''}`}
                    >
                      {item.label}
                    </NavigationMenuLink>
                  </Link>
                </NavigationMenuItem>
              ))}
            </NavigationMenuList>
          </NavigationMenu>

          {/* Theme Controls & Mobile Menu */}
          <div className="flex items-center space-x-2">
            {/* Theme Switcher */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="control-button">
                  <Palette className="h-4 w-4 control-icon" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                {availableThemes.map((theme) => (
                  <DropdownMenuItem
                    key={theme}
                    onClick={() => setTheme(theme)}
                    className={`dropdown-item ${currentTheme === theme ? 'active' : ''}`}
                  >
                    {getThemeName(theme)}
                    {currentTheme === theme && (
                      <Badge variant="outline" className="ml-auto text-xs">
                        Active
                      </Badge>
                    )}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Dark Mode Toggle */}
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleDarkMode}
              className="w-9 px-0 control-button"
            >
              {darkMode ? (
                <Sun className="h-4 w-4 control-icon" />
              ) : (
                <Moon className="h-4 w-4 control-icon" />
              )}
            </Button>

            {/* Mobile Menu Toggle */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden mobile-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? (
                <X className="h-4 w-4 mobile-icon" />
              ) : (
                <Menu className="h-4 w-4 mobile-icon" />
              )}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border mobile-menu">
            <div className="container mx-auto px-4 py-4 space-y-1">
              {navItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`block text-sm font-medium mobile-nav-item ${
                    isActive(item.path)
                      ? 'active'
                      : 'text-muted-foreground hover:text-primary'
                  }`}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        )}
      </nav>
    </>
  )
}

export default Navigation
