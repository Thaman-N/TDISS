import pytest
"""
Quick validation script to test system initialization.
This script validates that all constructors work correctly before running the full system.
"""

import sys
import os
import asyncio
import traceback

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

@pytest.mark.asyncio
async def test_system_initialization():
    """Test basic system initialization"""
    try:
        print("🔧 Testing System Initialization...")
        
        from integration_system import initialize_system
        
        # Simple test configuration
        config = {
            'model_path': '../models/rwf9425.pth',
            'device': 'cpu',  # Use CPU for testing
            'max_streams': 4,
            'max_batch_size': 2,
            'detection_threshold': 0.5,
            'max_rtsp_connections': 4,
            'bandwidth_limit_mbps': 10.0,
            'max_gpu_memory_gb': 2.0,
            'storage_path': './test_data',
            'max_storage_gb': 1,
            'db_path': './test_violence_events.db',
            'max_db_connections': 3,
            'discord_webhook_url': None,
            'batch_timeout': 2.0,
            'default_batch_size': 1
        }
        
        print("  Configuration prepared...")
        
        # Try to initialize
        success = await initialize_system(config)
        
        if success:
            print("✅ System initialization successful!")
            
            # Get the system and check status
            from integration_system import get_system
            system = await get_system()
            
            if system:
                status = await system.get_system_status()
                print(f"  System Status: {status}")
                
                # Cleanup
                from integration_system import cleanup_system
                await cleanup_system()
                print("  System cleaned up successfully")
                
                return True
            else:
                print("❌ System not available after initialization")
                return False
        else:
            print("❌ System initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error during initialization test: {e}")
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🧪 TDISS System Initialization Test")
    print("=" * 50)
    
    success = await test_system_initialization()
    
    if success:
        print("\n🎉 All tests passed! System is ready to start.")
    else:
        print("\n⚠️ Tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)